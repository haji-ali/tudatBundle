#!/usr/bin/env python3
import ctypes as ct
import numpy.ctypeslib as npct
import numpy as np
import time
import hau
from collections import namedtuple

from tqdm import trange, tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt

parallel=True
def pmap(fn, runs):
    runs = [[len(runs), i] + r for i, r in enumerate(runs)]
    if parallel:
        with Pool() as p:
            return list(tqdm(p.imap(fn, runs), total=len(runs)))
    return list(map(fn, runs))

__lib__ = npct.load_library("../lib/libSatellitePropagator.so",
                            __file__)

__arr_double__ = npct.ndpointer(dtype=np.double, flags='C_CONTIGUOUS')
__arr_uint32__ = npct.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS')


class PerturbedSettings(ct.Structure):
    _fields_ = [("mass", ct.c_double),
                ("referenceArea", ct.c_double),
                ("aerodynamicCoefficient", ct.c_double),
                ("referenceAreaRadiation", ct.c_double),
                ("radiationPressureCoefficient", ct.c_double)]

    def __init__(self, mass=4,
                 referenceArea=4,
                 aerodynamicCoefficient=1.2,
                 referenceAreaRadiation=4,
                 radiationPressureCoefficient=1.2):
        self.mass = mass
        self.referenceArea = referenceArea
        self.aerodynamicCoefficient = aerodynamicCoefficient
        self.referenceAreaRadiation = referenceAreaRadiation
        self.radiationPressureCoefficient = radiationPressureCoefficient

__lib__.GetOrbit.restype = ct.c_uint32
__lib__.GetOrbit.argtypes = [__arr_double__, # init (size bodies x 6)
                             ct.c_uint32,  # bodies
                             ct.POINTER(PerturbedSettings),
                             ct.c_double, # simulationEndEpoch
                             ct.c_uint32,  # N
                             __arr_double__, # output (N x bodies x 6)
                             ct.c_bool
]

__lib__.GetMinDist.restype = ct.c_uint32
__lib__.GetMinDist.argtypes = [__arr_double__, # init1 (size bodies x 6)
                               __arr_double__, # init2 (size bodies x 6)
                               ct.c_uint32,  # bodies
                               ct.c_double, # simulationEndEpoch
                               ct.c_double,  # max_h
                               ct.c_double,  # min_h
                               ct.c_double,  # radius2
                               ct.c_double,  # adaptive_factor
                               __arr_double__, # output (bodies)
                               __arr_uint32__, # output_N (bodies)
]


def RK4_step(t, x, h, fnRHS):
    k1 = fnRHS(t, x)
    k2 = fnRHS(t, x+.5*h*k1)
    k3 = fnRHS(t, x+.5*h*k2)
    k4 = fnRHS(t, x+h*k3)
    return x + h*(k1+2*k2+2*k3+k4)/6

def RK4(x, fnStep):
    xi = [x]
    t = [0]
    while True:
        h, x = fnStep(t[-1], xi[-1])
        if h <= 0:
            break
        xi.append(x)
        t.append(t[-1]+h)
    return t, xi

def getOrbit_RK4(init, T, N, radius=0, h_min=None):
    m = 3.986004418e+14
    
    # function that returns dy/dt
    def model(t, y):
        # assert(len(y)%6 == 0 and len(y.shape) == 1)
        # y = y.reshape((-1, 6), order="C")
        dydt = np.zeros_like(y)
        dydt[..., 0:3] = y[..., 3:6]
        dydt[..., 3:6] = - m * y[..., :3] / np.sum(y[..., :3]**2, axis=-1)[...,None]**(3./2.)
        return dydt#.ravel(order="C")

    h0 = T/N
    if h_min is None:
        step = lambda t, x, h=h0, mod=model: (min(T-t, h), RK4_step(t, x, h, mod))
    else:
        def step(t, y):
            h = h0
            old_dist = np.maximum(np.sum((y[0,...,0:3] - y[1,...,0:3])**2, axis=-1), radius)

            yn = y[..., 0:3] + h_min*y[..., 3:6]
            min_dist = np.sum((yn[0,...,0:3] - yn[1,...,0:3])**2, axis=-1)
            
            if np.all(min_dist <= radius*10):    # Getting closer
                while h > h_min:
                    yn = y[..., 0:3] + h*y[..., 3:6]
                    new_dist = np.sum((yn[0,...,0:3] - yn[1,...,0:3])**2, axis=-1)
                    if np.all(new_dist <= min_dist):
                        break
                    h /= 2
                    
            h = min(T-t, h)
            return h, RK4_step(t, y, h, model)

    # time points
    t, steps = RK4(init, step)
    # solve ODE
    return np.array(t), np.array(steps)
    
def getOrbit(init, T, N, perturbed_settings=None):
    # init size is bodies x 6
    assert(len(init.shape) in (1,2) and init.shape[-1] == 6)
    N = int(N)
    bodies = 1 if len(init.shape)==1 else init.shape[0]
    output = np.empty(int((N+1)*np.prod(init.shape)), dtype=np.double)

    count = __lib__.GetOrbit(
        init.ravel(order="C"),
        ct.c_uint32(bodies),
        ct.byref(perturbed_settings) if perturbed_settings is not None else ct.POINTER(PerturbedSettings)(),
        ct.c_double(T),
        ct.c_uint32(N),
        output, False)
    
    if count != len(output):   # Otherwise something is weird
        raise Exception("N={}, Expected {}, got {}".format(N, len(output), count))
    return output.reshape((N+1,) + init.shape, order="C")

def getMinDist(init1, init2, T, max_h, min_h, radius2, adaptive_factor):
    # init size is bodies x 6
    assert(len(init1.shape) in (1,2) and init1.shape[-1] == 6)
    assert(init2.shape == init1.shape)

    bodies = 1 if len(init1.shape)==1 else init1.shape[0]
    output = np.empty(bodies, dtype=np.double)
    output_N = np.empty(bodies, dtype=np.uint32)
    __lib__.GetMinDist(init1.ravel(order="C"), init2.ravel(order="C"), ct.c_uint32(bodies),
                       ct.c_double(T), ct.c_double(max_h), ct.c_double(min_h),
                       ct.c_double(radius2),
                       ct.c_double(adaptive_factor),
                       output, output_N)
    return output, output_N
    

def gridDist(S1, S2):
    """Returns the distance between the time series S1 and S2.
    S1 and S2 have size N x ... x 3
    Only considers the grid points
    return array of size N x ..."""
    return np.sqrt(np.sum((S1-S2)**2, axis=-1))

def linDist(S1, S2):
    """Returns the distance between the time series S1 and S2.
    S1 and S2 have size N x ... x 3
    Assumes linear interpolation between points and that the objects
    move at constant speed between points
    return array of size N x ..."""
    p1 = S1[:-1, ...]
    q1 = S1[1:, ...]

    p2 = S2[:-1, ...]
    q2 = S2[1:, ...]

    u = p1-p2
    v = (q1-p1) - (q2-p2)
    t = -np.sum(u, axis=-1) / np.sum(v, axis=-1)
    mid = u + t[..., None] * (t[..., None] > 0) * (t[..., None] < 1) * v

    m1 = np.sum(u**2, axis=-1)
    return np.sqrt(
        np.concatenate((m1[0:1, ...],
                        np.minimum(m1,np.sum((q1-q2)**2, axis=-1),
                                   np.sum((mid)**2, axis=-1)))))

def shortestDist(S1, S2):
    """Returns the distance between the time series S1 and S2.
    S1 and S2 have size N x ... x 3
    Assumes linear interpolation between points but no assumption about the
    velocity between grid points (hence the shortest distance is between segments is returned)
    return array of size N x ..."""
    # http://geomalgorithms.com/a07-_distance.html
    SMALL_NUM = 1e-8 # anything that avoids division overflow
    
    def dot(A, B):
        return np.sum(A * B, axis=-1)

    u = S1[1:, ...] - S1[:-1, ...]
    v = S2[1:, ...] - S2[:-1, ...]
    w = S1[:-1, ...] - S2[:-1, ...]
    a = dot(u,v)         # always >= 0
    b = dot(u,v)
    c = dot(v,v)         # always >= 0
    d = dot(u,w)
    e = dot(v,w)
    D = a*c - b*b        # always >= 0
    sD, tD = D.copy(), D.copy()

    sN = (b*e - c*d)
    tN = (a*e - b*d)

    cond1 = D < SMALL_NUM
    cond2 = np.logical_and(np.logical_not(cond1), sN<0)
    cond3 = np.logical_and(np.logical_not(cond1),
                           sN>=0,sN>sD)

    sN[cond1], tN[cond1] = 0., e[cond1]
    sD[cond1], tD[cond1] = 1., c[cond1]

    sN[cond2], tN[cond2] = 0., e[cond2]
    tD[cond2]            =     c[cond2]
    

    sN[cond3], tN[cond3] = sD[cond3], e[cond3] + b[cond3]
    tD[cond3]            = c[cond3]

    cond1 = tN<0
    cond2 = np.logical_and(cond1, -d<0)
    cond3 = np.logical_and(cond1, -d>a)
    cond4 = np.logical_and(cond1, np.logical_and(-d>=0, -d<=a))
    cond5 = tN > tD
    cond6 = np.logical_and(cond5, (-d + b) < 0.0)
    cond7 = np.logical_and(cond5, (-d + b) > a)
    cond8 = np.logical_and(cond5, np.logical_and((-d + b) >= 0.0, (-d + b) <= a))
    
    tN[cond1] = 0
    sN[cond2] = 0
    sN[cond3] = sD[cond3]
    sN[cond4] = -d[cond4]
    sD[cond4] = a[cond4]

    tN[cond5] = tD[cond5]
    sN[cond6] = 0
    sN[cond7] = sD[cond7]
    sN[cond8] = -d[cond8] + b[cond8]
    sD[cond8] = a[cond8]

    # finally do the division to get sc and tc
    cond1 = np.abs(abs(sN) >= SMALL_NUM)
    cond2 = np.abs(abs(sN) >= SMALL_NUM)
    sc = np.zeros(sN.shape)
    tc = np.zeros(tN.shape)
    sc[cond1] = sN[cond1]/sD[cond1]
    tc[cond2] = tN[cond2]/tD[cond2]

    # get the difference of the two closest points
    dP = w + (sc[..., None] * u) - (tc[..., None] * v)  # =  S1(sc) - S2(tc)
    dP = np.concatenate((S2[0:1, ...]-S1[0:1, ...], dP))  # Add first point
    return np.sqrt(np.sum(dP**2, axis=-1))   # return the closest distance


def mlmc_l(data, L, M0):
    prob = 0
    sums = np.zeros((L, 5))
    init1 = np.random.multivariate_normal(data.mean1, data.C1, size=M0)
    init2 = np.random.multivariate_normal(data.mean2, data.C2, size=M0)
    c = None
    for ell in range(0, L):
        N = 64 * 2**ell
        _, res1 = getOrbit_RK4(init1, data.T, N)
        _, res2 = getOrbit_RK4(init2, data.T, N)
        dist = linDist(res1[:, :, :3], res2[:, :, :3])
        f = (np.sum(dist < data.radius, axis=0)>0).astype(np.float)
        sums[ell, 0] += np.sum(f, axis=0)
        sums[ell, 1] += np.sum(f**2, axis=0)
        if c is not None:
            sums[ell, 2] += np.sum(f-c, axis=0)
            sums[ell, 3] += np.sum((f-c)**2, axis=0)
        c = f
    return sums, M0

def mlmc_l_2(data, L, M0, adaptive_factor):
    prob = 0
    sums = np.zeros((L, 6))
    init1 = np.random.multivariate_normal(data.mean1, data.C1, size=M0)
    init2 = np.random.multivariate_normal(data.mean2, data.C2, size=M0)
    c = None
    for ell in range(0, L):
        N_min = 64 * 2**ell
        N_max = 64 * 2**np.maximum(L, ell+3)
        max_h = data.T / N_min
        min_h = data.T / N_max

        D, N = getMinDist(init1, init2, data.T, max_h, min_h,
                          data.radius**2, adaptive_factor)
        f = (D < data.radius**2).astype(np.float)
        sums[ell, 0] += np.sum(N)
        sums[ell, 1] += np.sum(f, axis=0)
        sums[ell, 2] += np.sum(f**2, axis=0)
        if c is not None:
            sums[ell, 3] += np.sum(f-c, axis=0)
            sums[ell, 4] += np.sum((f-c)**2, axis=0)
        c = f
    return sums, M0

def _do_mlmc_l(args):
    M,m,data,L,M0,adaptive_factor = args
    return mlmc_l_2(data, L, M0, adaptive_factor)

Data = namedtuple('Data', 'mean1 mean2 C1 C2 T radius')
data_dict = {k:np.array(v) for k,v in hau.load_file("objects.txt").items()}
data = Data(mean1=data_dict["Primary"][0, :], C1=data_dict["Primary"][1:, :],
            mean2=data_dict["Secondary"][0,:], C2=data_dict["Secondary"][1:, :],
            T=280800+21600, radius=15)

if __name__ == "__main__":
    M, M0, L = 1000000, 100, 18
    result = pmap(_do_mlmc_l, [[data, L, M0, 0] for i in range(int(np.ceil(M/M0)))])
    sums, totalM = np.sum(np.array(result, dtype=object), axis=0)
    np.savez("mlmc_l_nonadaptive.npz", sums=sums, M=totalM)

    result = pmap(_do_mlmc_l, [[data, L, M0, 100] for i in range(int(np.ceil(M/M0)))])
    sums, totalM = np.sum(np.array(result, dtype=object), axis=0)
    np.savez("mlmc_l_adaptive.npz", sums=sums, M=totalM)

    # mlmc_l(data, 20, 10000, 100)
    ps = PerturbedSettings()
    for i in range(6, 14):
        N = 2**i
        res1 = getOrbit(data.mean1, data.T, N, ps)[:,:]
        res2 = getOrbit(data.mean2, data.T, N, ps)[:,:]
        t = np.linspace(0, data.T, res1.shape[0])
        dist = linDist(res1[:, :3], res2[:, :3])
        plt.plot(t, dist)
    
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # xyz1 = res1[:2, :3]
    # xyz2 = res2[:2, :3]
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(xyz1[:, 0], xyz1[:, 1], xyz1[:, 2])
    # ax.plot(xyz2[:, 0], xyz2[:, 1], xyz2[:, 2])

    # totalM = 0
    # prob = 0
    # for i in trange(0, M//100):
    #     init1 = np.random.multivariate_normal(mean1, C1, size=M0)
    #     init2 = np.random.multivariate_normal(mean2, C2, size=M0)

    #     res1 = getOrbit(init1, T, N)
    #     res2 = getOrbit(init2, T, N)
    #     dist = linDist(res1[:, :, :3], res2[:, :, :3])
    #     prob += np.sum(np.cumsum(dist < radius, axis=0)>0, axis=1)
    #     totalM += M0

    # plt.plot(np.linspace(0, T, N+1), prob/totalM);
    # plt.show()

    # work = np.zeros(6)
    # for l in trange(len(work)):
    #     tStart = time.time()
    #     N = 10 * 2**l

    #     res1 = getOrbit(init1, T, N)
    #     res2 = getOrbit(init2, T, N)
    #     dist = shortestDistance(res1[:, :, :3], res2[:, :, :3])  # Should be NxM

    #     np.mean(np.cumsum(dist<15, axis=0)>0, axis=1)

    #     work[l] = time.time()-tStart
    #
    #
    # hau.load("getorbit", "*")
    # M0 = 100
    # init1 = np.random.multivariate_normal(data.mean1, data.C1, size=M0)
    # init2 = np.random.multivariate_normal(data.mean2, data.C2, size=M0)

    # for i in range(1, 10):
    #     N_min = 64 * 2**i
    #     N_max = 64 * 2**10
    #     max_h = data.T / N_min
    #     min_h = data.T / N_max
    #     D, N = getMinDist(init1, init2, data.T, max_h, min_h, data.radius**2, 100)
    #     print(N_min, np.mean(N), N_max)
