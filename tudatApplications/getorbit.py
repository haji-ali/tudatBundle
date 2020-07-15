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
        with Pool(len(runs), initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
            return p.map(fn, runs)
    return list(map(fn, runs))

__lib__ = npct.load_library("../lib/libSatellitePropagator.so",
                            __file__)

__arr_double__ = npct.ndpointer(dtype=np.double, flags='C_CONTIGUOUS')

__lib__.GetOrbit.restype = ct.c_uint32
__lib__.GetOrbit.argtypes = [__arr_double__, # init (size bodies x 6)
                             ct.c_uint32,  # bodies
                             ct.c_double, # simulationEndEpoch
                             ct.c_uint32,  # N
                             __arr_double__, # output (N x bodies x 6)
                             ct.c_bool
]


def test(y0, T, N):
    from scipy.integrate import odeint
    m = 3.986e+14 #3.986004418e14
    # function that returns dy/dt
    def model(y,t):
        assert(len(y)%6 == 0 and len(y.shape) == 1)
        y = y.reshape((-1, 6), order="C")
        dydt = np.zeros_like(y)
        dydt[:, 0:3] = y[:, 3:6]
        dydt[:, 3:6] = - m * y[:, :3] / np.sum(y[:, :3]**2, axis=1)[:,None]**(3./2.)
        return dydt.ravel(order="C")

    # time points
    t = np.linspace(0,T, N)

    # solve ODE
    return odeint(model,y0.ravel(order="C"),t).reshape((N, -1, 6), order="C")
    
def getOrbit(init, T, N):
    # init size is bodies x 6
    assert(len(init.shape) in (1,2) and init.shape[-1] == 6)
    N = int(N)
    bodies = 1 if len(init.shape)==1 else init.shape[0]
    output = np.empty(int((N+1)*np.prod(init.shape)), dtype=np.double)
    count = __lib__.GetOrbit(
        init.ravel(order="C"),
        ct.c_uint32(1 if len(init.shape)==1 else init.shape[0]),
        ct.c_double(T),
        ct.c_uint32(N),
        output, False)
    if count != len(output):   # Otherwise something is weird
        raise Exception("Expected {}, got {}".format(len(output), count))
    return output.reshape((N+1,) + init.shape, order="C")

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


def mlmc_l(data, L, M, M0):
    totalM = 0
    prob = 0
    sums = np.zeros((L, 4))
    for i in trange(0, int(np.ceil(M/M0))):
        init1 = np.random.multivariate_normal(data.mean1, data.C1, size=M0)
        init2 = np.random.multivariate_normal(data.mean2, data.C2, size=M0)
        c = None
        for ell in range(0, L):
            N = 2**ell
            res1 = getOrbit(init1, data.T, N)
            res2 = getOrbit(init2, data.T, N)
            dist = linDist(res1[:, :, :3], res2[:, :, :3])  # Should be NxM
            f = (np.sum(dist < data.radius, axis=0)>0).astype(np.float)
            sums[ell, 0] += np.sum(f, axis=0)
            sums[ell, 1] += np.sum(f**2, axis=0)
            if c is not None:
                sums[ell, 2] += np.sum(f-c, axis=0)
                sums[ell, 3] += np.sum((f-c)**2, axis=0)
                
            c = f
        totalM += M0
    return sums, totalM

def _do_mlmc_l(args):
    M,m,data,L,M0 = args
    return mlmc_l(data, L, M0, M0)

Data = namedtuple('Data', 'mean1 mean2 C1 C2 T radius')
data_dict = {k:np.array(v) for k,v in hau.load_file("objects.txt").items()}
data = Data(mean1=data_dict["Primary"][0, :], C1=data_dict["Primary"][1:, :],
            mean2=data_dict["Secondary"][0,:], C2=data_dict["Secondary"][1:, :],
            T=280800+21600, radius=15)

M, M0, L = 10000, 100, 19
result = pmap(_do_mlmc_l, [[data, L, M0] for i in range(int(np.ceil(M/M0)))])
sums, totalM = np.sum(np.array(result), axis=0)
np.savez("mlmc_l.npz", sums=sums, M=totalM)

# mlmc_l(data, 20, 10000, 100)

# res1 = test(mean1, T, N)[:, 0,:]
# res2 = test(mean2, T, N)[:, 0,:]
# t = np.linspace(0,T, N)
# dist = shortestDist(res1[:, :3], res2[:, :3])
# dist3 = linDist(res1[:, :3], res2[:, :3])
# dist2 = np.sqrt(np.sum((res1[:, :3]-res2[:, :3])**2, axis=1))
# plt.plot(t, dist)
# plt.plot(t, dist2)
# plt.plot(t, dist3)

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
