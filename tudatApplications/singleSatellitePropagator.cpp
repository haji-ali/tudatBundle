/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */
#include <sstream>
#include <limits>
#include <Tudat/SimulationSetup/tudatSimulationHeader.h>

struct PerturbedSettings {
    double mass;
    double referenceArea;
    double aerodynamicCoefficient;
    double referenceAreaRadiation;
    double radiationPressureCoefficient;

};

extern "C"
unsigned int GetOrbit(double* init, unsigned int bodies,
                      PerturbedSettings *pSettings,
                      double simulationEndEpoch, unsigned int N,
                      double* output, bool debug)
{
    using namespace tudat;
    using namespace tudat::simulation_setup;
    using namespace tudat::propagators;
    using namespace tudat::numerical_integrators;
    using namespace tudat::orbital_element_conversions;
    using namespace tudat::basic_mathematics;
    using namespace tudat::gravitation;
    using namespace tudat::numerical_integrators;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////     CREATE ENVIRONMENT AND VEHICLE       //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Load Spice kernels.
    spice_interface::loadStandardSpiceKernels( );

    // Set simulation time settings.
    const double simulationStartEpoch = 0.0;

    // Define body settings for simulation.
    std::vector< std::string > bodiesToCreate;
    bodiesToCreate.push_back( "Earth" );

    if (pSettings) {
        bodiesToCreate.push_back( "Sun" );
        bodiesToCreate.push_back( "Moon" );
        bodiesToCreate.push_back( "Mars" );
        bodiesToCreate.push_back( "Venus" );
    }

    // Create body objects.
    std::map< std::string, std::shared_ptr< BodySettings > > bodySettings;

    if (pSettings){
        bodySettings = getDefaultBodySettings( bodiesToCreate, simulationStartEpoch - 300.0, simulationEndEpoch + 300.0 );
        for( unsigned int i = 0; i < bodiesToCreate.size( ); i++ )
            {
                bodySettings[ bodiesToCreate.at( i ) ]->ephemerisSettings->resetFrameOrientation( "J2000" );
                bodySettings[ bodiesToCreate.at( i ) ]->rotationModelSettings->resetOriginalFrame( "J2000" );
            }
    }
    else{
        bodySettings = getDefaultBodySettings( bodiesToCreate );
        bodySettings[ "Earth" ]->ephemerisSettings = std::make_shared< ConstantEphemerisSettings >(
            Eigen::Vector6d::Zero( ) );
    }

    NamedBodyMap bodyMap = createBodies( bodySettings );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             CREATE VEHICLE            /////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Define propagator settings variables.
    SelectedAccelerationMap accelerationMap;
    std::vector< std::string > bodiesToPropagate;
    std::vector< std::string > centralBodies;

    // Define propagation settings.
    std::map< std::string, std::vector< std::shared_ptr< AccelerationSettings > > > accelerationsOfAsterix;

    if (pSettings){
        accelerationsOfAsterix[ "Earth" ].push_back( std::make_shared< SphericalHarmonicAccelerationSettings >( 5, 5 ) );

        accelerationsOfAsterix[ "Sun" ].push_back( std::make_shared< AccelerationSettings >(
                                                       basic_astrodynamics::central_gravity ) );
        accelerationsOfAsterix[ "Moon" ].push_back( std::make_shared< AccelerationSettings >(
                                                        basic_astrodynamics::central_gravity ) );
        accelerationsOfAsterix[ "Mars" ].push_back( std::make_shared< AccelerationSettings >(
                                                        basic_astrodynamics::central_gravity ) );
        accelerationsOfAsterix[ "Venus" ].push_back( std::make_shared< AccelerationSettings >(
                                                         basic_astrodynamics::central_gravity ) );

        accelerationsOfAsterix[ "Sun" ].push_back( std::make_shared< AccelerationSettings >(
                                                       basic_astrodynamics::cannon_ball_radiation_pressure ) );

        accelerationsOfAsterix[ "Earth" ].push_back( std::make_shared< AccelerationSettings >(
                                                         basic_astrodynamics::aerodynamic ) );
    }
    else{
        accelerationsOfAsterix[ "Earth" ].push_back( std::make_shared< AccelerationSettings >(
                                                         basic_astrodynamics::central_gravity ) );
    }

    Eigen::VectorXd systemInitialState(6 * bodies);
    for (unsigned int i=0;i<bodies;i++){
        std::stringstream ss;
        ss << "Body" << i << std::endl;
        const std::string& name= ss.str();

        bodyMap[ name ] = std::make_shared< simulation_setup::Body >( );
        if (pSettings){
            // Create spacecraft object.
            bodyMap[ name ]->setConstantBodyMass( pSettings->mass );

            // Create aerodynamic coefficient interface settings.
            std::shared_ptr< AerodynamicCoefficientSettings > aerodynamicCoefficientSettings =
                std::make_shared< ConstantAerodynamicCoefficientSettings >(
                    pSettings->referenceArea, pSettings->aerodynamicCoefficient * Eigen::Vector3d::UnitX( ), 1, 1 );

            // Create and set aerodynamic coefficients object
            bodyMap[ name ]->setAerodynamicCoefficientInterface(
                createAerodynamicCoefficientInterface( aerodynamicCoefficientSettings, name ) );

            // Create radiation pressure settings
            std::vector< std::string > occultingBodies;
            occultingBodies.push_back( "Earth" );
            std::shared_ptr< RadiationPressureInterfaceSettings > asterixRadiationPressureSettings =
                std::make_shared< CannonBallRadiationPressureInterfaceSettings >(
                    "Sun", pSettings->referenceAreaRadiation,
                    pSettings->radiationPressureCoefficient, occultingBodies );

            // Create and set radiation pressure settings
            bodyMap[ name ]->setRadiationPressureInterface(
                "Sun", createRadiationPressureInterface(
                    asterixRadiationPressureSettings, name, bodyMap ) );
        }
        
        accelerationMap[ name ] = accelerationsOfAsterix;
        bodiesToPropagate.push_back( name );
        centralBodies.push_back( "Earth" );

        int j = 6*i;
        systemInitialState(j+xCartesianPositionIndex) = init[j];
        systemInitialState(j+yCartesianPositionIndex) = init[j+1];
        systemInitialState(j+zCartesianPositionIndex) = init[j+2];

        systemInitialState(j+xCartesianVelocityIndex) = init[j+3];
        systemInitialState(j+yCartesianVelocityIndex) = init[j+4];
        systemInitialState(j+zCartesianVelocityIndex) = init[j+5];
    }

    // Finalize body creation.
    setGlobalFrameBodyEphemerides( bodyMap, "SSB", pSettings ? "J2000" : "ECLIPJ2000");

    basic_astrodynamics::AccelerationMap accelerationModelMap = createAccelerationModelsMap(
        bodyMap, accelerationMap, bodiesToPropagate, centralBodies );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             CREATE PROPAGATION SETTINGS            ////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::shared_ptr< TranslationalStatePropagatorSettings< double > > propagatorSettings =
        std::make_shared< TranslationalStatePropagatorSettings< double > >
        ( centralBodies, accelerationModelMap, bodiesToPropagate, systemInitialState, simulationEndEpoch );

    const double fixedStepSize = simulationEndEpoch / N;
    std::shared_ptr< IntegratorSettings< > > integratorSettings =
        std::make_shared< IntegratorSettings< > >
        ( rungeKutta4, 0.0, fixedStepSize );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             PROPAGATE ORBIT            ////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Create simulation object and propagate dynamics.
    SingleArcDynamicsSimulator< > dynamicsSimulator( bodyMap, integratorSettings, propagatorSettings );
    std::map< double, Eigen::VectorXd > integrationResult = dynamicsSimulator.getEquationsOfMotionNumericalSolution( );


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////        PROVIDE OUTPUT TO CONSOLE AND FILES           //////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (debug){
        Eigen::VectorXd finalIntegratedState = ( --integrationResult.end( ) )->second;

        std::cout << "First: " << ( --integrationResult.end( ) )->first << std::endl;
        std::cout << "Result: " << integrationResult.size() << std::endl;
        std::cout << "Size per time: " << finalIntegratedState.size() << std::endl;
        // Print the position (in km) and the velocity (in km/s) at t = 0.
        std::cout << "Single Earth-Orbiting Satellite Example." << std::endl <<
            "The initial position vector of Asterix is [km]:" << std::endl <<
            systemInitialState.segment( 0, 3 ) / 1E3 << std::endl <<
            "The initial velocity vector of Asterix is [km/s]:" << std::endl <<
            systemInitialState.segment( 3, 3 ) / 1E3 << std::endl;

        // Print the position (in km) and the velocity (in km/s) at t = 86400.
        std::cout << "After " << simulationEndEpoch <<
            " seconds, the position vector of Asterix is [km]:" << std::endl <<
            finalIntegratedState.segment( 0, 3 ) / 1E3 << std::endl <<
            "And the velocity vector of Asterix is [km/s]:" << std::endl <<
            finalIntegratedState.segment( 3, 3 ) / 1E3 << std::endl;
    }

    unsigned int count = 0;
    const unsigned int total_count = bodies*(N+1)*6;
    for (auto itr=integrationResult.begin(); itr != integrationResult.end();++itr){
        const Eigen::VectorXd &res = itr->second;
        for (int i=0; i < res.size();++i,++count)
            if (output && count < total_count)
                output[count] = res[i];
    }

    spice_interface::clearSpiceKernels();

    return count;
}

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b){

    assert(a.size() == b.size());
    std::vector<T> result(a.size());
    std::transform(a.begin(), a.end(), b.begin(),
                   result.begin(), std::plus<T>());
    return result;
}
template <typename T>
std::vector<T> operator*(double d, const std::vector<T>& a){
    std::vector<T> result(a.size());
    std::transform(a.begin(), a.end(), result.begin(),
                   [d](T x) { return x*d; });
    return result;
}

double MinDist_RK4(std::vector<double> x,
                   double T,
                   double max_h,
                   double min_h,
                   double radius2,
                   double adaptive_factor,
                   unsigned int& N){
    // TOOD:
    // 1. DONE fnRHS
    // 2. DONE CalcMinDist
    // 3. DONE Operators
    // 4. Complicate distance function?
    
    assert(x.size() == 12);
    const double m = 3.986004418e+14;
    auto fnRHS = [m](const std::vector<double>& x) {
                     std::vector<double> y(x.size());
                     for (int i=0;i<12;i+=6){
                         double norm = 0;
                         for (int j=i;j<i+3;j++)
                             norm += x[j]*x[j];
                         norm = std::pow(norm, 3./2.);
                         for (int j=i;j<i+3;j++){
                             y[j] = x[j+3];
                             y[j+3] = -m * x[j] / norm;
                         }
                     }
                     return y;
                 };
    
    auto fnDist2 = [m](const std::vector<double>& x) {
                      double norm = 0;
                      for (int j=0;j<3;j++){
                          double d = x[j] - x[6+j];
                          norm += d*d;
                      }
                      return norm;
                  };

    double min_dist = std::numeric_limits<double>::infinity();
    double t=0;
    std::vector<double> k[4];
    N = 0;
    while (t<T && min_dist > radius2){
        // Figure out step size
        double h = max_h;
        if (min_h < max_h && adaptive_factor > 0){
            double next_dist2 = fnDist2(x + min_h * x);
            if (next_dist2 < adaptive_factor * radius2){
                while (h > min_h){
                    double try_dist = fnDist2(x + h * x);
                    if (try_dist <= next_dist2)
                        break;
                    h /= 2.;
                }
            }
        }
        h = std::min(h, T-t);

        // Advance with RK_4
        k[0] = fnRHS(x);
        k[1] = fnRHS(x + 0.5 * h * k[0]);
        k[2] = fnRHS(x + 0.5 * h * k[1]);
        k[3] = fnRHS(x +       h * k[2]);
        x = x + (h/6.) * (k[0] + 2.*k[1] + 2.*k[2] + k[3]);
        t += h;
        min_dist = std::min(fnDist2(x), min_dist);
        
        N++;
    }
    return min_dist;
}

extern "C"
void GetMinDist(double* init1, // Size * bodies * h
                double* init2, // Size * bodies * h
                unsigned int bodies,
                double simulationEndEpoch,
                double max_h, double min_h,
                double radius2,
                double adaptive_factor,
                double* distances, // bodies
                unsigned int *N
    )
{
    for (unsigned int i=0;i<bodies;i++){
        std::vector<double> x (init1+6*i, init1+6*(i+1));
        x.insert(x.end(), init2+6*i, init2+6*(i+1));

        distances[i] =
            MinDist_RK4(x, simulationEndEpoch,
                        max_h, min_h, radius2, adaptive_factor, N[i]);
    }
}

#ifndef __WITHOUT_MAIN
//! Execute propagation of orbit of Asterix around the Earth.
int main( )
{
    double init[] = {-33552459.274056, -23728303.048015, 0.0,
                     -1828.997179397, 2534.1074695609, 0.0};

    std::cout << "----------- Unperturbed" << std::endl;
    GetOrbit(init,
             1,
             NULL,
             280800,
             2808,
             NULL, true);

    PerturbedSettings p = {4,4,1.2,4,1.2};
    std::cout << "----------- perturbed" << std::endl;
    GetOrbit(init,
             1,
             &p,
             280800,
             2808,
             NULL, true);
}
#endif
