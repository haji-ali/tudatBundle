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
#include <Tudat/SimulationSetup/tudatSimulationHeader.h>

extern "C"
unsigned int GetOrbit(double* init, unsigned int bodies,
                      double simulationEndEpoch, unsigned int N,
                      double* output, bool debug) {
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////            USING STATEMENTS              //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    using namespace tudat;
    using namespace tudat::simulation_setup;
    using namespace tudat::propagators;
    using namespace tudat::numerical_integrators;
    using namespace tudat::orbital_element_conversions;
    using namespace tudat::basic_mathematics;
    using namespace tudat::unit_conversions;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////     CREATE ENVIRONMENT AND VEHICLE       //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Load Spice kernels.
    spice_interface::loadStandardSpiceKernels( );

    // Create body objects.
    std::vector< std::string > bodiesToCreate;
    bodiesToCreate.push_back( "Earth" );
    std::map< std::string, std::shared_ptr< BodySettings > > bodySettings =
            getDefaultBodySettings( bodiesToCreate );
    bodySettings[ "Earth" ]->ephemerisSettings = std::make_shared< ConstantEphemerisSettings >(
                Eigen::Vector6d::Zero( ) );

    // Create Earth object
    NamedBodyMap bodyMap = createBodies( bodySettings );
    std::vector< std::string > bodiesToPropagate;
    std::vector< std::string > centralBodies;
    SelectedAccelerationMap accelerationMap;
    // Define propagation settings.
    std::map< std::string, std::vector< std::shared_ptr< AccelerationSettings > > > accelerationsOfAsterix;
    accelerationsOfAsterix[ "Earth" ].push_back( std::make_shared< AccelerationSettings >(
                                                     basic_astrodynamics::central_gravity ) );

    Eigen::VectorXd systemInitialState(6 * bodies);

    // Create spacecraft object.
    for (unsigned int i=0;i<bodies;i++){
        centralBodies.push_back( "Earth" );

        std::stringstream ss;
        ss << "Body" << i << std::endl;
        bodyMap[ ss.str() ] = std::make_shared< simulation_setup::Body >( );
        bodiesToPropagate.push_back(ss.str());
        accelerationMap[ ss.str() ] = accelerationsOfAsterix;

        int j = 6*i;
        systemInitialState(j+xCartesianPositionIndex) = init[j];
        systemInitialState(j+yCartesianPositionIndex) = init[j+1];
        systemInitialState(j+zCartesianPositionIndex) = init[j+2];

        systemInitialState(j+xCartesianVelocityIndex) = init[j+3];
        systemInitialState(j+yCartesianVelocityIndex) = init[j+4];
        systemInitialState(j+zCartesianVelocityIndex) = init[j+5];
        
    }

    // Finalize body creation.
    setGlobalFrameBodyEphemerides( bodyMap, "SSB", "ECLIPJ2000" );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////            CREATE ACCELERATIONS          //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Create acceleration models and propagation settings.
    basic_astrodynamics::AccelerationMap accelerationModelMap = createAccelerationModelsMap(
                bodyMap, accelerationMap, bodiesToPropagate, centralBodies );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             CREATE PROPAGATION SETTINGS            ////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Set initial conditions for the Asterix satellite that will be propagated in this simulation.
    // The initial conditions are given in Keplerian elements and later on converted to Cartesian
    // elements.

    // // Set Keplerian elements for Asterix.
    // Eigen::Vector6d asterixInitialStateInKeplerianElements;
    // asterixInitialStateInKeplerianElements( semiMajorAxisIndex ) = 7500.0E3;
    // asterixInitialStateInKeplerianElements( eccentricityIndex ) = 0.1;
    // asterixInitialStateInKeplerianElements( inclinationIndex ) = convertDegreesToRadians( 85.3 );
    // asterixInitialStateInKeplerianElements( argumentOfPeriapsisIndex ) =
    //         convertDegreesToRadians( 235.7 );
    // asterixInitialStateInKeplerianElements( longitudeOfAscendingNodeIndex ) =
    //         convertDegreesToRadians( 23.4 );
    // asterixInitialStateInKeplerianElements( trueAnomalyIndex ) = convertDegreesToRadians( 139.87 );

    // // Convert Asterix state from Keplerian elements to Cartesian elements.
    // double earthGravitationalParameter = bodyMap.at( "Earth" )->getGravityFieldModel( )->getGravitationalParameter( );
    // Eigen::VectorXd systemInitialState = convertKeplerianToCartesianElements(
    //             asterixInitialStateInKeplerianElements,
    //             earthGravitationalParameter );
    //
    // std::cout << "Earth Gravitational parameter: " << earthGravitationalParameter << std::endl;


    // Create propagator settings.
    std::shared_ptr< TranslationalStatePropagatorSettings< double > > propagatorSettings =
            std::make_shared< TranslationalStatePropagatorSettings< double > >
            ( centralBodies, accelerationModelMap, bodiesToPropagate, systemInitialState, simulationEndEpoch );

    // Create numerical integrator settings.
    double simulationStartEpoch = 0.0;
    const double fixedStepSize = simulationEndEpoch / N;
    std::shared_ptr< IntegratorSettings< > > integratorSettings =
            std::make_shared< IntegratorSettings< > >
            ( rungeKutta4, simulationStartEpoch, fixedStepSize );

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
    // std::string outputSubFolder = "UnperturbedSatelliteExample/";

    // // Write satellite propagation history to file.
    // input_output::writeDataMapToTextFile( integrationResult,
    //                                       "singleSatellitePropagationHistory.dat",
    //                                       tudat_applications::getOutputPath( ) + outputSubFolder,
    //                                       "",
    //                                       std::numeric_limits< double >::digits10,
    //                                       std::numeric_limits< double >::digits10,
    //                                       "," );

    // Final statement.
    // The exit code EXIT_SUCCESS indicates that the program was successfully executed.
    
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

#ifndef __WITHOUT_MAIN
//! Execute propagation of orbit of Asterix around the Earth.
int main( )
{
    double init[] = {-33552459.274056, -23728303.048015, 0.0,
                     -1828.997179397, 2534.1074695609, 0.0};
    GetOrbit(init,
             1,
             280800,
             2808,
             NULL, true);
}
#endif
