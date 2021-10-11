#include <odri_control_interface/calibration.hpp>
#include <odri_control_interface/robot.hpp>
#include <odri_control_interface/utils.hpp>

#include "qrw/Types.h"
#include "qrw/Params.hpp"
#include "qrw/Controller.hpp"
#include "qrw/FakeRobot.hpp"

using namespace odri_control_interface;

#include <iostream>
#include <stdexcept>

//int put_on_the_floor(std::shared_ptr<Robot> robot, Vector12 const& q_init)
int put_on_the_floor(FakeRobot *robot, Vector12 const& q_init)
{
    /*Make the robot go to the default initial position and wait for the user
    to press the Enter key to start the main control loop

    Args:
        device (robot wrapper): a wrapper to communicate with the robot
        q_init (array): the default position of the robot
    */

    printf("PUT ON THE FLOOR\n");

    double Kp_pos = 6.;
    double Kd_pos = 0.3;

    robot->joints->SetPositionGains(Kp_pos * Vector12::Ones());
    robot->joints->SetVelocityGains(Kd_pos * Vector12::Ones());
    robot->joints->SetDesiredPositions(q_init);
    robot->joints->SetDesiredVelocities(Vector12::Zero());
    robot->joints->SetTorques(Vector12::Zero());

    /* CONVERT TO C++
    i = threading.Thread(target=get_input)
    i.start()
    print("Put the robot on the floor and press Enter")

    while i.is_alive():
        device.parse_sensor_data()
        device.send_command_and_wait_end_of_cycle(params.dt_wbc)
    */
    // USE robot->ParseSensorData();
    // USE robot->SendCommandAndWaitEndOfCycle(params.dt_wbc);

    return 0;
}

int main()
{
    nice(-20);  // Give the process a high priority.

    // Object that holds all controller parameters
    Params params = Params();

    // Define the robot from a yaml file.
    // std::shared_ptr<Robot> robot = RobotFromYamlFile(CONFIG_SOLO12_YAML);
    FakeRobot* robot = new FakeRobot();

    // Store initial position data.
    Vector12 des_pos;
    des_pos << 0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7,
        +1.4;

    // Initialization of variables
    Controller controller; // Main controller
    controller.initialize(params);
    std::thread checking_thread(check_memory); // spawn new thread that calls check_memory()
    int k_loop = 0;

    // Initialize the communication, session, joints, wait for motors to be ready
    // and run the joint calibration.
    robot->Initialize(des_pos);
    robot->joints->SetZeroCommands();
    robot->ParseSensorData();

    // Main loop
    while ((!robot->IsTimeout()) && (k_loop < params.N_SIMULATION-2) && (!controller.error))
    {
        // Parse sensor data from the robot
        robot->ParseSensorData();

        // Run the main controller
        controller.compute(robot);

        // Check that the initial position of actuators is not too far from the
        // desired position of actuators to avoid breaking the robot
        if (k_loop <= 10)
        {
            Vector12 pos = robot->joints->GetPositions();
            if ((controller.q_des - pos).cwiseAbs().maxCoeff() > 0.15)
            {
                std::cout << "DIFFERENCE: " << (controller.q_des - pos).transpose() << std::endl;
                std::cout << "q_des: " << controller.q_des.transpose() << std::endl;
                std::cout << "q_mes: " << pos.transpose() << std::endl;
                break;
            }
        }

        // Send commands to the robot
        robot->joints->SetPositionGains(Vector12::Zero());
        robot->joints->SetVelocityGains(Vector12::Zero());
        robot->joints->SetDesiredPositions(Vector12::Zero());
        robot->joints->SetDesiredVelocities(Vector12::Zero());
        robot->joints->SetTorques(Vector12::Zero());

        // Checks if the robot is in error state (that is, if any component
        // returns an error). If there is an error, the commands to send
        // are changed to send the safety control.
        robot->SendCommandAndWaitEndOfCycle(params.dt_wbc);

        k_loop++;
        if (k_loop % 1000 == 0)
        {
            std::cout << "Joints: ";
            robot->joints->PrintVector(robot->joints->GetPositions());
            std::cout << std::endl;
        }

        break;
    }

    // Close parallel thread
    stop_thread();
    checking_thread.join();
    std::cout << "Parallel thread closed" << std::endl ;

    // DAMPING TO GET ON THE GROUND PROGRESSIVELY *********************
    double t = 0.0;
    double t_max = 2.5;
    while ((!robot->IsTimeout()) && (t < t_max))
    {
        // Parse sensor data from the robot
        robot->ParseSensorData();

        // Send commands to the robot
        robot->joints->SetPositionGains(Vector12::Zero());
        robot->joints->SetVelocityGains(0.1 * Vector12::Ones());
        robot->joints->SetDesiredPositions(Vector12::Zero());
        robot->joints->SetDesiredVelocities(Vector12::Zero());
        robot->joints->SetTorques(Vector12::Zero());

        // Checks if the robot is in error state (that is, if any component
        // returns an error). If there is an error, the commands to send
        // are changed to send the safety control.
        robot->SendCommandAndWaitEndOfCycle(params.dt_wbc);

        t += params.dt_wbc;
    }
    // FINAL SHUTDOWN *************************************************

    // Whatever happened we send 0 torques to the motors.
    robot->joints->SetZeroCommands();
    robot->SendCommandAndWaitEndOfCycle(params.dt_wbc);

    if (robot->IsTimeout())
    {
        printf("Masterboard timeout detected.");
        printf("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.");
    }

    return 0;
}
