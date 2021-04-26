#include "qrw/Params.hpp"

using namespace yaml_control_interface;

Params::Params()
    : interface("")
    , SIMULATION(false)
    , LOGGING(false)
    , PLOTTING(false)
    , dt_wbc(0.0)
    , N_gait(0)
    , envID(0)
    , velID(0)
    , dt_mpc(0.0)
    , T_gait(0.0)
    , T_mpc(0.0)
    , N_SIMULATION(0)
    , type_MPC(false)
    , use_flat_plane(false)
    , predefined_vel(false)
    , kf_enabled(false)
    , enable_pyb_GUI(false)
{
    initialize(CONFIG_SOLO12_YAML);
}

void Params::initialize(const std::string& file_path)
{
    // Load YAML file
    assert_file_exists(file_path);
    YAML::Node param = YAML::LoadFile(file_path);

    // Check if YAML node is detected and retrieve it
    assert_yaml_parsing(param, file_path, "robot");
    const YAML::Node& robot_node = param["robot"];

    // Retrieve robot parameters
    assert_yaml_parsing(robot_node, "robot", "interface");
    interface = robot_node["interface"].as<std::string>();

    assert_yaml_parsing(robot_node, "robot", "SIMULATION");
    SIMULATION = robot_node["SIMULATION"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "LOGGING");
    LOGGING = robot_node["LOGGING"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "PLOTTING");
    PLOTTING = robot_node["PLOTTING"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "dt_wbc");
    dt_wbc = robot_node["dt_wbc"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "N_gait");
    N_gait = robot_node["N_gait"].as<int>();

    assert_yaml_parsing(robot_node, "robot", "envID");
    envID = robot_node["envID"].as<int>();

    assert_yaml_parsing(robot_node, "robot", "velID");
    velID = robot_node["velID"].as<int>();

    assert_yaml_parsing(robot_node, "robot", "dt_mpc");
    dt_mpc = robot_node["dt_mpc"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "T_gait");
    T_gait = robot_node["T_gait"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "T_mpc");
    T_mpc = robot_node["T_mpc"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "N_SIMULATION");
    N_SIMULATION = robot_node["N_SIMULATION"].as<int>();

    assert_yaml_parsing(robot_node, "robot", "type_MPC");
    type_MPC = robot_node["type_MPC"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "use_flat_plane");
    use_flat_plane = robot_node["use_flat_plane"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "predefined_vel");
    predefined_vel = robot_node["predefined_vel"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "kf_enabled");
    kf_enabled = robot_node["kf_enabled"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "enable_pyb_GUI");
    enable_pyb_GUI = robot_node["enable_pyb_GUI"].as<bool>();

}
