#include "qrw/Params.hpp"

using namespace yaml_control_interface;

Params::Params()
    : interface("")
    , SIMULATION(false)
    , LOGGING(false)
    , PLOTTING(false)
    , envID(0)
    , use_flat_plane(false)
    , predefined_vel(false)
    , velID(0)
    , N_SIMULATION(0)
    , enable_pyb_GUI(false)
    , enable_multiprocessing(false)
    , perfect_estimator(false)

    , q_init(12, 0.0) // Fill with zeros, will be filled with values later
    , dt_wbc(0.0)
    , N_gait(0)
    , dt_mpc(0.0)
    , T_gait(0.0)
    , T_mpc(0.0)
    , type_MPC(0)
    , kf_enabled(false)
    
    , k_feedback(0.0)

    , max_height(0.0)
    , lock_time(0.0)

    , osqp_w_states(12, 0.0) // Fill with zeros, will be filled with values later
    , osqp_w_forces(3, 0.0) // Fill with zeros, will be filled with values later
    , osqp_Nz_lim(0.0)

    , Kp_flyingfeet(0.0)
    , Kd_flyingfeet(0.0)

    , Q1(0.0)
    , Q2(0.0)
    , Fz_max(0.0)
    , Fz_min(0.0)

    , mass(0.0)
    , I_mat(9, 0.0) // Fill with zeros, will be filled with values later
    , h_ref(0.0)
    , shoulders(12, 0.0) // Fill with zeros, will be filled with values later
    , footsteps_init(12, 0.0) // Fill with zeros, will be filled with values later
    , footsteps_under_shoulders(12, 0.0) // Fill with zeros, will be filled with values later
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

    assert_yaml_parsing(robot_node, "robot", "q_init");
    q_init = robot_node["q_init"].as<std::vector<double> >();

    assert_yaml_parsing(robot_node, "robot", "dt_mpc");
    dt_mpc = robot_node["dt_mpc"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "T_gait");
    T_gait = robot_node["T_gait"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "T_mpc");
    T_mpc = robot_node["T_mpc"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "N_SIMULATION");
    N_SIMULATION = robot_node["N_SIMULATION"].as<int>();

    assert_yaml_parsing(robot_node, "robot", "type_MPC");
    type_MPC = robot_node["type_MPC"].as<int>();

    assert_yaml_parsing(robot_node, "robot", "use_flat_plane");
    use_flat_plane = robot_node["use_flat_plane"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "predefined_vel");
    predefined_vel = robot_node["predefined_vel"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "kf_enabled");
    kf_enabled = robot_node["kf_enabled"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "enable_pyb_GUI");
    enable_pyb_GUI = robot_node["enable_pyb_GUI"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "enable_multiprocessing");
    enable_multiprocessing = robot_node["enable_multiprocessing"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "perfect_estimator");
    perfect_estimator = robot_node["perfect_estimator"].as<bool>();

    assert_yaml_parsing(robot_node, "robot", "k_feedback");
    k_feedback = robot_node["k_feedback"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "max_height");
    max_height = robot_node["max_height"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "lock_time");
    lock_time = robot_node["lock_time"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "osqp_w_states");
    osqp_w_states = robot_node["osqp_w_states"].as<std::vector<double> >();

    assert_yaml_parsing(robot_node, "robot", "osqp_w_forces");
    osqp_w_forces = robot_node["osqp_w_forces"].as<std::vector<double> >();

    assert_yaml_parsing(robot_node, "robot", "osqp_Nz_lim");
    osqp_Nz_lim = robot_node["osqp_Nz_lim"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "Kp_flyingfeet");
    Kp_flyingfeet = robot_node["Kp_flyingfeet"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "Kd_flyingfeet");
    Kd_flyingfeet = robot_node["Kd_flyingfeet"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "Q1");
    Q1 = robot_node["Q1"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "Q2");
    Q2 = robot_node["Q2"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "Fz_max");
    Fz_max = robot_node["Fz_max"].as<double>();

    assert_yaml_parsing(robot_node, "robot", "Fz_min");
    Fz_min = robot_node["Fz_min"].as<double>();

}
