///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Params class
///
/// \details Planner that outputs the reference trajectory of the base based on the reference 
///          velocity given by the user and the current position/velocity of the base
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef PARAMS_H_INCLUDED
#define PARAMS_H_INCLUDED

#include "qrw/Types.h"
#include <yaml-cpp/yaml.h>

class Params
{
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Empty constructor
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    Params();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Destructor.
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ~Params() {}

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Initializer
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void initialize(const std::string& file_path);


    // See .yaml file for meaning of parameters
    std::string interface;
    bool SIMULATION;
    bool LOGGING;
    bool PLOTTING;
    int envID;
    bool use_flat_plane;
    bool predefined_vel;
    int velID;
    int N_SIMULATION;
    bool enable_pyb_GUI;
    bool enable_corba_viewer;
    bool enable_multiprocessing;
    bool perfect_estimator;

    std::vector<double> q_init;
    double dt_wbc;
    int N_gait;
    double dt_mpc;
    double T_gait;
    double T_mpc;
    int type_MPC;
    bool kf_enabled;
    double Kp_main;
    double Kd_main;
    double Kff_main;

    double fc_v_esti;

    double k_feedback;

    double max_height;
    double lock_time;
    double vert_time;

    std::vector<double> osqp_w_states;
    std::vector<double> osqp_w_forces;
    double osqp_Nz_lim;

    double Kp_flyingfeet;
    double Kd_flyingfeet;

    double Q1;
    double Q2;
    double Fz_max;
    double Fz_min;


    // Not defined in yaml
    double mass;  // Mass of the robot
    std::vector<double> I_mat;  // Inertia matrix
    double h_ref;  // Reference height for the base
    std::vector<double> shoulders;  // Position of shoulders in base frame
    std::vector<double> footsteps_init;  // Initial 3D position of footsteps in base frame
    std::vector<double> footsteps_under_shoulders;  // // Positions of footsteps to be "under the shoulder"

};

namespace yaml_control_interface
{
#define assert_yaml_parsing(yaml_node, parent_node_name, child_node_name)      \
    if (!yaml_node[child_node_name])                                           \
    {                                                                          \
        std::ostringstream oss;                                                \
        oss << "Error: Wrong parsing of the YAML file from src file: ["        \
            << __FILE__ << "], in function: [" << __FUNCTION__ << "], line: [" \
            << __LINE__ << ". Node [" << child_node_name                       \
            << "] does not exists under the node [" << parent_node_name        \
            << "].";                                                           \
        throw std::runtime_error(oss.str());                                   \
    }                                                                          \
    assert(true)

#define assert_file_exists(filename)                                    \
    std::ifstream f(filename.c_str());                                  \
    if (!f.good())                                                      \
    {                                                                   \
        std::ostringstream oss;                                         \
        oss << "Error: Problem opening the file [" << filename          \
            << "], from src file: [" << __FILE__ << "], in function: [" \
            << __FUNCTION__ << "], line: [" << __LINE__                 \
            << ". The file may not exists.";                            \
        throw std::runtime_error(oss.str());                            \
    }                                                                   \
    assert(true)
}  // end of yaml_control_interface namespace

#endif  // PARAMS_H_INCLUDED
