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
    double dt_wbc;
    int N_gait;
    int envID;
    int velID;
    double dt_mpc;
    double T_gait;
    double T_mpc;
    int N_SIMULATION;
    bool type_MPC;
    bool use_flat_plane;
    bool predefined_vel;
    bool kf_enabled;
    bool enable_pyb_GUI;

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
