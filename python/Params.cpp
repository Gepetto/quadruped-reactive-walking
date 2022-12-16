#include "qrw/Params.hpp"

#include "bindings/python.hpp"

template <typename Params>
struct ParamsVisitor : public bp::def_visitor<ParamsVisitor<Params>> {
  template <class PyClassParams>
  void visit(PyClassParams& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("read_yaml", &Params::read_yaml, bp::args("file_path"),
             "Read yaml file to retrieve parameters from Python.\n")
        .def("convert_gait_vec", &Params::convert_gait_vec,
             "Convert gait vector into matrix.\n")
        .def("initialize", &Params::initialize,
             "Initialize Params from Python.\n")

        // Read Params from Python
        .def_readwrite("config_file", &Params::config_file)
        .def_readwrite("interface", &Params::interface)
        .def_readwrite("DEMONSTRATION", &Params::DEMONSTRATION)
        .def_readwrite("SIMULATION", &Params::SIMULATION)
        .def_readwrite("LOGGING", &Params::LOGGING)
        .def_readwrite("PLOTTING", &Params::PLOTTING)
        .def_readwrite("dt_wbc", &Params::dt_wbc)
        .def_readwrite("envID", &Params::envID)
        .def_readwrite("q_init", &Params::q_init)
        .def_readwrite("dt_mpc", &Params::dt_mpc)
        .def_readwrite("N_periods", &Params::N_periods)
        .def_readwrite("N_SIMULATION", &Params::N_SIMULATION)
        .def_readwrite("type_MPC", &Params::type_MPC)
        .def_readwrite("use_flat_plane", &Params::use_flat_plane)
        .def_readwrite("predefined_vel", &Params::predefined_vel)
        .def_readwrite("kf_enabled", &Params::kf_enabled)
        .def_readwrite("Kp_main", &Params::Kp_main)
        .def_readwrite("Kd_main", &Params::Kd_main)
        .def_readwrite("Kff_main", &Params::Kff_main)
        .def_readwrite("osqp_w_states", &Params::osqp_w_states)
        .def_readwrite("osqp_w_forces", &Params::osqp_w_forces)
        .def_readwrite("gait_vec", &Params::gait_vec)
        .def_readonly("gait", &Params::get_gait)
        .def_readonly("t_switch", &Params::get_t_switch)
        .def_readonly("v_switch", &Params::get_v_switch)
        .def("set_v_switch", &Params::set_v_switch, bp::args("v_switch"),
             "Set v_switch matrix from Python.\n")
        .def_readwrite("enable_pyb_GUI", &Params::enable_pyb_GUI)
        .def_readwrite("enable_corba_viewer", &Params::enable_corba_viewer)
        .def_readwrite("enable_multiprocessing",
                       &Params::enable_multiprocessing)
        .def_readwrite("perfect_estimator", &Params::perfect_estimator)
        .def_readwrite("w_tasks", &Params::w_tasks)
        .def_readwrite("T_gait", &Params::T_gait)
        .def_readwrite("mass", &Params::mass)
        .def_readwrite("I_mat", &Params::I_mat)
        .def_readwrite("CoM_offset", &Params::CoM_offset)
        .def_readwrite("h_ref", &Params::h_ref)
        .def_readwrite("shoulders", &Params::shoulders)
        .def_readwrite("lock_time", &Params::lock_time)
        .def_readwrite("vert_time", &Params::vert_time)
        .def_readwrite("footsteps_init", &Params::footsteps_init)
        .def_readwrite("footsteps_under_shoulders",
                       &Params::footsteps_under_shoulders)
        .def_readwrite("enable_comp_forces", &Params::enable_comp_forces)
        .def_readwrite("solo3D", &Params::solo3D)
        .def_readwrite("enable_multiprocessing_mip",
                       &Params::enable_multiprocessing_mip)
        .def_readwrite("environment_URDF", &Params::environment_URDF)
        .def_readwrite("environment_heightmap", &Params::environment_heightmap)
        .def_readwrite("heightmap_fit_length", &Params::heightmap_fit_length)
        .def_readwrite("heightmap_fit_size", &Params::heightmap_fit_size)
        .def_readwrite("number_steps", &Params::number_steps)
        .def_readwrite("max_velocity", &Params::max_velocity)
        .def_readwrite("use_bezier", &Params::use_bezier)
        .def_readwrite("use_sl1m", &Params::use_sl1m)
        .def_readwrite("use_heuristic", &Params::max_velocity);
  }

  static void expose() {
    bp::class_<Params>("Params", bp::no_init).def(ParamsVisitor<Params>());
  }
};

void exposeParams() { ParamsVisitor<Params>::expose(); }
