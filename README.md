# qrw

Implementation of a reactive walking controller for quadruped robots. Architecture mainly in Python with some parts in C++ with bindings to Python.

# Dependencies

* Install a Python 3 version (for instance Python 3.6)

* For the following installations each time you see something like "py36", replace it by your Python version ("py35" for instance)

* Install Pinocchio: https://stack-of-tasks.github.io/pinocchio/download.html

* Install Gepetto Viewer: `sudo apt install robotpkg-py36-qt4-gepetto-viewer-corba`

* Install robot data: `sudo apt install robotpkg-example-robot-data`

* Install Scipy, Numpy, Matplotlib, IPython: `python3.6 -m pip install --user numpy scipy matplotlib ipython`

* Install PyBullet: `pip3 install --user pybullet`

* Install OSQP solver: [https://osqp.org/docs/get_started/sources.html#build-the-binaries]
    * git clone --recursive https://github.com/oxfordcontrol/osqp
    * cd osqp
    * Edit CMakeLists.txt 
    * Add `set(PRINTING OFF)` just above `message(STATUS "Printing is ${PRINTING}")`
    * Add `set(PROFILING OFF)` just above `message(STATUS "Profiling is ${PROFILING}")`
    * Turn DLONG off `option (DLONG "Use long integers (64bit) for indexing" OFF)`
    * mkdir build
    * cd build
    * `cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/install -DPYTHON_EXECUTABLE=$(which python3.6) -DPYTHON_STANDARD_LAYOUT=ON` (to install in ~/install folder)
    * make install

* Install YAML parser for C++: [https://github.com/jbeder/yaml-cpp] => sudo apt install libyaml-cpp-dev

* Install package that handles the gamepad: `pip3 install --user inputs`

* Install eiquadprog: `sudo apt install robotpkg-eiquadprog`

* Install TSID: [https://github.com/stack-of-tasks/tsid#installation] You can put the repo in another folder if you want, like `cd ~/install/` instead of `cd $DEVEL/openrobots/src/` for the first line.

* Clone interface repository: in `/scripts`, `git clone https://github.com/paLeziart/solopython`

# Compiling the C++ parts

* Initialize the cmake submodule: `git submodule init`

* Update the cmake submodule: `git submodule udpdate`

* Create a build folder: `mkdir build`

* Get inside and cmake: `cd build` then `cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/install -DPYTHON_EXECUTABLE=$(which python3.6) -DPYTHON_STANDARD_LAYOUT=ON`

* Compile Python bindings: `make`

* Copy them to the script folder so that the scripts can access the compiled code: `cp python/quadruped_reactive_walking/libquadruped_reactive_walking.so ../scripts/`

# Run the simulation

* if you install you hpp with robotpkg run `export INSTALL_HPP_DIR=/opt/openrobots/share/`

* Run `python3.6 -m quadruped_reactive_walking.main_solo12_control -i test` while being in the `scripts` folder

* Sometimes the parallel process that runs the MPC does not end properly so it will keep running in the background forever, you can manually end all python processes with `pkill -9 python3`

# Run with sl1m using rbprm

- 1/ Edit the `./config/walk_parameters.yaml` to change:
```
diff --git a/config/walk_parameters.yaml b/config/walk_parameters.yaml
index 72bcfb2..b77b72c 100644
--- a/config/walk_parameters.yaml
+++ b/config/walk_parameters.yaml
@@ -3,7 +3,7 @@ robot:
     config_file: config_solo12.yaml  #  Name of the yaml file containing hardware information
 
     interface: eth0  # Name of the communication inerface (check with ifconfig)
-    LOGGING: true  # Enable/disable logging during the experiment
+    LOGGING: false  # Enable/disable logging during the experiment
     PLOTTING: false  # Enable/disable automatic plotting at the end of the experiment
     DEMONSTRATION: false  # Enable/disable demonstration functionalities
     SIMULATION: true  # Enable/disable PyBullet simulation or running on real robot
@@ -23,7 +23,7 @@ robot:
     q_init: [0.0, 0.7, -1.4, 0.0, 0.7, -1.4, 0.0, -0.7, 1.4, 0.0, -0.7, 1.4]  # Initial articular positions
     dt_wbc: 0.001  # Time step of the whole body control
     dt_mpc: 0.02  # Time step of the model predictive control
-    type_MPC: 3  # Which MPC solver you want to use: 0 for OSQP MPC, 1, 2, 3 for Crocoddyl MPCs
+    type_MPC: 0  # Which MPC solver you want to use: 0 for OSQP MPC, 1, 2, 3 for Crocoddyl MPCs
 #     Kp_main: [0.0, 0.0, 0.0]  # Proportional gains for the PD+
     Kp_main: [3.0, 3.0, 3.0]  # Proportional gains for the PD+
 #     Kd_main: [0., 0., 0.]  # Derivative gains for the PD+
@@ -81,7 +81,7 @@ robot:
     Fz_min: 0.0  # Minimal vertical contact force [N]
 
     # Parameters fro solo3D simulation
-    solo3D: false  # Activation of the 3D environment, and corresponding planner blocks
+    solo3D: true  # Activation of the 3D environment, and corresponding planner blocks
     enable_multiprocessing_mip: true  # Enable/disable running the MIP in another process in parallel of the main loop
     environment_URDF: "/short_bricks/short_bricks.urdf"
     environment_heightmap: "/short_bricks/short_bricks.bin"
```

- 2/ In a first terminal run 
```
hpp-rbprm-server
```

- 3/ In second terminal download the environment models (ask the package maintainer) and setup your bash environment variables:
```
export INSTALL_HPP_DIR=/opt/openrobots/share/
export SOLO3D_ENV_DIR=/path/to/Solo3D
export ROS_PACKAGE_PATH=/opt/openrobots/share/example-robot-data/robots:${SOLO3D_ENV_DIR}
python3.8 -m quadruped_reactive_walking.main_solo12_control
```

# Tune the simulation

* In `main_solo12_control.py`, you can change some of the parameters defined at the beginning of the `control_loop` function.

* Set `envID` to 1 to load obstacles and stairs.

* Set `use_flat_plane` to False to load a ground with lots of small bumps.

* If you have a gamepad you can control the robot with two joysticks by turning `predefined_vel` to False in `main_solo12_control.py`. Velocity limits with the joystick are defined in `Joystick.py` by `self.VxScale` (maximul lateral velocity), `self.VyScale` (maximum forward velocity) and `self.vYawScale` (maximum yaw velocity).

* If `predefined_vel = True` the robot follows the reference velocity pattern. Velocity patterns are defined in walk_parameters`, you can modify them or add new ones. Each profile defines forward, lateral and yaw velocities that should be reached at the associated loop iterations (in `self.k_switch`). There is an automatic interpolation between milestones to have a smooth reference velocity command.

* You can define a new gait in `src/Planner.cpp` using `createTrot` or `createWalk` as models. Create a new function (like `createBounding` for a bounding gait) and call it inside the Planner constructor before `create_gait_f()`.

* You can modify the swinging feet apex height in `include/quadruped-reactive-control/Planner.hpp` with `maxHeight_` or the lock time before touchdown with `lockTime_` (to lock the target location on the ground before touchdown).

* For the MPC QP problem you can tune weights of the Q and P matrices in the `create_weight_matrices` function of `src/MPC.cpp`.

* Remember that if you modify C++ code you need to recompile the library.