# quadruped-reactive-walking

Implementation of a reactive walking controller for quadruped robots. Architecture mainly in Python with some parts in C++ with bindings to Python.

## Dependencies

* Standard python scientific tools
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio/)
* [Gepetto-Viewer](https://github.com/gepetto/gepetto-viewer)
* [Example-Robot-Data](https://github.com/gepetto/example-robot-data)
* [eiquadprog](https://github.com/stack-of-tasks/eiquadprog)
* [TSID](https://github.com/stack-of-tasks/tsid)
* [PyBullet](https://pybullet.org/wordpress/)
* [OSQP](https://osqp.org)
* [inputs](https://github.com/zeth/inputs)

To get all of these, on Debian or Ubuntu, you can use:

```
python3 -m pip install --user numpy scipy matplotlib ipython pybullet inputs
sudo apt install robotpkg-py3\*-{pinocchio,qt5-gepetto-viewer-corba,example-robot-data,tsid} robotpkg-osqp
```

(you'll need to setup [robotpkg apt repositories first](http://robotpkg.openrobots.org/debian.html))

## Compiling the C++ parts

* Clone interface repository:  `git clone --recursive https://github.com/gepetto/quadruped-reactive-walking`
* Get inside and create a build folder: `cd quadruped-reactive-walking` then `mkdir build`
* Get inside and configure: `cd build` then
  `cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/install -DPYTHON_EXECUTABLE=$(which python3) -DPYTHON_STANDARD_LAYOUT=ON`
* Compile and install the project in your prefix: `make install`

## Run the simulation

* Run `python3 main_solo12_control.py -i test` while being in the `scripts` folder
* Sometimes the parallel process that runs the MPC does not end properly so it will keep running in the background
  forever, you can manually end all python processes with `pkill -9 python3`.

## Tune the simulation

* In `main_solo12_control.py`, you can change some of the parameters defined at the beginning of the `control_loop`
  function.
* Set `envID` to 1 to load obstacles and stairs.
* Set `use_flat_plane` to False to load a ground with lots of small bumps.
* If you have a gamepad you can control the robot with two joysticks by turning `predefined_vel` to False in
  `main_solo12_control.py`. Velocity limits with the joystick are defined in `Joystick.py` by `self.VxScale` (maximul
  lateral velocity), `self.VyScale` (maximum forward velocity) and `self.vYawScale` (maximum yaw velocity).
* If `predefined_vel = True` the robot follows the reference velocity pattern velID. Velocity patterns are defined in
  `Joystick.py`, you can modify them or add new ones. Each profile defines forward, lateral and yaw velocities that
  should be reached at the associated loop iterations (in `self.k_switch`). There is an automatic interpolation between
  milestones to have a smooth reference velocity command.
* You can define a new gait in `src/Planner.cpp` using `create_trot` or `create_walk` as models. Create a new function
  (like `create_bounding` for a bounding gait) and call it inside the Planner constructor before `create_gait_f()`.
* You can modify the swinging feet apex height in `include/quadruped-reactive-control/Planner.hpp` with
  `max_height_feet` or the lock time before touchdown with `t_lock_before_touchdown` (to lock the target location on
  the ground before touchdown).
* For the MPC QP problem you can tune weights of the Q and P matrices in the `create_weight_matrices` function of
  `src/MPC.cpp`.
* Remember that if you modify C++ code you need to recompile and reinstall the library.
