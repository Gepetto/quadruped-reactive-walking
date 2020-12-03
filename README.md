# quadruped-reactive-walking

Implementation of a reactive walking controller for quadruped robots. Architecture mainly in Python with some parts in C++ with bindings to Python.

# Dependencies

* Install a Python 3 version (for instance Python 3.6)

* For the following installations each time you see something like "py36", replace it by your Python version ("py35" for instance)

* Install Pinocchio: https://stack-of-tasks.github.io/pinocchio/download.html

* Install Gepetto Viewer: `sudo apt install robotpkg-py36-qt4-gepetto-viewer-corba`

* Install robot data: `sudo apt install robotpkg-example-robot-data`

* Install Scipy, Numpy, Matplotlib, IPython: `python3.6 -m pip install --user numpy scipy matplotlib ipython`

* Install PyBullet: `pip3 install --user pybullet`

* Install OSQP solver: `pip3 install --user osqp`

* Install package that handles the gamepad: `pip3 install --user inputs`

* Install TSID: https://github.com/stack-of-tasks/tsid#installation You can put the repo in another folder if you want, like `cd ~/install/` instead of `cd $DEVEL/openrobots/src/` for the first line.

# Compiling the C++ parts

* Initialize the cmake submodule: `git submodule init`

* Update the cmake submodule: `git submodule udpdate`

* Create a build folder: `mkdir build`

* Get inside and cmake: `cd build` then `cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/install -DPYTHON_EXECUTABLE=$(which python3.6) -DPYTHON_STANDARD_LAYOUT=ON`

* Compile and and install Python bindings: `make install`

* If at some point you want to uninstall the bindings: `make uninstall`

# Run the simulation

* Run `python3.6 main_solo12_openloop.py -i test` while being in the `scripts` folder

# Tune the simulation

* In `main_solo12_control.py`, you can change some of the parameters defined at the beginning of the `control_loop` function.

* You can define a new gait in `src/Planner.cpp` using `create_trot` or `create_walk` as models. Create a new function (like `create_bounding` for a bounding gait) and call it inside the Planner constructor before `create_gait_f()`.

* For the MPC QP problem you can tune weights of the Q and P matrices in the `create_weight_matrices` function of `src/MPC.cpp`.

* Remember that if you change C++ code you need to recompile the library.