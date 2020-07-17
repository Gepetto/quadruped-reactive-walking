# mpc-tsid

# Dependencies

* Install a Python 3 version (for instance Python 3.6)

* For the following installations each time you see something like "py27", replace it by your Python version ("py36" for instance)

* Install Pinocchio: https://stack-of-tasks.github.io/pinocchio/download.html

* Install Gepetto Viewer: `sudo apt install robotpkg-py36-qt4-gepetto-viewer-corba`

* Install robot data: `sudo apt install robotpkg-example-robot-data`

* Install Scipy, Numpy, Matplotlib, IPython: `python3.6 -m pip install --user numpy scipy matplotlib ipython`

* Install PyBullet: `pip3 install --user pybullet`

* Install OSQP solver: `pip3 install --user osqp`

* Install package that handles the gamepad: `pip3 install --user inputs`

* Install TSID: https://github.com/stack-of-tasks/tsid#installation You can put the repo in another folder if you want, like `cd ~/install/` instead of `cd $DEVEL/openrobots/src/` for the first line.

# Optional 

To reduce computation time:

* Install Cython: `pip3 install --user cython`

* Cythonize the foot trajectory generator: `python3.6 setup.py build_ext --inplace`
