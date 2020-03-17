from distutils.core import setup
from Cython.Build import cythonize


modules = ["FootTrajectoryGenerator.py"]

setup(name='MPC TSID app',
      ext_modules=cythonize(modules))
