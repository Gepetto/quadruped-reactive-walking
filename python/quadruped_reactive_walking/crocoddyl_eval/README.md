test_1 : Some tests to compare the two MPC solvers in simple case (no obstacles) : ddp & osqp. The solvers operate in parallel and control is looped over the ddp algorithm. All data is stored during the simulation inside the log_eval folder, the algorithms have exactly the same input parameters for each control cycle. Then, to analyze the behavior of solvers, the file analyse_simu.py is used to evaluate each control cycle. Both algorithms can be relaunched and weights on the ddp algorithm can be changed to evaluate the modifications (line 55, Relaunch_DDP boolean to relaunch the ddp algorithm) 

	-> Run python3 crocoddyl_eval/test_1/run_scenarios.py
	-> Run python3 crocoddyl_eval/test_1/analyse_simu.py 


test_2 : Test to check the derivatives using the crocoddyl.ActionModelNumDiff (derivatives computed with finite differences). The simulation is run for a short time to get initializations of the foot position. The Luu term (hessian of the cost with regards to the command vector) cannot be accurate because it uses the residual cost (Luu ~= R^T R) which is not written that way in that case.

	-> Run python3 crocoddyl_eval/test_2/unit_test.py


test_3 : Test to run the simulation using the ddp as foot step planner too. It involves new c++ models with augmented states to  optimises the feet position.  

	-> Run python3 crocoddyl_eval/test_3/run_scenarios.py


test_5 : Test to run the simulation using the ddp at the frequency of TSID. The number of node in the ddp has been adjusted to avoid 
a too large number of them. Each control cycle can be monitored after the log in the analyse_simu file.

	-> Run python3 crocoddyl_eval/test_5/run_scenarios.py
	-> Run ipython3 crocoddyl_eval/test_5/analyse_simu.py -i
