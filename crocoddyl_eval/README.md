test_1 : Some tests to compare the two MPC solvers in simple case (no obstacles) : ddp & osqp. The solvers operate in parallel and control is looped over the ddp algorithm. All data is stored during the simulation inside the log_eval folder, the algorithms have exactly the same input parameters for each control cycle. Then, to analyze the behavior of solvers, the file analyse_simu.py is used to evaluate each control cycle. Both algorithms can be relaunched and weights on the ddp algorithm can be changed to evaluate the modifications (line 55, Relaunch_DDP boolean to relaunch the ddp algorithm) 

	-> Run python3 crocoddyl_eval/test_1/run_scenarios.py
	-> Run python3 crocoddyl_eval/test_1/analyse_simu.py 
