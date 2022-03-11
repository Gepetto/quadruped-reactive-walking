### EVALUATION 1 : 

Some tests to compare the two MPC solvers: ddp & osqp. The data collected during the simulation is used to restart and analyze each control cycle with different MPCs or different parameters. A third DDP MPC can be run to evaluate and compare the parameters. (see bool *Relaunch_DDP*)

    -> Run the simulation with boolean LOGGING on true.
    -> Modify the path in *Recover Logged data* section.
    -> Select the MPC control cycle to be analysed and tune the gains if necessary.
    -> Run python3 crocoddyl_eval/test_1/analyse_control_cycle.py


### EVALUATION 2 :

Test to check the derivatives computed with the action models. The derivatives are computed with finite differences using the crocodile.ActionMode NumDiff class. The derivatived tested coressponds to :

- Lx : Derivative of the cost wrt states x
- Lu : Derivative of the cost wrt command u
- Fx : Derivative of the dynamics wrt states x
- Fu : Derivative of the dynamics wrt command u

For the Hessian of the cost Lxx and Luu, the results are not accurate. With NumDiff, the hessian is approximated with the residual vector : Lxx ~= R^T R, which is not the case here because of the friction cost, non linear terms...

    -> Modify the parameters of the model in crocoddyl_eval/test_2/unittest_model.py (to evaluate each cost)
    -> Run python3 crocoddyl_eval/test_2/unittest_model.py


### EVALUATION 3 : 

Test to evaluate the MPC with the optimisation of the foosteps. Comparison of : ddp + fstep optimization & osqp. The data collected during the simulation is used to restart and analyze each control cycle. The parameters can be modified in *DDP MPC* section.

    -> Run the simulation with boolean LOGGING on true and type_MPC == 3.
    -> Modify the path in *Recover Logged data* section.
    -> Select the MPC control cycle to be analysed and tune the gains if necessary.
    -> Run python3 crocoddyl_eval/test_1/analyse_control_cycle.py

