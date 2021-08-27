import numpy as np
from matplotlib import pyplot as plt
import libquadruped_reactive_walking as lqrw
import MPC_Wrapper
import utils_mpc

"""dt_mpc = 0.02
T_mpc = 0.32

k_mpc = 20
N_gait = 100"""
h_ref = 0.2447495

def test_1():
    # Position of base in world frame
    q = np.zeros((18, 1))
    q[0:6, 0] = np.array([0.0, 0.0, h_ref, 0.0, 0.0, 0.0])
    q[6:, 0] = np.array([0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7, +1.4])

    # Velocity of base in horizontal frame
    h_v = np.zeros((18, 1))
    h_v[0, 0] = 0.1
    h_v[1, 0] = 0.1
    #h_v[1, 0] = 0.1

    # Velocity reference of base in horizontal frame
    v_ref = np.zeros((18, 1))
    v_ref[0, 0] = 0.1

    # Params
    params = lqrw.Params()  # Object that holds all controller parameters
    N_gait = params.N_gait

    """params.dt_mpc = dt_mpc
    params.T_mpc = T_mpc
    params.h_ref = h_ref
    params.enable_multiprocessing = False"""

    # Initialisation of the solo model/data and of the Gepetto viewer
    utils_mpc.init_robot(q[6:, 0], params)

    # State planner
    statePlanner = lqrw.StatePlanner()
    statePlanner.initialize(params)

    gait = lqrw.Gait()
    gait.initialize(params)

    footstepPlanner = lqrw.FootstepPlanner()
    footstepPlanner.initialize(params, gait)

    # MPC wrapper
    mpc_wrapper_classic = MPC_Wrapper.MPC_Wrapper(params, q)
    params.type_MPC = 3
    mpc_wrapper_croco = MPC_Wrapper.MPC_Wrapper(params, q)

    # Update gait
    for i in range(5):
        gait.updateGait(i*20, 20, 0)
        cgait = gait.getCurrentGait()

    # Compute target footstep based on current and reference velocities
    footstepPlanner.updateFootsteps(False, 20, q[:, 0:1], h_v[0:6, 0:1].copy(), v_ref[0:6, 0])
    fsteps = footstepPlanner.getFootsteps()

    fsteps[:12,  0] += h_v[0, 0] * params.T_gait * 0.25
    fsteps[:12,  1] += h_v[1, 0] * params.T_gait * 0.25
    fsteps[:12,  9] += h_v[0, 0] * params.T_gait * 0.25
    fsteps[:12, 10] += h_v[1, 0] * params.T_gait * 0.25
    
    """from IPython import embed
    embed()"""


    # Run state planner (outputs the reference trajectory of the base)
    statePlanner.computeReferenceStates(q[0:6, 0:1].copy(), h_v[0:6, 0:1].copy(), v_ref[0:6, 0:1].copy(), 0.0)
    xref = statePlanner.getReferenceStates()

    # Create fsteps matrix
    shoulders = np.zeros((3, 4))
    shoulders[0, :] = [0.1946, 0.1946, -0.1946, -0.1946]
    shoulders[1, :] = [0.14695, -0.14695, 0.14695, -0.14695]
    shoulders[2, :] = np.zeros(4)

    # Solve MPC problem once every k_mpc iterations of the main loop
    mpc_wrapper_classic.solve(0, xref.copy(), fsteps.copy(), cgait.copy(), np.zeros((3,4)))
    mpc_wrapper_croco.solve(0, xref.copy(), fsteps.copy(), cgait.copy(), shoulders.copy())

    # Retrieve reference contact forces in horizontal frame
    x_f_mpc_classic = mpc_wrapper_classic.get_latest_result()
    x_f_mpc_croco = mpc_wrapper_croco.get_latest_result()

    # Solve MPC problem once every k_mpc iterations of the main loop
    mpc_wrapper_classic.solve(1, xref, fsteps, cgait, np.zeros((3,4)))
    mpc_wrapper_croco.solve(1, xref, fsteps, cgait, shoulders)

    # Retrieve reference contact forces in horizontal frame
    x_f_mpc_classic = mpc_wrapper_classic.get_latest_result()
    x_f_mpc_croco_32 = mpc_wrapper_croco.get_latest_result()

    x_f_mpc_classic = np.hstack((np.vstack((xref[:, 0:1], np.zeros((12, 1)))), x_f_mpc_classic))
    x_f_mpc_croco = np.hstack((np.vstack((xref[:, 0:1], np.zeros((12, 1)))), x_f_mpc_croco_32[:24, :]))

    # print(xref)

    index6 = [1, 3, 5, 2, 4, 6]
    index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    titles = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    N = x_f_mpc_classic.shape[1]
    plt.figure()
    for i in range(6):
        plt.subplot(3, 2, index6[i])
        h1, = plt.plot(x_f_mpc_classic[i, :N], "b", linewidth=2)
        h2, = plt.plot(x_f_mpc_croco[i, :N], "r", linewidth=2)
        h3, = plt.plot(xref[i, :N], "b", linestyle="--", marker='x', color="g", linewidth=2)
        plt.xlabel("Time [s]")
        # plt.legend([h1, h2, h3], ["Output trajectory of classic MPC", "Output trajectory of croco MPC", "Input trajectory of planner"])
        plt.legend([h1, h2, h3], ["OSQP", "Croco", "Ref"])
        plt.title("Predicted trajectory for " + titles[i])
    plt.suptitle("Analysis of trajectories in position and orientation computed by the MPC")

    plt.figure()
    for i in range(6):
        plt.subplot(3, 2, index6[i])
        h1, = plt.plot(x_f_mpc_classic[i+6, :N], "b", linewidth=2)
        h2, = plt.plot(x_f_mpc_croco[i+6, :N], "r", linewidth=2)
        h3, = plt.plot(xref[i+6, :N], "b", linestyle="--", marker='x', color="g", linewidth=2)
        plt.xlabel("Time [s]")
        # plt.legend([h1, h2, h3], ["Output trajectory of classic MPC", "Output trajectory of croco MPC", "Input trajectory of planner"])
        plt.title("Predicted trajectory for velocity in " + titles[i])
    plt.suptitle("Analysis of trajectories of linear and angular velocities computed by the MPC")

    lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
    lgd2 = ["FL", "FR", "HL", "HR"]
    plt.figure()
    for i in range(12):
        if i == 0:
            ax0 = plt.subplot(3, 4, index12[i])
        else:
            plt.subplot(3, 4, index12[i], sharex=ax0)
        h1, = plt.plot(x_f_mpc_classic[i+12, 1:(N+1)], "b", linewidth=3)
        h2, = plt.plot(x_f_mpc_croco[i+12, 1:(N+1)], "r", linewidth=3)
        plt.xlabel("Time [s]")
        plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [N]")
        """if (i % 3) == 2:
            plt.ylim([-0.0, 26.0])
        else:
            plt.ylim([-26.0, 26.0])"""
        plt.legend([h1, h2], ["Classic", "Croco"])
    plt.suptitle("Contact forces (MPC command)")

    """index8 = [1, 5, 2, 6, 3, 7, 4, 8]
    lgd_x = ["FL", "FR", "HL", "HR"]
    lgd_y = [" x", " y"]
    plt.figure()
    for i in range(8):
        plt.subplot(2, 4, index8[i])
        plt.plot(shoulders[int(i % 2), int(i/2)] * np.ones(x_f_mpc_croco_32.shape[1]), "b", linewidth=3)
        plt.plot(x_f_mpc_croco_32[24+i, :], "r", linewidth=3)
        plt.legend([lgd_x[int(i/2)] + lgd_y[i % 2] + " classic", lgd_x[int(i/2)] + lgd_y[i % 2] + " croco"])
    plt.suptitle("Footsteps positions")"""

    plt.show(block=True)


def test_2():
    # Position of base in world frame
    q = np.zeros((19, 1))
    q[0:7, 0] = np.array([0.0, 0.0, h_ref, 0.0, 0.0, 0.0, 1.0])
    q[7:, 0] = np.array([0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7, +1.4])

    # Velocity of base in horizontal frame
    h_v_1 = np.zeros((18, 1))
    h_v_2 = np.zeros((18, 1))
    h_v_1[0, 0] = 0.1
    h_v_1[1, 0] = 0.0
    h_v_2[0, 0] = - h_v_1[0, 0]
    h_v_2[1, 0] = - h_v_1[1, 0]

    # Velocity reference of base in horizontal frame
    v_ref_1 = np.zeros((18, 1))
    v_ref_2 = np.zeros((18, 1))
    v_ref_1[0, 0] = 0.1
    v_ref_2[0, 0] = - v_ref_1[0, 0]

    # Params
    params = lqrw.Params()  # Object that holds all controller parameters

    # Initialisation of the solo model/data and of the Gepetto viewer
    utils_mpc.init_robot(q[7:, 0], params, False)

    # State planner
    statePlanner_1 = lqrw.StatePlanner()
    statePlanner_1.initialize(params)
    statePlanner_2 = lqrw.StatePlanner()
    statePlanner_2.initialize(params)

    gait = lqrw.Gait()
    gait.initialize(params)

    footstepPlanner_1 = lqrw.FootstepPlanner()
    footstepPlanner_1.initialize(params, gait)
    footstepPlanner_2 = lqrw.FootstepPlanner()
    footstepPlanner_2.initialize(params, gait)

    # MPC wrapper
    enable_multiprocessing = False
    mpc_wrapper_1 = MPC_Wrapper.MPC_Wrapper(params, q)
    mpc_wrapper_2 = MPC_Wrapper.MPC_Wrapper(params, q)

    # Update gait
    gait.updateGait(0, 20, q[0:7, 0:1], 0)

    # Compute target footstep based on current and reference velocities
    footstepPlanner_1.updateFootsteps(False, 20, q[:, 0:1], h_v_1[0:6, 0:1].copy(), v_ref_1[0:6, 0])
    footstepPlanner_2.updateFootsteps(False, 20, q[:, 0:1], h_v_2[0:6, 0:1].copy(), v_ref_2[0:6, 0])

    # Run state planner (outputs the reference trajectory of the base)
    statePlanner_1.computeReferenceStates(q[0:7, 0:1].copy(), h_v_1[0:6, 0:1].copy(), v_ref_1[0:6, 0:1].copy(), 0.0)
    xref_1 = statePlanner_1.getReferenceStates()
    statePlanner_2.computeReferenceStates(q[0:7, 0:1].copy(), h_v_2[0:6, 0:1].copy(), v_ref_2[0:6, 0:1].copy(), 0.0)
    xref_2 = statePlanner_2.getReferenceStates()

    fsteps_1 = footstepPlanner_1.getFootsteps()
    fsteps_2 = footstepPlanner_2.getFootsteps()
    cgait = gait.getCurrentGait()

    xref_test = xref_2.copy()
    xref_test[0, :] *= -1.0
    xref_test[1, :] *= -1.0
    xref_test[5, :] *= -1.0
    xref_test[6, :] *= -1.0
    xref_test[7, :] *= -1.0
    xref_test[11, :] *= -1.0
    print(np.allclose(xref_1, xref_test))
    # print(xref_1 - xref_test)

    # Create gait matrix
    """cgait = np.zeros((params.N_gait, 4))
    for i in range(7):
        cgait[i, :] = np.array([1.0, 0.0, 0.0, 1.0])
    for i in range(8):
        cgait[7+i, :] = np.array([0.0, 1.0, 1.0, 0.0])
    cgait[15, :] = np.array([1.0, 0.0, 0.0, 1.0])
    for j in range(1, 10):
        for i in range(7):
            cgait[16*j+i, :] = np.array([1.0, 0.0, 0.0, 1.0])
        for i in range(8):
            cgait[16*j+7+i, :] = np.array([0.0, 1.0, 1.0, 0.0])
        cgait[16*j+15, :] = np.array([1.0, 0.0, 0.0, 1.0])"""

    # Create fsteps matrix
    """shoulders = np.zeros((3, 4))
    shoulders[0, :] = [0.1946, 0.1946, -0.1946, -0.1946]
    shoulders[1, :] = [0.14695, -0.14695, 0.14695, -0.14695]
    shoulders[2, :] = np.zeros(4)  # np.ones(4) * h_ref
    fsteps = np.zeros((params.N_gait, 12))
    for i in range(params.N_gait):
        for j in range(4):
            if cgait[i, j] == 1:
                fsteps[i, (3*j):(3*(j+1))] = shoulders[:, j]"""

    from IPython import embed
    embed()

    # Solve MPC problem once every k_mpc iterations of the main loop
    try:
        mpc_wrapper_1.solve(0, xref_1.copy(), fsteps_1.copy(), cgait.copy(), np.zeros((3, 4)))
        mpc_wrapper_2.solve(0, xref_2.copy(), fsteps_2.copy(), cgait.copy(), np.zeros((3, 4)))
    except ValueError:
        print("MPC Problem")

    # Retrieve reference contact forces in horizontal frame
    x_f_mpc_1 = mpc_wrapper_1.get_latest_result()
    x_f_mpc_2 = mpc_wrapper_2.get_latest_result()

    """from IPython import embed
    embed()"""

    # Solve MPC problem once every k_mpc iterations of the main loop
    try:
        mpc_wrapper_1.solve(1, xref_1.copy(), fsteps_1.copy(), cgait.copy(), np.zeros((3, 4)))
        mpc_wrapper_2.solve(1, xref_2.copy(), fsteps_2.copy(), cgait.copy(), np.zeros((3, 4)))
    except ValueError:
        print("MPC Problem")

    # Retrieve reference contact forces in horizontal frame
    x_f_mpc_1 = mpc_wrapper_1.get_latest_result()
    x_f_mpc_2 = mpc_wrapper_2.get_latest_result()

    """from IPython import embed
    embed()"""

    x_f_mpc_1 = np.hstack((np.vstack((xref_1[:, 0:1].copy(), np.zeros((12, 1)))), x_f_mpc_1.copy()))
    x_f_mpc_2 = np.hstack((np.vstack((xref_2[:, 0:1].copy(), np.zeros((12, 1)))), x_f_mpc_2.copy()))

    index6 = [1, 3, 5, 2, 4, 6]
    index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    coeff = [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
    titles = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    N = x_f_mpc_1.shape[1]
    plt.figure()
    for i in range(6):
        plt.subplot(3, 2, index6[i])
        h1, = plt.plot(x_f_mpc_1[i, :N], "b", linewidth=2)
        h2, = plt.plot(coeff[i] * x_f_mpc_2[i, :N], "r", linewidth=2)
        h3, = plt.plot(xref_1[i, :N], "forestgreen", linestyle="--", marker='x', color="g", linewidth=2)
        plt.xlabel("Time [s]")
        # plt.legend([h1, h2, h3], ["Output trajectory of classic MPC", "Output trajectory of croco MPC", "Input trajectory of planner"])
        plt.title("Predicted trajectory for " + titles[i])
    plt.suptitle("Analysis of trajectories in position and orientation computed by the MPC")

    plt.figure()
    for i in range(6):
        plt.subplot(3, 2, index6[i])
        h1, = plt.plot(x_f_mpc_1[i+6, :N], "b", linewidth=2)
        h2, = plt.plot(coeff[i] * x_f_mpc_2[i+6, :N], "r", linewidth=2)
        h3, = plt.plot(xref_1[i+6, :N], "forestgreen", linestyle="--", marker='x', color="g", linewidth=2)
        plt.xlabel("Time [s]")
        #plt.legend([h1, h2, h3], ["Output trajectory of classic MPC", "Output trajectory of croco MPC", "Input trajectory of planner"])
        plt.title("Predicted trajectory for velocity in " + titles[i])
    plt.suptitle("Analysis of trajectories of linear and angular velocities computed by the MPC")

    lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
    lgd2 = ["FL", "FR", "HL", "HR"]
    reorder = [9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]
    plt.figure()
    for i in range(12):
        if i == 0:
            ax0 = plt.subplot(3, 4, index12[i])
        else:
            plt.subplot(3, 4, index12[i], sharex=ax0)
        h1, = plt.plot(x_f_mpc_1[i+12, 1:(N+1)], "b", linewidth=3)
        h2, = plt.plot(coeff[i % 3] * x_f_mpc_2[reorder[i]+12, 1:(N+1)], "r", linewidth=3)
        # h3, = plt.plot(x_f_mpc_1[i+12, 1:(N+1)] - coeff[i % 3] * x_f_mpc_2[reorder[i]+12, 1:(N+1)], "g", linewidth=3)
        plt.xlabel("Time [s]")
        plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [N]")
        """if (i % 3) == 2:
            plt.ylim([-0.0, 26.0])
        else:
            plt.ylim([-26.0, 26.0])"""
        #plt.legend([h1, h2], ["Classic", "Croco"])
    plt.suptitle("Contact forces (MPC command)")

    plt.figure()
    for i in range(6):
        plt.subplot(3, 2, index6[i])
        h1, = plt.plot(x_f_mpc_1[i, :N] - coeff[i] * x_f_mpc_2[i, :N], "g", linewidth=2)
        plt.xlabel("Time [s]")
    plt.suptitle("Analysis of trajectories in position and orientation computed by the MPC")

    plt.figure()
    for i in range(6):
        plt.subplot(3, 2, index6[i])
        h1, = plt.plot(x_f_mpc_1[i+6, :N] - coeff[i] * x_f_mpc_2[i+6, :N], "g", linewidth=2)
        plt.xlabel("Time [s]")
    plt.suptitle("Analysis of trajectories of linear and angular velocities computed by the MPC")

    plt.figure()
    for i in range(12):
        if i == 0:
            ax0 = plt.subplot(3, 4, index12[i])
        else:
            plt.subplot(3, 4, index12[i], sharex=ax0)
        h3, = plt.plot(x_f_mpc_1[i+12, 1:(N+1)] - coeff[i % 3] * x_f_mpc_2[reorder[i]+12, 1:(N+1)], "g", linewidth=3)
        plt.xlabel("Time [s]")
        plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [N]")
    plt.suptitle("Contact forces (MPC command)")

    plt.figure()
    off = [9, 9, 9, 3, 3, 3]
    for i in range(6):
        if i == 0:
            ax0 = plt.subplot(3, 2, index6[i])
        else:
            plt.subplot(3, 2, index6[i], sharex=ax0)
        plt.plot(x_f_mpc_1[i+12, 1:(N+1)] - coeff[i % 3] * x_f_mpc_2[reorder[i]+12, 1:(N+1)] + (x_f_mpc_1[i+off[i]+12, 1:(N+1)] - coeff[i % 3] * x_f_mpc_2[reorder[i+off[i]]+12, 1:(N+1)]), "g", linewidth=3)
        plt.xlabel("Time [s]")
        plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [N]")
    plt.suptitle("Contact forces (MPC command)")

    plt.show(block=True)

def test_3(typeMPC=True):
    # Position of base in world frame
    q = np.zeros((19, 1))
    q[0:7, 0] = np.array([0.0, 0.0, h_ref, 0.0, 0.0, 0.0, 1.0])
    q[7:, 0] = np.array([0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7, +1.4])

    # Velocity of base in horizontal frame
    h_v_1 = np.zeros((18, 1))
    h_v_2 = np.zeros((18, 1))
    h_v_1[0, 0] = 0.1
    h_v_2[0, 0] = - h_v_1[0, 0]

    # Velocity reference of base in horizontal frame
    v_ref_1 = np.zeros((18, 1))
    v_ref_2 = np.zeros((18, 1))
    v_ref_1[0, 0] = 0.1
    v_ref_2[0, 0] = - v_ref_1[0, 0]

    # Params
    params = lqrw.Params()  # Object that holds all controller parameters
    params.dt_mpc = dt_mpc
    params.T_mpc = T_mpc
    params.h_ref = h_ref
    params.enable_multiprocessing = False

    # State planner
    statePlanner_1 = lqrw.StatePlanner()
    statePlanner_1.initialize(params)
    statePlanner_2 = lqrw.StatePlanner()
    statePlanner_2.initialize(params)

    # MPC wrapper
    mpc_wrapper_1 = MPC_Wrapper.MPC_Wrapper(params, q)
    mpc_wrapper_2 = MPC_Wrapper.MPC_Wrapper(params, q)

    # Run state planner (outputs the reference trajectory of the base)
    statePlanner_1.computeReferenceStates(q[0:7, 0:1].copy(), h_v_1[0:6, 0:1].copy(), v_ref_1[0:6, 0:1].copy(), 0.0)
    xref_1 = statePlanner_1.getReferenceStates()
    statePlanner_2.computeReferenceStates(q[0:7, 0:1].copy(), h_v_2[0:6, 0:1].copy(), v_ref_2[0:6, 0:1].copy(), 0.0)
    xref_2 = statePlanner_2.getReferenceStates()

    # Create gait matrix
    cgait = np.zeros((N_gait, 4))
    for i in range(7):
        cgait[i, :] = np.array([1.0, 0.0, 0.0, 1.0])
    for i in range(8):
        cgait[7+i, :] = np.array([0.0, 1.0, 1.0, 0.0])
    cgait[15, :] = np.array([1.0, 0.0, 0.0, 1.0])

    # Create fsteps matrix
    shoulders = np.zeros((3, 4))
    shoulders[0, :] = [0.1946, 0.1946, -0.1946, -0.1946]
    shoulders[1, :] = [0.14695, -0.14695, 0.14695, -0.14695]
    shoulders[2, :] = np.zeros(4)  # np.ones(4) * h_ref
    fsteps = np.zeros((N_gait, 12))
    for i in range(N_gait):
        for j in range(4):
            if cgait[i, j] == 1:
                fsteps[i, (3*j):(3*(j+1))] = shoulders[:, j]


    # Solve MPC problem once every k_mpc iterations of the main loop
    try:
        mpc_wrapper_1.solve(0, xref_1.copy(), fsteps.copy(), cgait.copy(), np.zeros((3,4)))
        mpc_wrapper_2.solve(0, xref_2.copy(), fsteps.copy(), cgait.copy(), np.zeros((3,4)))
    except ValueError:
        print("MPC Problem")

    # Retrieve reference contact forces in horizontal frame
    x_f_mpc_1 = mpc_wrapper_1.get_latest_result()
    x_f_mpc_2 = mpc_wrapper_2.get_latest_result()

    # Solve MPC problem once every k_mpc iterations of the main loop
    try:
        mpc_wrapper_1.solve(0, xref_1.copy(), fsteps.copy(), cgait.copy(), np.zeros((3,4)))
        mpc_wrapper_2.solve(0, xref_2.copy(), fsteps.copy(), cgait.copy(), np.zeros((3,4)))
    except ValueError:
        print("MPC Problem")

    # Retrieve reference contact forces in horizontal frame
    x_f_mpc_1 = mpc_wrapper_1.get_latest_result()
    x_f_mpc_2 = mpc_wrapper_2.get_latest_result()
    x_f_mpc_1 = np.hstack((np.vstack((xref_1[:, 0:1].copy(), np.zeros((12, 1)))), x_f_mpc_1.copy()))
    x_f_mpc_2 = np.hstack((np.vstack((xref_2[:, 0:1].copy(), np.zeros((12, 1)))), x_f_mpc_2.copy()))

    index6 = [1, 3, 5, 2, 4, 6]
    index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    coeff = [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
    titles = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    N = -1
    plt.figure()
    for i in range(6):
        plt.subplot(3, 2, index6[i])
        h1, = plt.plot(x_f_mpc_1[i, :N], "b", linewidth=2)
        h2, = plt.plot(coeff[i] * x_f_mpc_2[i, :N], "r", linewidth=2)
        h3, = plt.plot(xref_1[i, :N], "b", linestyle="--", marker='x', color="g", linewidth=2)
        plt.xlabel("Time [s]")
        # plt.legend([h1, h2, h3], ["Output trajectory of classic MPC", "Output trajectory of croco MPC", "Input trajectory of planner"])
        plt.title("Predicted trajectory for " + titles[i])
    plt.suptitle("Analysis of trajectories in position and orientation computed by the MPC")

    plt.figure()
    for i in range(6):
        plt.subplot(3, 2, index6[i])
        h1, = plt.plot(x_f_mpc_1[i+6, :N], "b", linewidth=2)
        h2, = plt.plot(coeff[i] * x_f_mpc_2[i+6, :N], "r", linewidth=2)
        h3, = plt.plot(xref_1[i+6, :N], "b", linestyle="--", marker='x', color="g", linewidth=2)
        plt.xlabel("Time [s]")
        #plt.legend([h1, h2, h3], ["Output trajectory of classic MPC", "Output trajectory of croco MPC", "Input trajectory of planner"])
        plt.title("Predicted trajectory for velocity in " + titles[i])
    plt.suptitle("Analysis of trajectories of linear and angular velocities computed by the MPC")

    lgd1 = ["Ctct force X", "Ctct force Y", "Ctct force Z"]
    lgd2 = ["FL", "FR", "HL", "HR"]
    reorder = [9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]
    plt.figure()
    for i in range(12):
        if i == 0:
            ax0 = plt.subplot(3, 4, index12[i])
        else:
            plt.subplot(3, 4, index12[i], sharex=ax0)
        h1, = plt.plot(x_f_mpc_1[i+12, 1:(N+1)], "b", linewidth=3)
        h2, = plt.plot(coeff[i % 3] * x_f_mpc_2[reorder[i]+12, 1:(N+1)], "r", linewidth=3)
        # h3, = plt.plot(x_f_mpc_1[i+12, 1:(N+1)] - coeff[i % 3] * x_f_mpc_2[reorder[i]+12, 1:(N+1)], "g", linewidth=3)
        plt.xlabel("Time [s]")
        plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)]+" [N]")
        """if (i % 3) == 2:
            plt.ylim([-0.0, 26.0])
        else:
            plt.ylim([-26.0, 26.0])"""
        #plt.legend([h1, h2], ["Classic", "Croco"])
    plt.suptitle("Contact forces (MPC command)")

    plt.show()

test_1()
#test_2()
#test_3()