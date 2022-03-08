from IPython import embed
import numpy as np
from example_robot_data.robots_loader import Solo12Loader
import pinocchio as pin

# Parameters of the Invkin 
l = 0.1946 * 2
L = 0.14695 * 2
h = 0.18
q_init = [0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, 0.7, -1.4, -0.0, 0.7, -1.4]

# Load robot model and data
# Initialisation of the Gepetto viewer
Solo12Loader.free_flyer = True
solo = Solo12Loader().robot
q = solo.q0.reshape((-1, 1))
q[7:, 0] = q_init  # Initial angular positions of actuators

# Get foot indexes
BASE_ID = solo.model.getFrameId('base_link')
foot_ids = [solo.model.getFrameId('FL_FOOT'),
            solo.model.getFrameId('FR_FOOT'),
            solo.model.getFrameId('HL_FOOT'),
            solo.model.getFrameId('HR_FOOT')]

# Init variables
Jf = np.zeros((12, 18))
posf = np.zeros((4, 3))

pos_ref = np.array([0.0, 0.0, h])
rot_ref = np.eye(3)
posf_ref = np.array([[l * 0.5, l * 0.5, -l * 0.5, -l * 0.5],
                     [L * 0.5, -L * 0.5, L * 0.5, -L * 0.5],
                     [0.0, 0.0, 0.0, 0.0]])

e_basispos = 1.0
while np.any(np.abs(e_basispos) > 0.001):

    # Update model and data of the robot
    pin.computeJointJacobians(solo.model, solo.data, q)
    pin.forwardKinematics(solo.model, solo.data, q, np.zeros(
        solo.model.nv), np.zeros(solo.model.nv))
    pin.updateFramePlacements(solo.model, solo.data)

    # Get data required by IK with Pinocchio
    pos = solo.com()
    rot = solo.data.oMf[BASE_ID].rotation
    Jbasis = pin.getFrameJacobian(solo.model, solo.data, BASE_ID, pin.LOCAL_WORLD_ALIGNED)[:3]
    Jwbasis = pin.getFrameJacobian(solo.model, solo.data, BASE_ID, pin.LOCAL_WORLD_ALIGNED)[3:]
    for i_ee in range(4):
        idx = int(foot_ids[i_ee])
        posf[i_ee, :] = solo.data.oMf[idx].translation
        Jf[(3*i_ee):(3*(i_ee+1)), :] = pin.getFrameJacobian(solo.model, solo.data, idx, pin.LOCAL_WORLD_ALIGNED)[:3]

    # Compute errors
    e_basispos = pos_ref - pos
    e_basisrot = -rot_ref @ pin.log3(rot_ref.T @ rot)
    pfeet_err = []
    for i in range(4):
        pfeet_err.append(posf_ref[:, i] - posf[i, :])

    # Set up matrices
    J = np.vstack([Jbasis, Jwbasis, Jf])
    x_err = np.concatenate([e_basispos, e_basisrot]+pfeet_err)

    # Loop
    q = pin.integrate(solo.model, q, 0.01 * np.linalg.pinv(J) @ x_err)

# Compute final position of CoM
#q[7:] = np.array([0.0, 0.764, -1.407, 0.0, 0.76407, -1.4, 0.0, 0.76407, -1.407, 0.0, 0.764, -1.407])
pin.forwardKinematics(solo.model, solo.data, q, np.zeros(
    solo.model.nv), np.zeros(solo.model.nv))
pos = solo.com()
for i_ee in range(4):
    idx = int(foot_ids[i_ee])
    posf[i_ee, :] = solo.data.oMf[idx].translation
print("Final error: ", pos_ref - pos)
print("Com: ", pos)
print("Feet: ", posf)
print("Joints: ", q[7:])

solo.initViewer(loadModel=True)
if ('viewer' in solo.viz.__dict__):
    solo.viewer.gui.addFloor('world/floor')
    solo.viewer.gui.setRefreshIsSynchronous(False)
solo.display(q)
