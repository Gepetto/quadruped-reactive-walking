import numpy as np
from example_robot_data.robots_loader import Solo12Loader
import pinocchio as pin
from scipy.special import erf as erf
import pybullet as pyb


class ContactDetector:
    def __init__(self, params):
        # Load robot model and data
        # Initialisation of the Gepetto viewer
        Solo12Loader.free_flyer = True
        self.r = Solo12Loader().robot

        # IDs of feet frames
        self.foot_ids = [
            self.r.model.getFrameId("FL_FOOT"),
            self.r.model.getFrameId("FR_FOOT"),
            self.r.model.getFrameId("HL_FOOT"),
            self.r.model.getFrameId("HR_FOOT"),
        ]

        self.q = np.zeros((19, 1))
        self.NLE = np.zeros((12, 1))
        self.Jf = np.zeros((12, 12))

        self.log_f = np.zeros((params.N_SIMULATION, 12))
        self.log_v = np.zeros((params.N_SIMULATION, 12))
        self.log_ctc = np.zeros((params.N_SIMULATION, 4))
        self.log_gait = np.zeros((params.N_SIMULATION, 4))
        self.log_qmes = np.zeros((params.N_SIMULATION, 12))
        self.log_qdes = np.zeros((params.N_SIMULATION, 12))

        self.contact_detection = False  # True when a new contact is detected
        self.k_since_flying = np.zeros(4)
        self.k_since_late = np.zeros(4)

        # Probabilities of contact
        self.P_f = np.zeros((params.N_SIMULATION, 4))  # Contact forces
        self.P_err = np.zeros((params.N_SIMULATION, 4))  # Tracking error
        self.P_v = np.zeros((params.N_SIMULATION, 4))  # Feet velocity
        self.P_cc = np.zeros((params.N_SIMULATION, 4))  # Contact plan
        self.P_tot = np.zeros((params.N_SIMULATION, 4))  # Final contact probability

        # If we are in simulation we can log the truth about contact states
        self.is_simulation = params.SIMULATION
        self.log_ctc_truth = np.zeros((params.N_SIMULATION, 4))
        self.log_v_truth = np.zeros((params.N_SIMULATION, 4))
        self.log_f_truth = np.zeros((params.N_SIMULATION, 12))

        self.dt_mpc = params.dt_mpc
        self.k_mpc = int(params.dt_mpc / params.dt_wbc)

    def getContactPoint(self, contactPoints):
        """Sort contacts points as there should be only one contact per foot
        and sometimes PyBullet detect several of them. There is one contact
        with a non zero force and the others have a zero contact force
        """

        for i in range(0, len(contactPoints)):
            # There may be several contact points for each foot but only one of them as a non zero normal force
            if contactPoints[i][9] != 0:
                return contactPoints[i]

        # If it returns 0 then it means there is no contact point with a non zero normal force (should not happen)
        return 0

    def getContactForces(self, device):

        # Info about contact points with the ground
        contactPoints_FL = pyb.getContactPoints(
            device.pyb_sim.robotId, device.pyb_sim.planeId, linkIndexA=3
        )  # Front left  foot
        contactPoints_FR = pyb.getContactPoints(
            device.pyb_sim.robotId, device.pyb_sim.planeId, linkIndexA=7
        )  # Front right foot
        contactPoints_HL = pyb.getContactPoints(
            device.pyb_sim.robotId, device.pyb_sim.planeId, linkIndexA=11
        )  # Hind left  foot
        contactPoints_HR = pyb.getContactPoints(
            device.pyb_sim.robotId, device.pyb_sim.planeId, linkIndexA=15
        )  # Hind right foot

        # Sort contacts points to get only one contact per foot
        contactPoints = []
        contactPoints.append(self.getContactPoint(contactPoints_FL))
        contactPoints.append(self.getContactPoint(contactPoints_FR))
        contactPoints.append(self.getContactPoint(contactPoints_HL))
        contactPoints.append(self.getContactPoint(contactPoints_HR))

        # Get total contact force for each foot
        cpt = 0
        forces = np.zeros(12)
        for contact in contactPoints:
            if not isinstance(contact, int):  # type(contact) != type(0):
                for i_direction in range(0, 3):
                    forces[3 * cpt + i_direction] = (
                        contact[9] * contact[7][i_direction]
                        + contact[10] * contact[11][i_direction]
                        + contact[12] * contact[13][i_direction]
                    )
            else:  # No contact for that foot
                forces[3 * cpt : 3 * (cpt + 1)] = 0.0
            cpt += 1

        return forces

    def P(self, x, mu, sigma):  # P(X <= x) for normal law (mu, sigma)
        return 0.5 * (1.0 + erf((x - mu) / (sigma * (2**0.5))))

    def run(self, k, gait, q, dq, tau, device, qdes):
        self.contact_detection = False
        gait_tmp = gait.matrix
        gait_tmp_past = gait.get_past_gait()

        # In simulation we can get the ground truth from the simulator
        if self.is_simulation and k > 0:
            # Log ground truth for contacts
            ctc_FL = pyb.getContactPoints(
                device.pyb_sim.robotId, device.pyb_sim.planeId, linkIndexA=3
            )  # Front left foot
            ctc_FR = pyb.getContactPoints(
                device.pyb_sim.robotId, device.pyb_sim.planeId, linkIndexA=7
            )  # Front right foot
            ctc_HL = pyb.getContactPoints(
                device.pyb_sim.robotId, device.pyb_sim.planeId, linkIndexA=11
            )  # Hind left foot
            ctc_HR = pyb.getContactPoints(
                device.pyb_sim.robotId, device.pyb_sim.planeId, linkIndexA=15
            )  # Hind right foot

            ctc_FL_block = pyb.getContactPoints(
                device.pyb_sim.robotId, 26, linkIndexA=3
            )  # Front left foot
            ctc_FR_block = pyb.getContactPoints(
                device.pyb_sim.robotId, 26, linkIndexA=7
            )  # Front right foot
            ctc_HL_block = pyb.getContactPoints(
                device.pyb_sim.robotId, 26, linkIndexA=11
            )  # Hind left foot
            ctc_HR_block = pyb.getContactPoints(
                device.pyb_sim.robotId, 26, linkIndexA=15
            )  # Hind right foot
            a = np.array([len(ctc_FL), len(ctc_FR), len(ctc_HL), len(ctc_HR)]) > 0
            b = (
                np.array(
                    [
                        len(ctc_FL_block),
                        len(ctc_FR_block),
                        len(ctc_HL_block),
                        len(ctc_HR_block),
                    ]
                )
                > 0
            )
            self.log_ctc_truth[k, :] = np.logical_or(a, b).astype(int)

            # Log ground truth for feet velocities
            linkStates = pyb.getLinkStates(
                device.pyb_sim.robotId, [3, 7, 11, 15], computeLinkVelocity=1
            )
            for i in range(4):
                self.log_v_truth[k, i] = (
                    linkStates[i][6][0] ** 2
                    + linkStates[i][6][1] ** 2
                    + linkStates[i][6][2] ** 2
                ) ** 0.5

            # Log ground truth for contact forces
            self.log_f_truth[k, :] = self.getContactForces(device)

            # Late and early contact detection
            """for i in range(4):
                if self.k_since_flying[i] > 100:
                    if self.log_ctc_truth[k, i] - self.log_ctc_truth[k-1, i] > 0.5 and self.log_ctc_truth[k, i] == 1 and gait_tmp[0, i] == 0:
                        # Contact detected before hardcoded timing
                        # print("Early contact foot ", i, "at step ", k)
                        j = 0
                        while(gait_tmp[j, i] == 0):
                            gait_tmp[j, i] = 1
                            j += 1
                        self.contact_detection = True
                        gait.set_current_gait(gait_tmp)
                        gait.set_new_phase(True)
                    elif self.log_ctc_truth[k, i] == 0 and gait_tmp[0, i] == 1 and gait_tmp_past[0, i] == 0:
                        # Contact detected after hardcoded timing
                        # print("Late contact foot ", i, "at step ", k)
                        gait_tmp[0, i] = 0
                        gait.setLate(i)
                        gait.set_current_gait(gait_tmp)"""

        """# Time spent in flying phase
        self.k_since_flying += (1 - gait_tmp[0, :])
        self.k_since_flying *= (1 - gait_tmp[0, :])"""

        # Retrieve state with orientation as quaternion
        self.q[:3, 0] = q[:3, 0]
        self.q[3:7, 0] = pin.Quaternion(pin.rpy.rpyToMatrix(q[3:6, 0])).coeffs()
        self.q[7:, 0] = q[6:, 0]

        self.log_qmes[k, :] = q[6:, 0]
        self.log_qdes[k, :] = qdes

        # Update model and jacobians
        pin.computeJointJacobians(self.r.model, self.r.data, self.q)
        pin.forwardKinematics(self.r.model, self.r.data, self.q, dq)

        # Gravitational and non linear forces
        pin.rnea(self.r.model, self.r.data, self.q, dq, np.zeros((18, 1)))
        self.NLE[:, 0] = self.r.data.tau[6:]

        # Feet jacobians
        for i in range(4):
            self.Jf[(3 * i) : (3 * (i + 1)), :] = pin.getFrameJacobian(
                self.r.model, self.r.data, self.foot_ids[i], pin.LOCAL_WORLD_ALIGNED
            )[:3, 6:]

        # Compute forces
        f = -np.linalg.pinv(self.Jf.transpose()) @ (tau - self.NLE)
        self.log_f[k, :] = f.ravel()

        # Retrieve spatial velocity
        for i in range(4):
            v = pin.getFrameVelocity(
                self.r.model, self.r.data, self.foot_ids[i], pin.LOCAL_WORLD_ALIGNED
            )
            self.log_v[k, (3 * i) : (3 * (i + 1))] = v.linear[:3]

        # Retrieve phase duration and remaining time
        for i in range(4):
            status = gait.get_gait_coeff(0, i)
            t_ph = gait.get_phase_duration(0, i)
            t_rem = gait.get_remaining_time(0, i)
            if t_rem <= (t_ph * 0.5 / self.dt_mpc):
                ctc = -np.ceil(t_rem)
            else:
                ctc = np.ceil((t_ph / self.dt_mpc) - t_rem)
            ctc *= self.k_mpc
            """if k % self.k_mpc != 0:
                ctc += (k % self.k_mpc)
            else:
            ctc += self.k_mpc"""
            ctc += k % self.k_mpc
            ctc += self.k_since_late[i]

            self.log_ctc[k, i] = ctc
            self.log_gait[k, i] = status

            """if k % 20 == 15:
                if status:
                    print("Foot ", i, " in contact")
                else:
                    print("Foot ", i, " in swing")
                print("Phase duration: ", t_ph)
                print("Remaining: ", t_rem)
                print("Before/After switch: ", ctc)"""

            """if i==0:
                from IPython import embed
                embed()"""

            """if k > 60:# % 20 == 15:
                from IPython import embed
                embed()"""

        # Probability computation
        for i in range(4):
            # Contact force
            f = self.log_f[k, 3 * i + 2]
            self.P_f[k, i] = self.P(f, 3.3, 2.06 * 0.65)

            # Tracking error
            err = np.abs(self.log_qdes[k, 3 * i + 2] - self.log_qmes[k, 3 * i + 2])
            self.P_err[k, i] = self.P(err, 0.04, 0.01)

            # Feet velocity
            v_norm = (
                self.log_v[k, 3 * i] ** 2
                + self.log_v[k, 3 * i + 1] ** 2
                + self.log_v[k, 3 * i + 2] ** 2
            ) ** 0.5
            self.P_v[k, i] = 1 - self.P(v_norm, 0.18, 0.074)  # 0.121, 0.0411

            # Contact plan
            P_0to1 = 0.5 * (1.0 + erf((self.log_ctc[k, i] - 0.0) / (80.0 * (2**0.5))))
            sw = np.any(
                [
                    np.logical_and(self.log_ctc[k, i] >= 0, self.log_gait[k, i] == 1),
                    np.logical_and(self.log_ctc[k, i] < 0, self.log_gait[k, i] == 0),
                    self.k_since_late[i] > 0,
                ]
            ).astype(int)
            self.P_cc[k, i] = sw * P_0to1

            # Product of probabilities for final estimation
            self.P_tot[k, i] = np.max(
                [self.P_f[k, i] * self.P_v[k, i]]
            )  # * self.P_cc[k, i]
            # self.P_tot[k, i] = np.max([self.log_ctc_truth[k, i] * self.P_v[k, i]])

            # Late and early contact detection
            if self.k_since_flying[i] > 160 and k > 360:
                p = 0.35
                if (self.P_tot[k, i] >= p) and (gait_tmp[0, i] == 0):
                    # Contact detected before hardcoded timing
                    # print("Early contact foot ", i, "at step ", k)
                    j = 0
                    while gait_tmp[j, i] == 0:
                        gait_tmp[j, i] = 1
                        j += 1
                    self.contact_detection = True
                    gait.set_current_gait(gait_tmp)
                    gait.set_new_phase(True)
                    # from IPython import embed
                    # embed()
                elif (
                    (self.P_tot[k, i] < p)
                    and (gait_tmp[0, i] == 1)
                    and (gait_tmp_past[0, i] == 0)
                ):
                    # Contact detected after hardcoded timing
                    # print("Late contact foot ", i, "at step ", k)
                    self.k_since_late[i] += self.k_mpc
                    gait_tmp[0, i] = 0
                    gait.setLate(i)
                    gait.set_current_gait(gait_tmp)
                    # from IPython import embed
                    # embed()

        self.log_gait[k, :] = gait.matrix[0, :]

        # Time spent in flying phase
        self.k_since_flying += 1 - gait_tmp[0, :]
        self.k_since_flying *= 1 - gait_tmp[0, :]

        # Reset k_since_late once contact is detected
        self.k_since_late *= 1 - gait_tmp[0, :]

        """print("----")
        print(self.P_tot[k, :])
        print(gait_tmp[0, :])"""

        return f


if __name__ == "__main__":

    cd = ContactDetector()
    q = np.zeros((19, 1))
    q[6, 0] = 1.0
    dq = np.zeros((18, 1))
    tau = np.zeros((12, 1))
    f = cd.run(q, dq, tau)
    print(f)
