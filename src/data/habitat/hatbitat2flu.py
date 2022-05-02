import numpy as np
from src.utils.transform import Transform as Tf


# class Habitat2FLU:
#     # habitat agent and global reference frame are oriented with x-left, y-up and z-backward.
#     tf_hab__flu = Tf.from_euler_trans(seq='xyz', angles=np.array([-90, 90, 0]), degrees=True,
#                                       trans=np.zeros(3)).as_matrix()
#     rot_hab__flu = Tf.from_matrix(tf_hab__flu).as_rot_mat()
#
#     tf_flu__hab = Tf.from_matrix(tf_hab__flu).inv().as_matrix()
#     rot_flu__hab = Tf.from_matrix(tf_flu__hab).as_rot_mat()
#
#     @classmethod
#     def state_habitat_2_state_flu(cls, state_habitat: np.ndarray):
#         """
#         Change state with global habitat reference frame to global flu.
#
#         param state_habitat: shape: (19).
#
#         return: state_flu: shape : (19).
#         """
#         trans, quat, lin_vel, ang_vel, lin_acc, ang_acc = (state_habitat[..., :3],
#                                                            state_habitat[..., 3:7],
#                                                            state_habitat[..., 7:10],
#                                                            state_habitat[..., 10:13],
#                                                            state_habitat[..., 13:16],
#                                                            state_habitat[..., 16:19])
#         tf_global_agent = Tf.from_quat_trans(quat, trans).as_matrix()
#         tf_global_flu__agent_flu = cls.tf_flu__hab @ tf_global_agent @ cls.tf_hab__flu
#         quat_flu_flu, trans_flu_flu = Tf.from_matrix(tf_global_flu__agent_flu).as_quat_trans()
#
#         lin_vel__flu = (cls.rot_flu__hab @ lin_vel[..., np.newaxis])[..., 0]  # in global
#         lin_acc__flu = (cls.rot_flu__hab @ lin_acc[..., np.newaxis])[..., 0]  # in global
#
#         rot_agent_flu__global = Tf.from_matrix(tf_global_agent @ cls.tf_hab__flu).as_rot_mat().swapaxes(-1, -2)
#         ang_vel__flu = (rot_agent_flu__global @ ang_vel[..., np.newaxis])[..., 0]  # in body
#         ang_acc__flu = (rot_agent_flu__global @ ang_acc[..., np.newaxis])[..., 0]  # in body
#
#         state_flu = np.concatenate([trans_flu_flu, quat_flu_flu, lin_vel__flu, ang_vel__flu,
#                                     lin_acc__flu,
#                                     ang_acc__flu,
#                                     ], axis=-1)
#         return state_flu
#
#     @classmethod
#     def state_flu_2_state_habitat(cls, state_flu: np.ndarray):
#         """
#         Change state with global flu reference frame to global habitat.
#
#         param state_flu: shape: (19).
#
#         return: state_habitat: shape : (19).
#         """
#         trans_flu, quat_flu, lin_vel_flu, ang_vel_flu, lin_acc_flu, ang_acc_flu = (state_flu[..., :3],
#                                                                                    state_flu[..., 3:7],
#                                                                                    state_flu[..., 7:10],
#                                                                                    state_flu[..., 10:13],
#                                                                                    state_flu[..., 13:16],
#                                                                                    state_flu[..., 16:19])
#         tf_global_flu__agent_flu = Tf.from_quat_trans(quat_flu, trans_flu).as_matrix()
#         tf_global_agent = cls.tf_hab__flu @ tf_global_flu__agent_flu @ cls.tf_flu__hab
#         quat_hab, trans_hab = Tf.from_matrix(tf_global_agent).as_quat_trans()
#
#         lin_vel_hab = (cls.rot_hab__flu @ lin_vel_flu[..., np.newaxis])[..., 0]  # in global
#         lin_acc_hab = (cls.rot_hab__flu @ lin_acc_flu[..., np.newaxis])[..., 0]  # in global
#
#         rot_global__agent_flu = Tf.from_matrix(tf_global_agent @ cls.tf_hab__flu).as_rot_mat()
#         ang_vel_hab = (rot_global__agent_flu @ ang_vel_flu[..., np.newaxis])[..., 0]  # in body
#         ang_acc_hab = (rot_global__agent_flu @ ang_acc_flu[..., np.newaxis])[..., 0]  # in body
#
#         state_habitat = np.concatenate([trans_hab, quat_hab, lin_vel_hab, ang_vel_hab,
#                                         lin_acc_hab,
#                                         ang_acc_hab,
#                                         ], axis=-1)
#         return state_habitat


class Habitat2FLU:
    # habitat agent and global reference frame are oriented with x-left, y-up and z-backward.
    tf_hab__flu = Tf.from_euler_trans(seq='xyz', angles=np.array([-90, 90, 0]), degrees=True,
                                      trans=np.zeros(3)).as_matrix()
    rot_hab__flu = Tf.from_matrix(tf_hab__flu).as_rot_mat()

    tf_flu__hab = Tf.from_matrix(tf_hab__flu).inv().as_matrix()
    rot_flu__hab = Tf.from_matrix(tf_flu__hab).as_rot_mat()

    @classmethod
    def state_habitat_2_state_flu(cls, state_habitat: np.ndarray):
        """
        Change state with global habitat reference frame to global flu.

        param state_habitat: shape: (19).

        return: state_flu: shape : (19).
        """
        trans = state_habitat[..., :3]
        quat = state_habitat[..., 3:7]
        lin_vel = state_habitat[..., 7:10]
        ang_vel = state_habitat[..., 10:13]
        # lin_acc = state_habitat[..., 13:16]
        # ang_acc = state_habitat[..., 16:19]

        tf_global_agent = Tf.from_quat_trans(quat, trans).as_matrix()
        tf_global_flu__agent_flu = cls.tf_flu__hab @ tf_global_agent @ cls.tf_hab__flu
        quat_flu_flu, trans_flu_flu = Tf.from_matrix(tf_global_flu__agent_flu).as_quat_trans()

        lin_vel__flu = (cls.rot_flu__hab @ lin_vel[..., np.newaxis])[..., 0]  # in global
        # lin_acc__flu = (cls.rot_flu__hab @ lin_acc[..., np.newaxis])[..., 0]  # in global

        rot_agent_flu__global = Tf.from_matrix(tf_global_agent @ cls.tf_hab__flu).as_rot_mat().swapaxes(-1, -2)
        ang_vel__flu = (rot_agent_flu__global @ ang_vel[..., np.newaxis])[..., 0]  # in body
        # ang_acc__flu = (rot_agent_flu__global @ ang_acc[..., np.newaxis])[..., 0]  # in body

        state_flu = np.concatenate([trans_flu_flu, quat_flu_flu, lin_vel__flu, ang_vel__flu,
                                    # lin_acc__flu,
                                    # ang_acc__flu,
                                    ], axis=-1)
        return state_flu

    @classmethod
    def state_flu_2_state_habitat(cls, state_flu: np.ndarray):
        """
        Change state with global flu reference frame to global habitat.

        param state_flu: shape: (19).

        return: state_habitat: shape : (19).
        """
        trans_flu = state_flu[..., :3]
        quat_flu = state_flu[..., 3:7]
        lin_vel_flu = state_flu[..., 7:10]
        ang_vel_flu = state_flu[..., 10:13]
        # lin_acc_flu = state_flu[..., 13:16]
        # ang_acc_flu = state_flu[..., 16:19]

        tf_global_flu__agent_flu = Tf.from_quat_trans(quat_flu, trans_flu).as_matrix()
        tf_global_agent = cls.tf_hab__flu @ tf_global_flu__agent_flu @ cls.tf_flu__hab
        quat_hab, trans_hab = Tf.from_matrix(tf_global_agent).as_quat_trans()

        lin_vel_hab = (cls.rot_hab__flu @ lin_vel_flu[..., np.newaxis])[..., 0]  # in global
        # lin_acc_hab = (cls.rot_hab__flu @ lin_acc_flu[..., np.newaxis])[..., 0]  # in global

        rot_global__agent_flu = Tf.from_matrix(tf_global_agent @ cls.tf_hab__flu).as_rot_mat()
        ang_vel_hab = (rot_global__agent_flu @ ang_vel_flu[..., np.newaxis])[..., 0]  # in body
        # ang_acc_hab = (rot_global__agent_flu @ ang_acc_flu[..., np.newaxis])[..., 0]  # in body

        state_habitat = np.concatenate([trans_hab, quat_hab, lin_vel_hab, ang_vel_hab,
                                        # lin_acc_hab,
                                        # ang_acc_hab,
                                        ], axis=-1)
        return state_habitat


class Habitat2FLUACC:
    # habitat agent and global reference frame are oriented with x-left, y-up and z-backward.
    tf_hab__flu = Tf.from_euler_trans(seq='xyz', angles=np.array([-90, 90, 0]), degrees=True,
                                      trans=np.zeros(3)).as_matrix()
    rot_hab__flu = Tf.from_matrix(tf_hab__flu).as_rot_mat()

    tf_flu__hab = Tf.from_matrix(tf_hab__flu).inv().as_matrix()
    rot_flu__hab = Tf.from_matrix(tf_flu__hab).as_rot_mat()

    @classmethod
    def state_habitat_2_state_flu(cls, state_habitat: np.ndarray):
        """
        Change state with global habitat reference frame to global flu.

        param state_habitat: shape: (19).

        return: state_flu: shape : (19).
        """
        trans = state_habitat[..., :3]
        quat = state_habitat[..., 3:7]
        lin_vel = state_habitat[..., 7:10]
        ang_vel = state_habitat[..., 10:13]
        lin_acc = state_habitat[..., 13:16]
        ang_acc = state_habitat[..., 16:19]

        tf_global_agent = Tf.from_quat_trans(quat, trans).as_matrix()
        tf_global_flu__agent_flu = cls.tf_flu__hab @ tf_global_agent @ cls.tf_hab__flu
        quat_flu_flu, trans_flu_flu = Tf.from_matrix(tf_global_flu__agent_flu).as_quat_trans()

        lin_vel__flu = (cls.rot_flu__hab @ lin_vel[..., np.newaxis])[..., 0]  # in global
        lin_acc__flu = (cls.rot_flu__hab @ lin_acc[..., np.newaxis])[..., 0]  # in global

        rot_agent_flu__global = Tf.from_matrix(tf_global_agent @ cls.tf_hab__flu).as_rot_mat().swapaxes(-1, -2)
        ang_vel__flu = (rot_agent_flu__global @ ang_vel[..., np.newaxis])[..., 0]  # in body
        ang_acc__flu = (rot_agent_flu__global @ ang_acc[..., np.newaxis])[..., 0]  # in body

        state_flu = np.concatenate([trans_flu_flu, quat_flu_flu, lin_vel__flu, ang_vel__flu,
                                    lin_acc__flu,
                                    ang_acc__flu,
                                    ], axis=-1)
        return state_flu

    @classmethod
    def state_flu_2_state_habitat(cls, state_flu: np.ndarray):
        """
        Change state with global flu reference frame to global habitat.

        param state_flu: shape: (19).

        return: state_habitat: shape : (19).
        """
        trans_flu = state_flu[..., :3]
        quat_flu = state_flu[..., 3:7]
        lin_vel_flu = state_flu[..., 7:10]
        ang_vel_flu = state_flu[..., 10:13]
        lin_acc_flu = state_flu[..., 13:16]
        ang_acc_flu = state_flu[..., 16:19]

        tf_global_flu__agent_flu = Tf.from_quat_trans(quat_flu, trans_flu).as_matrix()
        tf_global_agent = cls.tf_hab__flu @ tf_global_flu__agent_flu @ cls.tf_flu__hab
        quat_hab, trans_hab = Tf.from_matrix(tf_global_agent).as_quat_trans()

        lin_vel_hab = (cls.rot_hab__flu @ lin_vel_flu[..., np.newaxis])[..., 0]  # in global
        lin_acc_hab = (cls.rot_hab__flu @ lin_acc_flu[..., np.newaxis])[..., 0]  # in global

        rot_global__agent_flu = Tf.from_matrix(tf_global_agent @ cls.tf_hab__flu).as_rot_mat()
        ang_vel_hab = (rot_global__agent_flu @ ang_vel_flu[..., np.newaxis])[..., 0]  # in body
        ang_acc_hab = (rot_global__agent_flu @ ang_acc_flu[..., np.newaxis])[..., 0]  # in body

        state_habitat = np.concatenate([trans_hab, quat_hab, lin_vel_hab, ang_vel_hab,
                                        lin_acc_hab,
                                        ang_acc_hab,
                                        ], axis=-1)
        return state_habitat


def _test():
    """
    Assert reversibility of the transformation.
    """
    for i in range(1000):
        state_flu = np.random.randn(19)
        state_flu[3:7] = state_flu[3:7] / np.linalg.norm(state_flu[3:7])

        state_habitat = Habitat2FLUACC.state_flu_2_state_habitat(state_flu)
        state_flu_2 = Habitat2FLUACC.state_habitat_2_state_flu(state_habitat)
        state_flu_2_minus = np.copy(state_flu_2)
        state_flu_2_minus[3:7] = -state_flu_2[3:7]
        assert np.all(np.isclose(state_flu[..., :19], state_flu_2[..., :19])) or \
               np.all(np.isclose(state_flu[..., :19], state_flu_2_minus[..., :19]))
    print(f'Reversibility test passed.')


def main():
    _test()


if __name__ == '__main__':
    main()
