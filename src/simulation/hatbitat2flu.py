import numpy as np
from src.utils.transform import Transform


class Habitat2FLU:
    # habitat agent and global reference frame are oriented with x-left, y-up and z-backward.
    tf_habitat__flu = Transform.from_euler_trans(seq='xyz', angles=np.array([-90, 90, 0]), degrees=True,
                                                 trans=np.zeros(3)).as_matrix()
    rot_habitat__flu = Transform.from_matrix(tf_habitat__flu).as_rot_mat()

    tf_flu__habitat = Transform.from_matrix(tf_habitat__flu).inv().as_matrix()
    rot_flu__habitat = Transform.from_matrix(tf_flu__habitat).as_rot_mat()

    @classmethod
    def state_habitat_2_state_flu(cls, state_habitat: np.ndarray):
        """
        Change state with global habitat reference frame to global flu.

        param state_habitat: shape: (13).

        return: state_flu: shape : (13).
        """
        trans, quat, linear_velocity, angular_velocity = (state_habitat[:3], state_habitat[3:7],
                                                          state_habitat[7:10], state_habitat[10:])
        tf_global_agent = Transform.from_quat_trans(quat, trans).as_matrix()
        tf_global_flu__agent_flu = cls.tf_flu__habitat @ tf_global_agent @ cls.tf_habitat__flu
        quat_flu_flu, trans_flu_flu = Transform.from_matrix(tf_global_flu__agent_flu).as_quat_trans()

        linear_velocity__flu = cls.rot_flu__habitat @ linear_velocity  # linear velocity in global_flu

        rot_agent_flu__global = Transform.from_matrix(tf_global_agent @ cls.tf_habitat__flu).as_rot_mat().T
        angular_velocity__flu = rot_agent_flu__global @ angular_velocity  # angular velocity in agent_flu

        state_flu = np.concatenate([trans_flu_flu, quat_flu_flu, linear_velocity__flu, angular_velocity__flu], axis=-1)
        return state_flu

    @classmethod
    def state_flu_2_state_habitat(cls, state_flu: np.ndarray):
        """
        Change state with global flu reference frame to global habitat.

        param state_flu: shape: (13).

        return: state_habitat: shape : (13).
        """
        trans_flu, quat_flu, linear_velocity_flu, angular_velocity_flu = (state_flu[:3], state_flu[3:7],
                                                                          state_flu[7:10], state_flu[10:])
        tf_global_flu__agent_flu = Transform.from_quat_trans(quat_flu, trans_flu).as_matrix()
        tf_global_agent = cls.tf_habitat__flu @ tf_global_flu__agent_flu @ cls.tf_flu__habitat
        quat_habitat, trans_habitat = Transform.from_matrix(tf_global_agent).as_quat_trans()

        linear_velocity_habitat = cls.rot_habitat__flu @ linear_velocity_flu  # linear velocity in global_flu

        rot_global__agent_flu = Transform.from_matrix(tf_global_agent @ cls.tf_habitat__flu).as_rot_mat()
        angular_velocity_habitat = rot_global__agent_flu @ angular_velocity_flu  # angular velocity in agent_flu

        state_habitat = np.concatenate([trans_habitat, quat_habitat, linear_velocity_habitat,
                                        angular_velocity_habitat], axis=-1)
        return state_habitat


def _test():
    for i in range(1000):
        state_flu = np.random.randn(13)
        state_flu[3:7] = state_flu[3:7] / np.linalg.norm(state_flu[3:7])

        state_habitat = Habitat2FLU.state_flu_2_state_habitat(state_flu)
        state_flu_2 = Habitat2FLU.state_habitat_2_state_flu(state_habitat)
        state_flu_2_minus = np.copy(state_flu_2)
        state_flu_2_minus[3:7] = -state_flu_2[3:7]
        assert np.all(np.isclose(state_flu, state_flu_2)) or np.all(np.isclose(state_flu, state_flu_2_minus))


def main():
    _test()


if __name__ == '__main__':
    main()
