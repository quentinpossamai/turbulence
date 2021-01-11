from typing import Union, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import VideoWriter, VideoWriter_fourcc
from util import Progress


class ErrorEstimation(object):
    def __init__(self, input_data: pd.DataFrame, calibration_matrix: np.array, distortion_coefficient: np.array,
                 max_distance: float):
        """
        Create a video to verify with the eye if the position estimation and video flux.
        https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

        :param input_data: pandas.DataFrame with columns:
            'pose_time' : The time of the measured pose.
            'pose' : Each pose correspond to camera_tf_origin, a util.Transformation object. The z-axis origin MUST be
             aligned vertical (aligned to -gravity). Unit must be meters.
            'image_time' : The time of the measured image.
            'image' : numpy.ndarray of shape (height, width, 3)

        :param calibration_matrix: The intrinsic parameters of the camera. The camera matrix = [f_x,   0, c_x,
                                                                                                  0, f_y, c_y,
                                                                                                  0,   0,   1]
        :param distortion_coefficient: The distortion coefficient relative to the distortion model.
        cf distortion_model explication. Expressed in this order k1, k2, p1, p2, k3, k4, k5, k6.
        Possible computation can also be done with only k1, k2, p1, p2 or k1, k2, p1, p2, k3, k4.

        :param max_distance: The maximum distance to consider between the camera position and the projected red point.
        Must be positive.

        cf.
        https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=
        solvepnpran

        https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
        """
        assert input_data['image'][0].shape[2] == 3, 'Image must have 3rgb channels.'

        self.df = input_data
        self.k = calibration_matrix
        self.d = distortion_coefficient
        self.p3 = []

        assert max_distance > 0, 'A distance must be positive.'
        self.max_distance = max_distance

        self.frames_width = self.df['image'][0].shape[1]  # width along the second axis
        self.frames_height = self.df['image'][0].shape[0]

        # Distortion
        k1, k2, p1, p2, k3, k4, k5, k6 = 0, 0, 0, 0, 0, 0, 0, 0
        if len(self.d) == 4:
            k1, k2, p1, p2 = self.d
        elif len(self.d) == 5:
            k1, k2, p1, p2, k3 = self.d
        elif len(self.d) == 8:
            k1, k2, p1, p2, k3, k4, k5, k6 = self.d

        def tmp(point: Union[np.ndarray, list, tuple], distort: bool):
            """
            :param point: The 2D point to be affected.

            :param distort: If true will apply distortion, if false will remove the distortion
            (cf return explication).

            :return: The 2d point with the distortion applied if distort = True : output = (x_2, y_2).
            If distort = False, project the point into the camera reference frame and remove the distortion :
            output = (x_prime, y_prime).

            cf. https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?#undistortpoints

            K = [f_x,   0, c_x,
                   0, f_y, c_y,
                   0,   0,   1]
            is the camera matrix, the intrinsics parameters. Used to project points on /  reconstruct
            points from a distorted image.

            To project project points on / reconstruct points from an undistorted image you must use:
            K' = cv2.getOptimalNewCameraMatrix().
            """
            return self._radial_tangential_distortion(point, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3, k4=k4, k5=k5, k6=k6,
                                                      distort=distort)

        self.distort_func = tmp

    def image23d(self, idx: int, image_point: np.array) -> Tuple[np.array, Union[None, str]]:
        """
        From pixels coordinates, project a point in 3d on the floor relative to the camera reference frame.
        The floor is defined where the origin reference frame is defined.

        :param idx: The time that indicate the position of the drone.

        :param image_point: The point in the image (in pixels).

        :return: A 3D point in the camera reference frame and a potential error message.
        """

        # self.df[’pose'] is camera_tf_origin
        u = image_point[0]
        v = image_point[1]
        x_prime, y_prime = self.distort_func((u, v), distort=False)
        vect_camera = np.array([x_prime, y_prime, 1])

        origin_tf_camera = self.df['pose'][idx].inv()
        origin_rot_camera = origin_tf_camera.get_rot()
        vect_origin = origin_rot_camera @ vect_camera
        vect_origin /= np.linalg.norm(vect_origin)

        camera_position = origin_tf_camera.get_trans()
        h = camera_position[2]  # drone'sin height
        floor_normal = np.array([0, 0, 1])
        # If height is positive and camera sees the floor
        if (h > 0) and (floor_normal @ vect_origin < 0):
            t = - camera_position[2] / vect_origin[2]
            point_floor_origin = t * vect_origin + camera_position
            if np.linalg.norm(point_floor_origin - camera_position) > self.max_distance:
                return origin_tf_camera.inv() @ point_floor_origin, 'Point on the floor too far.'
            else:
                return origin_tf_camera.inv() @ point_floor_origin, None
        else:
            if h <= 0:
                return None, 'Camera has negative height.'
            elif floor_normal @ vect_origin >= 0:
                return None, 'Pixel chosen cannot be on the floor.'

    def _is_out_of_image(self, image_point: Union[list, tuple, np.ndarray]):
        if image_point is not None:
            assert len(image_point) == 2
            return not ((0 <= image_point[0] < self.frames_width) and (0 <= image_point[1] < self.frames_height))
        else:
            return True

    def _space2image(self, point3d: Union[list, tuple, np.ndarray]):
        """
        Project a 3D point expressed in the camera reference frame into the image of the camera.

        :param point3d: A 3D point.

        :return: A 2D point.

        Using this : https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?
        """
        assert len(point3d) == 3
        x, y, z = point3d
        if z > 0:
            x_prime = x / z
            y_prime = y / z

            x_2, y_2 = self.distort_func(point=(x_prime, y_prime), distort=True)

            u, v = (self.k @ np.array([x_2, y_2, 1]))[:-1]
            return u, v
        else:
            return None

    def _radial_tangential_distortion(self, point2d: Union[list, tuple, np.ndarray],
                                      k1: float, k2: float, p1: float, p2: float, k3: float,
                                      k4: float, k5: float, k6: float,
                                      distort: bool) -> np.ndarray:
        """
        k1, k2, p1, p2, k3, k4, k5, k6 are distortion parameters.

        :param distort: If true will apply distortion, if false will remove the distortion (cf return explication).

        :return: The 2d point with the distortion applied if distort = True : output = (x_2, y_2).
        If distort = False, project the point into the camera reference frame and remove the distortion :
        output = (x_prime, y_prime).

        cf. https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?#undistortpoints

        K = [f_x,   0, c_x, is the camera matrix, the intrinsics parameters. Used to project points on / reconstruct
               0, f_y, c_y,
               0,   0,   1]
        points from a distorted image.

        To project project points on / reconstruct points from an undistorted image you must use:
        K' = cv2.getOptimalNewCameraMatrix().
        """
        assert len(point2d) == 2, f'2D point must have len of 2 not len of {len(point2d)}'
        if not isinstance(point2d, np.ndarray):
            point2d = np.array(point2d)

        if distort:
            x_prime, y_prime = point2d
            r_2 = x_prime ** 2 + y_prime ** 2

            term1 = ((1 + k1 * r_2 + k2 * r_2 ** 2 + k3 * r_2 ** 3) /
                     (1 + k4 * r_2 + k5 * r_2 ** 2 + k6 * r_2 ** 3))
            # x_2 Computation
            x_term2 = 2 * p1 * x_prime * y_prime
            x_term3 = p2 * (r_2 + 2 * x_prime ** 2)
            x_2 = (x_prime * term1 + x_term2 + x_term3)

            # y_2 Computation
            y_term2 = 2 * p2 * x_prime * y_prime
            y_term3 = p1 * (r_2 + 2 * y_prime ** 2)
            y_2 = (y_prime * term1 + y_term2 + y_term3)

            return np.array([x_2, y_2])

        elif not distort:
            x_prime, y_prime = cv2.undistortPoints(src=point2d, cameraMatrix=self.k,
                                                   distCoeffs=np.array([k1, k2, p1, p2, k3, k4, k5, k6])).flatten()
            return np.array([x_prime, y_prime])

    def p3_generator(self):
        """
        p3 are points expressed in an image with a coordinate system expressed by ROS.
        cf. http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Image.html
        and http://wiki.ros.org/image_pipeline/CameraInfo

        :return: p3 points list.
        """
        cx, cy = self.k[0, 2], self.k[1, 2]
        p = np.array([cx, cy + cy / 2])
        p1, error = self.image23d(0, p)  # Ref camera
        is_reconstructed = error is None
        camera_1_tf_origin = self.df['pose'][0]  # Describing drone in reference frame origin
        progress = Progress(len(self.df), 'Computing p3 points')
        for i in range(len(self.df)):
            if is_reconstructed:
                camera_i_tf_origin = self.df['pose'][i]
                camera_i_tf_camera_1 = camera_i_tf_origin @ camera_1_tf_origin.inv()
                p2 = camera_i_tf_camera_1 @ p1
                p3 = np.int32(np.round(self._space2image(p2)))
                if self._is_out_of_image(p3):
                    p1, error = self.image23d(i, p)  # Ref camera
                    is_reconstructed = error is None
                    camera_1_tf_origin = self.df['pose'][i]
                    if is_reconstructed:
                        p3 = 'P3 out of image.'
                    else:
                        p3 = error + '\nP3 out of image.'
            else:
                p1, error = self.image23d(i, p)  # Ref camera
                is_reconstructed = error is None
                camera_1_tf_origin = self.df['pose'][i]
                if is_reconstructed:
                    p3 = p
                else:
                    p3 = error  # Send the message of failure
            self.p3.append(p3)
            progress.update()

    def p32video(self, fps: int, saving_path: str):
        codec = VideoWriter_fourcc(*'H264')
        # name = ABSOLUTE_PATH + 'error_estimation_videos/' + part1 + '_' + part2 + '_error_estimation.mkv'
        video = VideoWriter(saving_path, codec, float(fps), (self.frames_width, self.frames_height), True)
        print(f'Video will be saved at : {saving_path}')

        prg = Progress(len(self.df), 'Creating video')
        shape = (self.frames_height, self.frames_width, 3)
        for i, image in enumerate(self.df['image']):
            fig = plt.figure(figsize=(self.frames_width / 100., self.frames_height / 100.))
            ax = plt.axes([0, 0, 1, 1])
            ax.imshow(image)
            if isinstance(self.p3[i], np.ndarray):
                p = self.p3[i]
                ax.scatter(p[0], p[1], color='r')
            else:
                ax.annotate(self.p3[i], (self.frames_width / 2, self.frames_height / 2), xycoords='data', size=12,
                            ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.5, fc='wn'))
            xyz = self.df['pose'][i].inv().get_trans()
            angles = self.df['pose'][i].inv().get_rot_euler(seq='xyz', degrees=True)
            legend = (f'Camera position\n'
                      f'       x:{xyz[0]:8.3f}m\n'
                      f'       y:{xyz[1]:8.3f}m\n'
                      f'       z:{xyz[2]:8.3f}m\n'
                      f'   Rot x:{angles[0]:8.3f}°\n'
                      f'   Rot y:{angles[1]:8.3f}°\n'
                      f'   Rot z:{angles[2]:8.3f}°\n'
                      f'frame id:{i:9}')
            ax.annotate(legend, (self.frames_width - 40, 40), xycoords='data', size=12, ha='right', va='top',
                        bbox=dict(boxstyle='round', alpha=0.5, fc='wn'), family='monospace')
            ax.axis('off')
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
            plt.close('all')

            video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
            prg.update()

            # plt.imshow(data)
            # plt.show()

        video.release()
        cv2.destroyAllWindows()
