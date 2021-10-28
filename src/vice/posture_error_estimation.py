import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import imageio
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
from tqdm import tqdm
from typing import Union, Tuple

from src.utils.transform import Transform

matplotlib.use('TKAgg')


class ErrorEstimation(object):
    def __init__(self, input_data: dict,
                 calibration_matrix: np.array,
                 distortion_coefficient: np.array,
                 width: int,
                 height: int,
                 use_depth: bool,
                 max_distance: float = None,
                 depth_coef: float = None):
        """
        Create a video to verify with the eye if the position estimation and video flux.
        https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

        input_data: dict with the following keys:
            'time' : The time of the measured pose and the image.
            'pose' : Each pose correspond to tf_camera_origin, a (4, 4) np.array. The z-axis origin MUST be aligned
            vertical (aligned to -gravity). Unit must be meters.
            'img' : path to the image.
            'depth': path to the depth map.


        calibration_matrix: The intrinsic parameters of the camera. The camera matrix = [f_x, 0, c_x,
                                                                                         0, f_y, c_y,
                                                                                         0,   0,   1]

        distortion_coefficient: The distortion coefficient relative to the distortion model.
        cf distortion_model explication. Expressed in this order k1, k2, p1, p2, k3, k4, k5, k6.
        Possible computation can also be done with only k1, k2, p1, p2 or k1, k2, p1, p2, k3, k4.

        param width: The width of the image.

        param height: The height of the image.

        param depth_coef: To obtain the depth map the first channel of the depth image will be / 255 * depth_coef.

        cf.
        https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=
        solvepnpran

        https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
        """
        self.data = input_data
        self.k = calibration_matrix
        self.d = distortion_coefficient
        self.frames_width = width
        self.frames_height = height
        self.use_depth = use_depth
        self.max_distance = max_distance
        self.depth_coef = depth_coef

        self.p3 = []

        # Distortion
        k1, k2, p1, p2, k3, k4, k5, k6 = 0, 0, 0, 0, 0, 0, 0, 0
        if len(self.d) == 4:
            k1, k2, p1, p2 = self.d
        elif len(self.d) == 5:
            k1, k2, p1, p2, k3 = self.d
        elif len(self.d) == 8:
            k1, k2, p1, p2, k3, k4, k5, k6 = self.d

        def distortion_function(point: Union[np.ndarray, list, tuple], distort: bool):
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

        self.distort_func = distortion_function

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

    def _click_place_point(self, idx):
        # img = np.array(imageio.imread(self.data['img'][idx]))
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(img)
        # pixel_clicked = [np.zeros(1)]
        # print(f'Frame: {idx}')
        #
        # def onclick(event):
        #     pixel_clicked[0] = np.array([np.round(event.xdata), np.round(event.ydata)])
        #     plt.close()
        #
        # fig.canvas.mpl_connect('button_press_event', onclick)
        # plt.show()
        # print(f'Input pixel: {pixel_clicked[0]}\n')
        # return pixel_clicked[0]
        # 2. Loading annotations
        import src.data.euroc_mav.loading as euroc
        import json
        flight = euroc.FLIGHT_LIST[0]
        vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
        vice_point = pickle.load(open(f'{vice_path}p3_points.pkl', 'rb'))
        vice_point = np.stack(vice_point, axis=0).astype(int)

        vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
        with open(f'{vice_path}task_v1_01_easy_backup/annotations.json') as json_file:
            data = json.load(json_file)
        key_points = np.zeros((len(data[0]['tracks'][0]['shapes']), 3))  # frame_id, point_x, point_y
        for i, e in enumerate(data[0]['tracks'][0]['shapes']):
            key_points[i] = np.array([e['frame']] + e['points'])
        point_x = np.round(np.interp(range(len(vice_point)), key_points[:, 0], key_points[:, 1])).astype(int)
        point_y = np.round(np.interp(range(len(vice_point)), key_points[:, 0], key_points[:, 2])).astype(int)
        annotation_point = np.stack([point_x, point_y], axis=1).astype(float)
        return annotation_point[idx]

    def _is_out_of_image(self, image_point: Union[list, tuple, np.ndarray]):
        if image_point is not None:
            assert len(image_point) == 2
            criterion = 0.1
            cond1 = 0 / self.frames_width + criterion < image_point[0] / self.frames_width
            cond2 = image_point[0] / self.frames_width < self.frames_width / self.frames_width - criterion
            cond3 = 0 / self.frames_height + criterion < image_point[1] / self.frames_height
            cond4 = image_point[1] / self.frames_height < self.frames_height / self.frames_height - criterion
            return not (cond1 and cond2 and cond3 and cond4)
        else:
            return True

    def _image23d(self, idx: int, image_point: np.array) -> Tuple[np.array, Union[None, str]]:
        """
        From pixels coordinates, project a point in 3d on the floor relative to the camera reference frame.
        The floor is defined where the origin reference frame is defined.

        :param idx: The time that indicate the position of the drone.

        :param image_point: The point in the image (in pixels).

        :return: A 3D point in the camera reference frame and a potential error message.
        """
        # 1. Finding view vector
        # self.df[’pose'] is tf_camera_origin
        u = image_point[0]  # pixel point in image reference frame
        v = image_point[1]
        x_prime, y_prime = self.distort_func((u, v), distort=False)  # remove distortion, camera reference frame

        # 2. Depth estimation
        if self.use_depth:
            # Apply radial camera distortion used in depth_from_pointcloud.py and intrinsic matrix
            # to get image coordinates
            rr = x_prime ** 2 + y_prime ** 2
            k1, k2 = self.d[:2]  # Extracting distortion coefficients to find depth
            distortion = (1 + k1 * rr + k2 * rr * rr)
            u_depth = x_prime * distortion * self.k[0, 0] + self.k[0, 2]
            v_depth = y_prime * distortion * self.k[1, 1] + self.k[1, 2]
            u_depth = np.floor(0.5 * u_depth).astype(int)
            v_depth = np.floor(0.5 * v_depth).astype(int)

            depth_map = np.array(imageio.imread(self.data['depth'][idx]))[:, :, :3] / 255 * self.depth_coef
            z = depth_map[v_depth, u_depth][0]
            p1_camera_with_depth = np.array([x_prime * z, y_prime * z, z])
            return p1_camera_with_depth, None
        else:
            vect_camera = np.array([x_prime, y_prime, 1])

            tf_origin_camera = Transform.from_matrix(self.data['pose'][idx]).inv().as_matrix()
            rot_origin_camera = Transform.from_matrix(tf_origin_camera).as_rot_mat()
            vect_origin = rot_origin_camera @ vect_camera
            vect_origin /= np.linalg.norm(vect_origin)

            camera_position = Transform.from_matrix(tf_origin_camera).as_trans()
            h = camera_position[2]  # drone's height
            floor_normal = np.array([0, 0, 1])
            # If height is positive and camera sees the floor
            if (h > 0) and (floor_normal @ vect_origin < 0):
                t = - camera_position[2] / vect_origin[2]
                point_floor_origin = t * vect_origin + camera_position
                p1 = (Transform.from_matrix(tf_origin_camera).inv().as_matrix() @ np.append(point_floor_origin, 1))[:3]
                if np.linalg.norm(point_floor_origin - camera_position) > self.max_distance:
                    return p1, 'Point on the floor too far.'
                else:
                    return p1, None
            else:
                if h <= 0:
                    return None, 'Camera has negative height.'
                elif floor_normal @ vect_origin >= 0:
                    return None, 'Pixel chosen cannot be on the floor.'

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

            x_2, y_2 = self.distort_func(point=(x_prime, y_prime), distort=True)  # Apply distortion

            u, v = (self.k @ np.array([x_2, y_2, 1]))[:-1]
            return u, v
        else:
            return None

    def p3_generator(self, saving_path: str):
        """
        p3 are points expressed in an image with a coordinate system expressed by ROS.
        cf. http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Image.html
        and http://wiki.ros.org/image_pipeline/CameraInfo

        :return: p3 points list.
        """
        # cx, cy = self.k[0, 2], self.k[1, 2]
        # p = np.array([cx, cy + cy / 2])
        p = self._click_place_point(0)
        p1, error = self._image23d(0, p)  # Ref camera
        is_reconstructed = error is None
        tf_camera1_origin = self.data['pose'][0]  # Describing drone in reference frame origin
        for i in tqdm(range(len(self.data["pose"])), desc='Computing p3 points'):
            if is_reconstructed:
                tf_camera_i_origin = self.data['pose'][i]
                tf_camera_i_camera_1 = tf_camera_i_origin @ Transform.from_matrix(tf_camera1_origin).inv().as_matrix()
                p2 = (tf_camera_i_camera_1 @ np.append(p1, 1))[:3]
                p3 = np.int32(np.round(self._space2image(p2)))
                if self._is_out_of_image(p3):
                    p = self._click_place_point(i)
                    p1, error = self._image23d(i, p)  # Ref camera
                    is_reconstructed = error is None
                    tf_camera1_origin = self.data['pose'][i]
                    if is_reconstructed:
                        p3 = p
                    else:
                        p3 = error + '\nP3 out of image.'
            else:
                p = self._click_place_point(i)
                p1, error = self._image23d(i, p)  # Ref camera
                is_reconstructed = error is None
                tf_camera1_origin = self.data['pose'][i]
                if is_reconstructed:
                    p3 = p
                else:
                    p3 = error  # Send the message of failure
            self.p3.append(p3)
        pickle.dump(self.p3, open(saving_path, 'wb'))

    def build_video(self, fps, saving_path):
        codec = VideoWriter_fourcc(*'H264')
        video = VideoWriter(saving_path, codec, float(fps), (self.frames_width, self.frames_height), True)
        print(f'Video will be saved at : {saving_path}')
        shape = (self.frames_height, self.frames_width, 3)
        for i, image_path in enumerate(tqdm(self.data['img'], desc="Creating a video")):
            img = np.array(imageio.imread(image_path))
            color_frame = np.uint8(np.zeros((shape[0], shape[1], 3)))
            for j in range(3):
                color_frame[:, :, j] = img
            img = color_frame

            fig = plt.figure(figsize=(self.frames_width / 100., self.frames_height / 100.))
            ax = plt.axes([0, 0, 1, 1])
            ax.imshow(img)
            shape = img.shape
            # Contribution part, comment to get a raw video.
            if isinstance(self.p3[i], np.ndarray):
                p = self.p3[i]
                ax.scatter(p[0], p[1], color='r')
            else:
                ax.annotate(self.p3[i], (self.frames_width / 2, self.frames_height / 2), xycoords='data', size=12,
                            ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
            tf_origin_camera = Transform.from_matrix(self.data['pose'][i]).inv()
            trans = tf_origin_camera.as_trans()
            angles = tf_origin_camera.as_rot_euler(seq='zyx', degrees=True)[::-1]
            legend = (f'Camera position\n'
                      f'       x:{trans[0]:8.3f}m\n'
                      f'       y:{trans[1]:8.3f}m\n'
                      f'       z:{trans[2]:8.3f}m\n'
                      f'   Rot x:{angles[0]:8.3f}°\n'
                      f'   Rot y:{angles[1]:8.3f}°\n'
                      f'   Rot z:{angles[2]:8.3f}°\n'
                      f'frame id:{i:9}')
            ax.annotate(legend, (self.frames_width - 40, 40), xycoords='data', size=12, ha='right', va='top',
                        bbox=dict(boxstyle='round', alpha=0.5, fc='w'), family='monospace')
            # Contribution part end

            ax.axis('off')
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
            plt.close('all')
            video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
        video.release()
        cv2.destroyAllWindows()
