#! /usr/bin/env python3

from _camtrack import _remove_correspondences_with_ids

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np
import sortednp as snp
import frameseq
from _camtrack import *
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import cv2


def _initialize_with_two_frames(corners_1, corners_2, intrinsic_mat, triangulation_parameters):
    correspondences = build_correspondences(corners_1, corners_2)
    if len(correspondences.ids) <= 5:
        return None, None, None
    E, mask_E = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2, intrinsic_mat,
                                     method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return None, None, None
    fundamental_inliers = np.sum(mask_E)
    H, mask_H = cv2.findHomography(correspondences.points_1, correspondences.points_2, method=cv2.RANSAC,
                                   ransacReprojThreshold=1.0, confidence=0.999)
    homography_inliers = np.sum(mask_H)
    if fundamental_inliers / homography_inliers < 1.2:
        return None, None, None
    correspondences = _remove_correspondences_with_ids(correspondences, np.where(mask_E == 0)[0])
    best_points, best_ids, best_pose = None, None, None
    R1, R2, t_d = cv2.decomposeEssentialMat(E)
    for pose in [Pose(R1.T, R1.T @ t_d), Pose(R2.T, R2.T @ t_d), Pose(R1.T, R1.T @ -t_d), Pose(R2.T, R2.T @ -t_d)]:
        points, ids = triangulate_correspondences(
            correspondences,
            eye3x4(),
            pose_to_view_mat3x4(pose),
            intrinsic_mat,
            triangulation_parameters,
        )
        if best_ids is None or len(ids) > len(best_ids):
            best_points, best_ids, best_pose = points, ids, pose
    return best_points, best_ids, best_pose


def _initialize_with_storage(corner_storage, intrinsic_mat, triangulation_parameters):
    best_index, best_points, best_ids, best_pose = None, None, None, None
    for i in range(1, len(corner_storage)):
        points, ids, pose = _initialize_with_two_frames(corner_storage[0], corner_storage[i], intrinsic_mat,
                                                        triangulation_parameters)
        if ids is None:
            continue
        if best_ids is None or len(ids) > len(best_ids):
            best_index, best_points, best_ids, best_pose = i, points, ids, pose
            print("init 0 and", i, "size =", len(ids), "best_size =", len(best_ids), flush=True)
    if best_ids is None:
        return None, None, None, None
    point_cloud_builder = PointCloudBuilder()
    point_cloud_builder.add_points(best_ids, best_points)
    return best_index, best_pose, point_cloud_builder


def _track_camera_with_parameters(corner_storage, intrinsic_mat, triangulation_parameters):
    view_mats = [eye3x4()]
    init_index, init_pose, point_cloud_builder = _initialize_with_storage(corner_storage, intrinsic_mat,
                                                                          triangulation_parameters)
    for i in range(1, len(corner_storage)):
        print(i, end=' ', flush=True)
        if i == init_index:
            view_mats.append(pose_to_view_mat3x4(init_pose))
            continue
        _, (object_ids, image_ids) = snp.intersect(point_cloud_builder.ids.flatten(), corner_storage[i].ids.flatten(),
                                                   indices=True)
        if len(object_ids) < 4:
            return None, None
        object_points = point_cloud_builder.points[object_ids]
        image_points = corner_storage[i].points[image_ids]
        solve_result, R, t, inliers = cv2.solvePnPRansac(object_points.reshape((len(object_points), 1, 3)),
                                                         image_points.reshape((len(image_points), 1, 2)),
                                                         cameraMatrix=intrinsic_mat, distCoeffs=None)
        if not solve_result:
            return None, None

        view_mats.append(rodrigues_and_translation_to_view_mat3x4(R, t))

        res_ids = np.delete(point_cloud_builder.ids[object_ids], inliers, axis=0)
        new_points = 0
        for j in range(i):
            correspondences = build_correspondences(corner_storage[j], corner_storage[i], ids_to_remove=res_ids)
            if len(correspondences.ids) == 0:
                continue
            points, ids = triangulate_correspondences(
                correspondences,
                view_mats[j],
                view_mats[i],
                intrinsic_mat,
                triangulation_parameters,
            )
            point_cloud_builder.add_points(ids, points)
            new_points += len(points)
        print(new_points, "new points,", len(point_cloud_builder.points), "points in point cloud builder")
    return view_mats, point_cloud_builder


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    for min_triangulation_angle_deg in [5.0, 3.0, 1.0, 0.25]:
        triangulation_parameters = TriangulationParameters(max_reprojection_error=1.0,
                                                           min_triangulation_angle_deg=min_triangulation_angle_deg,
                                                           min_depth=0.1)
        view_mats, point_cloud_builder = _track_camera_with_parameters(corner_storage, intrinsic_mat,
                                                                       triangulation_parameters)
        if view_mats is not None:
            return view_mats, point_cloud_builder

    return [], PointCloudBuilder()


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
