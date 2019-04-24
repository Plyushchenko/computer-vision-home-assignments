from collections import namedtuple
from typing import List

import numpy as np
import scipy
import sortednp as snp
from scipy.optimize import approx_fprime
from scipy.sparse import csr_matrix

from corners import FrameCorners
from _camtrack import PointCloudBuilder, eye3x4, view_mat3x4_to_rodrigues_and_translation, calc_inlier_indices, \
    rodrigues_and_translation_to_view_mat3x4

CornerInlier = namedtuple('CornerInlier', ('frame_index', 'object_index', 'image_point'))

def _reprojection_error(p, point_2d, intrinsic_mat):
    r_vec = p[0 : 3].reshape(3, 1)
    t_vec = p[3 : 6].reshape(3, 1)
    view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
    proj_mat = np.dot(intrinsic_mat, view_mat)
    point_3d = p[6 : 9]
    point3d_hom = np.hstack((point_3d, 1))
    proj_point_2d = np.dot(proj_mat, point3d_hom)
    proj_point_2d = proj_point_2d / proj_point_2d[2]
    proj_point_2d = proj_point_2d.T[:2]
    err = (point_2d - proj_point_2d).reshape(-1)
    return np.linalg.norm(err)


def _reprojection_errors(p, N, inliers, intrinsic_mat):
    errors = np.zeros(len(inliers))
    for i, (frame_index, index_3d, point_2d) in enumerate(inliers):
        p1 = np.hstack((p[6 * frame_index: 6 * (frame_index + 1)], p[6 * N + 3 * index_3d: 6 * N + 3 * (index_3d + 1)]))
        errors[i] = _reprojection_error(p1, point_2d, intrinsic_mat)
    return np.array(errors)


def _compute_jacobian(p, N, M, intrinsic_mat, inliers):
    data = []
    row_indices = []
    column_indices = []
    for i, (frame_index, index_3d, point_2d) in enumerate(inliers):
        p1 = np.hstack((p[6 * frame_index : 6 * (frame_index + 1)], p[6 * N + 3 * index_3d : 6 * N + 3 * (index_3d + 1)]))
        p1_prime = scipy.optimize.approx_fprime(
            p1, lambda p1: _reprojection_error(p1, point_2d, intrinsic_mat), np.full(p1.size, 1e-9))
        for j in range(9):
            data.append(p1_prime[j])
            row_indices.append(i)
        for j in range(6):
            column_indices.append(6 * frame_index + j)
        for j in range(3):
            column_indices.append(6 * N + 3 * index_3d + j)
    return scipy.sparse.csr_matrix((data, (row_indices, column_indices)), shape=(len(inliers), 6 * N + 3 * M))


def _optimize_parameters(p, N, M, intrinsic_mat, inliers):
    print("Optimization", flush=True)
    J = _compute_jacobian(p, N, M, intrinsic_mat, inliers)
    lambda_coefficient = 1.
    for i in range(10):
        errors = _reprojection_errors(p, N, inliers, intrinsic_mat)
        start_error = errors @ errors
        print(i, "lambda_coefficient =", lambda_coefficient, "start_error =", start_error, flush=True)
        JJ = J.T.dot(J).toarray()
        JJ += lambda_coefficient * np.diag(np.diagonal(JJ))
        U = JJ[:6 * N, :6 * N]
        W = JJ[:6 * N, 6 * N:]
        V = JJ[6 * N:, 6 * N:]
        V = np.linalg.inv(V)

        g = J.toarray().T.dot(_reprojection_errors(p, N, inliers, intrinsic_mat))
        A = U - W.dot(V).dot(W.T)
        B = W.dot(V).dot(g[6 * N:]) - g[:6 * N]

        try:
            delta_c = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            lambda_coefficient *= 5.
            continue

        delta_x = V.dot(-g[6 * N:] - W.T.dot(delta_c))
        tmp = p.copy() + np.hstack((delta_c, delta_x))
        errors = _reprojection_errors(tmp, N, inliers, intrinsic_mat)
        error = errors @ errors
        print("error at the end of the step =", error, flush=True)
        if error < start_error:
            p[:] = tmp
            J = _compute_jacobian(p, N, M, intrinsic_mat, inliers)
            lambda_coefficient /= 5.
        else:
            lambda_coefficient *= 5.
    return p

def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:
    print("Started ba, max_inlier_reprojection_error =", max_inlier_reprojection_error, flush=True)
    used_indices_3d = set()
    inliers = []
    for i, (corners, view_mat) in enumerate(zip(list_of_corners, view_mats)):
        _, (indices_3d, indices_2d) = snp.intersect(pc_builder.ids.flatten(), corners.ids.flatten(), indices=True)
        points_3d = pc_builder.points[indices_3d]
        points_2d = corners.points[indices_2d]
        inlier_indices = calc_inlier_indices(points_3d, points_2d, intrinsic_mat @ view_mat, max_inlier_reprojection_error)
        for inlier_index in inlier_indices:
            index_3d = pc_builder.ids[indices_3d[inlier_index], 0]
            inliers.append(CornerInlier(i, index_3d, points_2d[inlier_index]))
            used_indices_3d.add(index_3d)

    if not used_indices_3d:
        print("No inliers found", flush=True)
        return view_mats

    used_indices_3d = list(sorted(used_indices_3d))
    for i in range(len(inliers)):
        inliers[i] = CornerInlier(
            inliers[i].frame_index,
            used_indices_3d.index(inliers[i].object_index),
            inliers[i].image_point,
        )

    N = len(view_mats)
    M = len(used_indices_3d)
    p = np.zeros(6 * N + 3 * M)
    for i in range(N):
        r_vec, t_vec = view_mat3x4_to_rodrigues_and_translation(view_mats[i])
        p[6 * i : 6 * i + 3] = r_vec.reshape(-1)
        p[6 * i + 3 : 6 * i + 6] = t_vec.reshape(-1)
    _, (indices, _) = snp.intersect(pc_builder.ids.flatten(), np.array(used_indices_3d), indices=True)
    p[6 * N:] = pc_builder.points[indices].reshape(-1)

    p = _optimize_parameters(p, N, M, intrinsic_mat, inliers)

    for i in range(N):
        view_mats[i] = rodrigues_and_translation_to_view_mat3x4(
            p[6 * i : 6 * i + 3].reshape(3, 1),
            p[6 * i + 3 : 6 * i + 6].reshape(3, 1),
        )
    pc_builder.update_points(np.array(used_indices_3d), p[6 * len(view_mats):].reshape(-1, 3))
    return view_mats
