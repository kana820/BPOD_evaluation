import os
import numpy as np
import argparse
from pathlib import Path
from os import path
from ...file_system import listdir
from ...colors import cyan
from ...dataset import Dataset
from statistics import mean, median
import matplotlib.pyplot as plt
from statistics import mean
import math
from scipy.optimize import least_squares
import pickle as pkl
from math import cos, sin

from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})


def hat(v):
    """
    vecotrized version of the hat function, creating for a vector its skew symmetric matrix.

    Args:
        v (np.array<float>(..., 3, 1)): The input vector.

    Returns:
        (np.array<float>(..., 3, 3)): The output skew symmetric matrix.

    """
    E1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    E2 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    E3 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    return v[..., 0:1, :] * E1 + v[..., 1:2, :] * E2 + v[..., 2:3, :] * E3


def exp(v, der=False):
    """
    Vectorized version of the exponential map.

    Args:
        v (np.array<float>(..., 3, 1)): The input axis-angle vector.
        der (bool, optional): Wether to output the derivative as well. Defaults to False.

    Returns:
        R (np.array<float>(..., 3, 3)): The corresponding rotation matrix.
        [dR (np.array<float>(3, ..., 3, 3)): The derivative of each rotation matrix.
                                            The matrix dR[i, ..., :, :] corresponds to
                                            the derivative d R[..., :, :] / d v[..., i, :],
                                            so the derivative of the rotation R gained
                                            through the axis-angle vector v with respect
                                            to v_i. Note that this is not a Jacobian of
                                            any form but a vectorized version of derivatives.]

    """
    n = np.linalg.norm(v, axis=-2, keepdims=True)
    H = hat(v)

    with np.errstate(all="ignore"):
        R = np.identity(3) + (np.sin(n) / n) * H + ((1 - np.cos(n)) / n ** 2) * (H @ H)
    R = np.where(n == 0, np.identity(3), R)

    if der:
        sh = (3,) + tuple(1 for _ in range(v.ndim - 2)) + (3, 1)
        dR = np.swapaxes(np.expand_dims(v, axis=0), 0, -2) * H
        dR = dR + hat(
            np.cross(v, ((np.identity(3) - R) @ np.identity(3).reshape(sh)), axis=-2)
        )
        dR = dR @ R

        n = n ** 2  # redifinition
        with np.errstate(all="ignore"):
            dR = dR / n
        dR = np.where(n == 0, hat(np.identity(3).reshape(sh)), dR)

        return R, dR

    else:
        return R


def axis_angle(vec, theta):
    vec_normal = vec / np.linalg.norm(vec)

    R = np.zeros((3, 3))
    R = (
        cos(theta) * np.eye(3, 3)
        + sin(theta) * matrixX(vec_normal)
        + (1 - cos(theta)) * np.outer(vec_normal, vec_normal)
    )
    return R


def matrixX(vec):
    vecX = np.zeros((3, 3))
    vecX[0, 1] = -vec[2]
    vecX[0, 2] = vec[1]
    vecX[1, 0] = vec[2]
    vecX[1, 2] = -vec[0]
    vecX[2, 0] = -vec[1]
    vecX[2, 1] = vec[0]
    return vecX


def rotate_v1_to_v2(v1, v2):
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    v = np.cross(v1_norm, v2_norm)
    s = np.linalg.norm(v)
    c = np.dot(v1_norm, v2_norm)

    R = np.eye(3) + matrixX(v) + ((1 - c) / (s * s)) * np.matmul(matrixX(v), matrixX(v))
    return R


def normalizeTrajectory(trac):
    gt_xyz = np.zeros(trac.shape)
    for i in range(gt_xyz.shape[1]):
        if i == 0:
            gt_xyz[:, 0] = trac[:, 0]
        else:
            vec = trac[:, i] - trac[:, i - 1]
            if np.linalg.norm(vec) == 0:
                gt_xyz[:, i] = trac[:, i]
            else:
                norm_vec = vec / np.linalg.norm(vec)
                gt_xyz[:, i] = gt_xyz[:, i - 1] + norm_vec
    return gt_xyz


def getIntervalTime(time_seq, interval):
    time_sec = time_seq * 1e-6
    temp_value = time_sec[0]
    time_index = [0]
    for i in range(time_sec.shape[0]):
        if time_sec[i] - temp_value > 1:
            temp_value = time_sec[i]
            time_index.append(i)
    return time_index


def computeDrift(time, gt_xyz, result_xyz, interval):
    angle = []
    scale = []
    distance = []
    for i in range(time.shape[0]):
        # Get Next time index
        first_time = time[i]
        second_time = time[i]
        third_time = time[i]
        if (time[-1] * 1e-6) - (first_time * 1e-6) < 6:
            break
        j = 1
        while True:
            second_time = time[i + j]
            if (second_time * 1e-6) - (first_time * 1e-6) > interval:
                break
            j += 1

        k = 1
        while True:
            if i + j + k >= time.shape[0]:
                return angle, scale, distance
            third_time = time[i + j + k]
            if (third_time * 1e-6) - (second_time * 1e-6) > interval:
                break
            k += 1

        if (time[i + j + k] - time[i + j]) * 1e-6 > 10:
            break
        if i + j + k >= time.shape[0]:
            break
        vec_gt_1 = gt_xyz[:, i + j] - gt_xyz[:, i]
        vec_gt_2 = gt_xyz[:, i + j + k] - gt_xyz[:, i + j]
        if np.linalg.norm(vec_gt_1) == 0 or np.linalg.norm(vec_gt_2) == 0:
            continue
        vec_result_1 = result_xyz[:, i + j] - result_xyz[:, i]
        vec_result_2 = result_xyz[:, i + j + k] - result_xyz[:, i + j]

        # Compute Rotate
        R = rotate_v1_to_v2(vec_result_1, vec_gt_1)
        vec_result_2_rotated = np.matmul(R, vec_result_2)

        # Compute Scale
        s1 = np.linalg.norm(vec_gt_1)
        s2 = np.linalg.norm(vec_result_1)
        ss = s1 / s2
        vec_result_2_rotated_scaled = vec_result_2_rotated * ss

        # Minimize this Rotation
        def loss(x):
            R = axis_angle(vec_gt_1, x)

            y = vec_gt_1 + np.matmul(R, vec_result_2_rotated_scaled)
            y_gt = vec_gt_1 + vec_gt_2
            y = np.linalg.norm(y - y_gt)
            return y

        x0 = 0
        res = least_squares(loss, x0)
        R = axis_angle(vec_gt_1, res.x)
        vec_result_2_minimized = np.matmul(R, vec_result_2_rotated_scaled)

        # Compute the angle between gt and result
        unit_vector_1 = vec_result_2_minimized / np.linalg.norm(vec_result_2_minimized)
        unit_vector_2 = vec_gt_2 / np.linalg.norm(vec_gt_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle_temp = np.rad2deg(np.arccos(dot_product))
        angle.append(angle_temp)
        distance.append(res.fun[0])

        # Compute The change in scale
        score = np.linalg.norm(vec_result_2_minimized) / np.linalg.norm(vec_gt_2)
        scale.append(score)
    return angle, scale, distance


def alignTrajectory(
    gt_start_time, result_time, result_xyz, location, sequence, interval
):
    diff = result_time - gt_start_time
    start_pos = 0
    for time in diff:
        if time < 0:
            start_pos += 1
            continue
        else:
            break
    # Find Translation
    gt_xy = location.map.sample(
        (result_time[start_pos:] - gt_start_time) * 1e-6,
        sequence.marker_annotation,
        sequence.is_backward,
    ).transpose(1, 0)
    gt_xy_start = gt_xy[:, 0]
    gt_xyz = np.zeros((3, gt_xy.shape[1]))
    gt_xyz[0, :] = gt_xy[0, :]
    gt_xyz[2, :] = gt_xy[1, :]
    result_xyz_cropped = result_xyz[:, start_pos:]
    result_time_cropped = result_time[start_pos:]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(gt_xyz[0, 0:40], gt_xyz[2, 0:40], s=1, c="g")
    # ax.scatter(
    #     result_xyz_cropped[0, 0:40],
    #     result_xyz_cropped[2, 0:40],
    #     s=1,
    #     c="b",
    # )
    # ax.scatter(gt_xyz[0, 0], gt_xyz[2, 0], s=4, c="m")
    # ax.set_aspect("equal", "box")
    # plt.savefig("Test.png")
    angles, scales, distances = computeDrift(
        result_time_cropped, gt_xyz, result_xyz_cropped, interval
    )

    return angles, scales, distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the time coverage of the results"
    )
    parser.add_argument("root", type=Path, help="the dataset root path")
    parser.add_argument("orbslam_path", type=Path, help="the output folder path")
    parser.add_argument("dso_path", type=Path, help="the output folder path")
    parser.add_argument("deep_path", type=Path, help="the output folder path")
    parser.add_argument("camera", type=str, help="the camera usage")
    parser.add_argument("figure_path", type=Path, help="the path of output figure")
    args = parser.parse_args()

    dataset = Dataset(args.root)
    result_file = open(
        args.figure_path
        / Path(args.camera)
        / Path("result_comparison").with_suffix(".txt"),
        "a",
    )

    for location in dataset.locations:
        print(location.name)
        for sequence in location.sequences:

            # if location.name != "SmithBuonanno":
            #     continue
            # if sequence.name != "forward_auto_exposure":
            #     continue
            # if not location.map.is_annotation_consistent(
            #     sequence.marker_annotation, sequence.is_backward
            # ):
            #     continue

            # print(sequence.name)

            interval = 1
            output_name = location.name + "-" + sequence.name
            # Get Ground Truth Sequences
            filenames = sequence.get_image_paths(
                sequence.cameras[args.camera],
                sequence.cameras[args.camera].streams["left"],
            )
            times_list = np.zeros(len(filenames))
            i = 0
            for filename in filenames:
                times_list[i] = int(filename.parts[-1].split(".")[0])
                i += 1

            gt_start_time = sequence.snap_annotation.cameras[args.camera].stereo.start
            diff_time = (times_list - gt_start_time) * 1e-6

            gt_xy = location.map.sample(
                diff_time, sequence.marker_annotation, sequence.is_backward
            ).transpose(1, 0)
            gt_xyz = np.zeros((3, gt_xy.shape[1]))
            gt_xyz[0, :] = gt_xy[0, :]
            gt_xyz[2, :] = gt_xy[1, :]

            # Read DSO result
            dso_name = sequence.name
            dso_filename = (
                args.dso_path
                / Path(args.camera)
                / Path(location.name + "-" + dso_name).with_suffix(".txt")
            )
            # Load DSO Result
            # dso_est = np.loadtxt(dso_filename)
            # dso_xyz = dso_est[:, 1:4].transpose(1, 0)
            # dso_time = (dso_est[:, 0] * 1e6).astype("int64")
            # angles_dso, scales_dso, distances_dso = alignTrajectory(
            #     gt_start_time, dso_time, dso_xyz, location, sequence, interval
            # )
            # print(mean(angles_dso) / interval)
            # print(mean(distances_dso) / interval)
            # # Load Orb SLAM Result monocular no loop closure
            orb_mono_noloop_name = sequence.name
            orb_mono_noloop_filename = (
                args.orbslam_path
                / Path("Monocular")
                / Path("Result_NoLoop")
                / Path(args.camera)
                / Path(location.name + "-" + orb_mono_noloop_name).with_suffix(".txt")
            )
            # Load ORB-SLAM Result
            orb_mono_noloop_est = np.loadtxt(orb_mono_noloop_filename)
            orb_mono_noloop_xyz = orb_mono_noloop_est[:, 1:4].transpose(1, 0)
            orb_mono_noloop_time = times_list[0] + orb_mono_noloop_est[:, 0] * 1e6
            angles_noloop, scales_noloop, distances_noloop = alignTrajectory(
                gt_start_time,
                orb_mono_noloop_time,
                orb_mono_noloop_xyz,
                location,
                sequence,
                interval,
            )

            # Load Orb SLAM Result monocular with loop closure
            orb_mono_loop_name = sequence.name
            orb_mono_loop_filename = (
                args.orbslam_path
                / Path("Monocular")
                / Path("Result_Loop")
                / Path(args.camera)
                / Path(location.name + "-" + orb_mono_loop_name).with_suffix(".txt")
            )
            # Load ORB-SLAM Result
            orb_mono_loop_est = np.loadtxt(orb_mono_loop_filename)
            orb_mono_loop_xyz = orb_mono_loop_est[:, 1:4].transpose(1, 0)
            orb_mono_loop_time = times_list[0] + orb_mono_loop_est[:, 0] * 1e6
            xyz_orb_mono_loop = alignTrajectory(
                gt_start_time,
                orb_mono_loop_time,
                orb_mono_loop_xyz,
                location,
                sequence,
                interval,
            )
            print(mean(angles_noloop) / interval)
            print(mean(xyz_orb_mono_loop[0]) / interval)
            print(mean(distances_noloop) / interval)
            print(mean(xyz_orb_mono_loop[2]) / interval)
            # ax.scatter(
            #     xyz_orb_mono_loop[0, :],
            #     xyz_orb_mono_loop[2, :],
            #     c="m",
            #     label="ORB-SLAM3 w/ Loop Closure",
            # )
            # # ax.set_aspect("equal", "box")
            # ax.legend()
            # plt.grid()
            # plt.autoscale()
            # plt.ioff()
            # plt.savefig("Test.png")
            # print(np.linalg.norm(gt_xyz[:, 0] - xyz_orb_mono_noloop[:, -1]))
            # print(np.linalg.norm(gt_xyz[:, 0] - xyz_orb_mono_loop[:, -1]))
            # Load Deep Method
            # deep_name = sequence.name
            # deep_filename = (
            #     args.deep_path
            #     / Path(args.camera)
            #     / Path(location.name + "-" + deep_name).with_suffix(".txt")
            # )
            # # Load ORB-SLAM Result
            # deep_est = np.loadtxt(deep_filename)
            # deep_xyz = deep_est[:, [3, 7, 11]].transpose(1, 0)
            # deep_time = times_list
            # xyz_deep = alignTrajectory(
            #     gt_start_time, deep_time, deep_xyz, location, sequence, interval
            # )
            # print(mean(xyz_deep[0]) / interval)
            # print(mean(xyz_deep[2]) / interval)
            # print("Angles:")
            # print(mean(angles_noloop) / interval)
            # print(mean(xyz_orb_mono_loop[0]) / interval)
            # print(mean(angles_dso) / interval)
            # print(mean(xyz_deep[0]) / interval)
            # print("Average Distance:")
            # print(mean(distances_noloop) / interval)
            # print(mean(xyz_orb_mono_loop[2]) / interval)
            # print(mean(distances_dso) / interval)
            # print(mean(xyz_deep[2]) / interval)
            print(0)
            # ax.scatter(
            #     xyz_deep[0, :],
            #     xyz_deep[2, :],
            #     s=1,
            #     c="k",
            #     label="TrianFlow",
            # )

            # # ax.set_aspect("equal", "box")
            # ax.legend()
            # # plt.grid()
            # plt.autoscale()
            # plt.ioff()
            # # plt.savefig("Test.png")
            # plt.savefig(output_name + ".png")

            # # compare point-to-point distance
            # result_file.write(location.name + "_" + output_name + " ")
            # result_file.write(str(m_e))
            # result_file.write("\n")
    result_file.close()
