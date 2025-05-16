import os
import numpy as np
import argparse
from pathlib import Path
from os import path
from file_system import listdir
from colors import cyan
from dataset import Dataset
from statistics import mean, median
import matplotlib.pyplot as plt
from statistics import mean
import math
from scipy.optimize import least_squares
import pickle as pkl
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


def alignTrajectory(gt_start_time, result_time, result_xyz, location, sequence):
    diff = result_time - gt_start_time
    start_pos = 0
    for time in diff:
        if time < 0:
            start_pos += 1
            continue
        else:
            break
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(
    #     result_xyz[0, :],
    #     result_xyz[1, :],
    #     result_xyz[2, :],
    #     c="g",
    #     label="result",
    # )
    # plt.savefig("Result.png")
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

    # Compute
    # Align two trajectories at first point
    translation = (
        np.array([gt_xy_start[0], 0, gt_xy_start[1]]) - result_xyz_cropped[:, 0]
    )
    result_xyz_cropped_tranlated = result_xyz_cropped + translation.reshape(3, 1)

    # # Find Initial Scale and Rotation
    # npoint = int(result_xyz_cropped_tranlated.shape[1] * 0.4)
    # npoint2 = int(result_xyz_cropped_tranlated.shape[1])
    # # The bearing vector of estimation
    # next_xyz = result_xyz_cropped_tranlated[:, npoint]
    # first_xyz = result_xyz_cropped_tranlated[:, 0]
    # bearing_vec = next_xyz - first_xyz
    # # The bearing vector of ground truth
    # next_xy_gt = gt_xy[:, npoint]
    # first_xy_gt = gt_xy[:, 0]
    # bearing_vec_gt = next_xy_gt - first_xy_gt

    # # Initial Scale:
    # norm_est = np.linalg.norm(bearing_vec)
    # norm_gt = np.linalg.norm(bearing_vec_gt)
    # scale = norm_gt / norm_est
    # result_xyz_cropped_translated = result_xyz_cropped_tranlated * scale

    def loss(x):
        R = exp(x[0:3].reshape(1, 3, 1))
        s = x[3]
        y = (gt_xyz - gt_xyz[:, 0].reshape(3, 1)) - (
            R @ ((result_xyz_cropped_tranlated - gt_xyz[:, 0].reshape(3, 1)) * s)
        )[0, :, :]
        y = y[0, :] * y[0, :] + y[1, :] * y[1, :] + y[2, :] * y[2, :]
        return y

    x0 = np.random.rand(4)
    x0[3] = 50  # scale
    res = least_squares(loss, x0)
    s = res.x[3]
    R = exp(res.x[0:3].reshape(1, 3, 1))[0, :, :]
    result_xyz_cropped_translated = (
        result_xyz_cropped_tranlated - gt_xyz[:, 0].reshape(3, 1)
    ) * s
    # Initial Rotation:
    # R = rotate_v1_to_v2(
    #     bearing_vec, np.array([bearing_vec_gt[0], 0, bearing_vec_gt[1]])
    # )
    result_xyz_cropped_translated_rotated = np.matmul(
        R, result_xyz_cropped_translated
    ) + gt_xyz[:, 0].reshape(3, 1)

    return result_xyz_cropped_translated_rotated


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

            if location.name != "CIC_Balcony":
                continue
            if sequence.name != "backward_auto_exposure":
                continue

            output_name = location.name + "-" + sequence.name

            ######################Ground Truth Part#################################################
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

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(
                gt_xy[0, :],
                gt_xy[1, :],
                c="g",
                label="Ground Truth",
            )
            ax.scatter(
                gt_xy[0, 0], gt_xy[1, 0], s=50, c="k", marker="s", label="StartPoint"
            )
            # ax.set_aspect("equal", "box")
            ax.legend()
            plt.grid()
            plt.ioff()
            plt.savefig("Test.png")

            ##################################DSO##############################################
            dso_name = sequence.name
            dso_filename = (
                args.dso_path
                / Path(args.camera)
                / Path(location.name + "-" + dso_name).with_suffix(".txt")
            )
            # Load DSO Result
            dso_est = np.loadtxt(dso_filename)
            dso_xyz = dso_est[:, 1:4].transpose(1, 0)
            dso_time = (dso_est[:, 0] * 1e6).astype("int64")
            xyz_dso = alignTrajectory(
                gt_start_time, dso_time, dso_xyz, location, sequence
            )
            ax.scatter(
                xyz_dso[0, :],
                xyz_dso[2, :],
                s=1,
                c="r",
                label="DSO",
            )
            # ax.set_aspect("equal", "box")
            ax.legend()
            plt.grid()
            plt.ioff()
            plt.savefig("Test.png")
            print("DSO:")
            print(np.linalg.norm(gt_xyz[:, 0] - xyz_dso[:, -1]))

            ################################ORB-SLAM No Loop###########################
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
            xyz_orb_mono_noloop = alignTrajectory(
                gt_start_time,
                orb_mono_noloop_time,
                orb_mono_noloop_xyz,
                location,
                sequence,
            )
            ax.scatter(
                xyz_orb_mono_noloop[0, :],
                xyz_orb_mono_noloop[2, :],
                s=3,
                c="b",
                label="ORB-SLAM3 w/o Loop Closure",
            )
            ax.set_aspect("equal", "box")
            ax.legend()
            plt.grid()
            plt.ioff()
            plt.savefig("Test.png")
            print("ORB-SLAM No Loop:")
            print(np.linalg.norm(gt_xyz[:, 0] - xyz_orb_mono_noloop[:, -1]))
            ####################################ORB-SLAM Loop###########################
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
                gt_start_time, orb_mono_loop_time, orb_mono_loop_xyz, location, sequence
            )
            ax.scatter(
                xyz_orb_mono_loop[0, :],
                xyz_orb_mono_loop[2, :],
                s=3,
                c="m",
                label="ORB-SLAM3 w/ Loop Closure",
            )
            # ax.set_aspect("equal", "box")
            ax.legend()
            plt.grid()
            plt.autoscale()
            plt.ioff()
            plt.savefig("Test.png")
            print("ORB-SLAM Loop:")
            print(np.linalg.norm(gt_xyz[:, 0] - xyz_orb_mono_loop[:, -1]))
            ################################DEEP VO (TrianFlow)#########################
            # Load Deep Method
            deep_name = sequence.name
            deep_filename = (
                args.deep_path
                / Path(args.camera)
                / Path(location.name + "-" + deep_name).with_suffix(".txt")
            )
            # Load ORB-SLAM Result
            deep_est = np.loadtxt(deep_filename)
            deep_xyz = deep_est[:, [3, 7, 11]].transpose(1, 0)
            deep_time = times_list
            xyz_deep = alignTrajectory(
                gt_start_time,
                deep_time,
                deep_xyz,
                location,
                sequence,
            )
            ax.scatter(
                xyz_deep[0, :],
                xyz_deep[2, :],
                s=1,
                c="k",
                label="TrianFlow",
            )

            ax.set_aspect("equal", "box")
            ax.legend()
            # plt.grid()
            plt.autoscale()
            plt.ioff()
            plt.savefig("Test.png")
            print("DEEP:")
            print(np.linalg.norm(gt_xyz[:, 0] - xyz_deep[:, -1]))

            i = 1