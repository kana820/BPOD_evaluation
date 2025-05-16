import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

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

def alignTrajectory(gt_xyz, result_xyz):

    # Compute
    # Align two trajectories at first point
    translation = (
        gt_xyz[:, 0] - result_xyz[:, 0]
    )
    print("Translation of Alignment:", translation)
    result_xyz_translated = result_xyz + translation.reshape(3, 1)

    # Find Initial Scale and Rotation
    npoint = int(result_xyz_translated.shape[1] * 0.4)
    npoint2 = int(result_xyz_translated.shape[1])
    # The bearing vector of estimation
    next_xyz = result_xyz_translated[:, npoint]
    first_xyz = result_xyz_translated[:, 0]
    bearing_vec = next_xyz - first_xyz
    # The bearing vector of ground truth
    next_xy_gt = gt_xyz[:2, npoint]
    first_xy_gt = gt_xyz[:2, 0]
    bearing_vec_gt = next_xy_gt - first_xy_gt

    # Initial Scale:
    norm_est = np.linalg.norm(bearing_vec)
    norm_gt = np.linalg.norm(bearing_vec_gt)
    scale = norm_gt / norm_est
    result_xyz_translated = result_xyz_translated * scale

    def loss(x):
        R = exp(x[0:3].reshape(1, 3, 1))
        s = x[3]
        y = (gt_xyz - gt_xyz[:, 0].reshape(3, 1)) - (
            R @ ((result_xyz_translated - gt_xyz[:, 0].reshape(3, 1)) * s)
        )[0, :, :]
        y = y[0, :] * y[0, :] + y[1, :] * y[1, :] + y[2, :] * y[2, :]
        return y

    x0 = np.random.rand(4)
    x0[3] = scale  # scale
    res = least_squares(loss, x0)
    s = res.x[3]
    print("Scale:", s)
    R = exp(res.x[0:3].reshape(1, 3, 1))[0, :, :]
    result_xyz_translated = (
        result_xyz_translated - gt_xyz[:, 0].reshape(3, 1)
    ) * s
    # # Initial Rotation: for global alignment
    # R = rotate_v1_to_v2(
    #     bearing_vec, np.array([bearing_vec_gt[0], 0, bearing_vec_gt[1]])
    # )
    # print("Rotation of Alignment:\n", R)
    # result_xyz_translated_rotated = np.matmul(
    #     R, result_xyz_translated
    # ) + gt_xyz[:, 0].reshape(3, 1)
    
    # return result_xyz_translated_rotated
    
    return result_xyz_translated

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate the time coverage of the results"
    )
    # Photo-SLAM
    # timestamp_path = "../results/Photo-SLAM/timestamps.txt"
    # data_path = "../results/Photo-SLAM/photo-slam.txt"
    # gt_path = "../gt_455/d455-BDW-backward_auto_exposure.txt"

    # DSV-SLAM
    # timestamp_path = "../results/DSV-SLAM/timestamps.txt"
    # data_path = "../results/DSV-SLAM/dslam_bpod_tum.txt"
    # gt_path = "../results/DSV-SLAM/gt_tum.txt"
    parser.add_argument("timestamp_path", help="the timestamp path")
    parser.add_argument("data_path", help="the estimated data path")
    parser.add_argument("gt_path", help="the ground truth data path")
    args = parser.parse_args()

    ## Ground Truth ##
    # Get Ground Truth Sequences    
    num_img = 0
    start_time = 0
    with open(args.timestamp_path, "r") as file:
        for ts in file:
            if (num_img == 0): start_time = int(ts)
            num_img += 1
    print("number of samples for evaluation:", num_img)

    gt_data = np.loadtxt(args.gt_path)  
    gt_times = gt_data[:, 0]
    gt_xy = gt_data[:, 1:3].transpose()

    timestamps = np.loadtxt(args.timestamp_path)
    mask = np.isin(gt_times, timestamps)

    gt_xyz = np.zeros((3, num_img))
    gt_xyz[0, :] = gt_xy[0, mask]
    gt_xyz[2, :] = gt_xy[1, mask]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        gt_xyz[0, :],
        gt_xyz[2, :],
        c="g",
        s=1,
        label="Ground Truth",
    )
    ax.scatter(
        gt_xyz[0, 0], gt_xyz[2, 0], s=50, c="k", marker="s", label="StartPoint"
    )
    ax.set_aspect("equal", "box")
    ax.legend()
    plt.grid()
    plt.ioff()
    # plt.savefig("Test.png")

    # #################### Estimated Trajectories #######################################
    # Load Estimated Result
    est_data = np.loadtxt(args.data_path)
    est_xyz = est_data[:, 1:4].transpose()
    # est_time = (est_data[:, 0] * 1e6).astype("int64")
    xyz_data = alignTrajectory(gt_xyz, est_xyz)
    ax.scatter(
        xyz_data[0, :],
        xyz_data[2, :],
        c="r",
        s=1,
        label="Estimated",
    )
    ax.set_aspect("equal", "box")
    ax.legend()
    plt.xlabel("x")
    plt.ylabel("z")
    plt.grid()
    plt.ioff()
    plt.savefig("result_global.png")
    print("Endpoint Distance:")
    print(np.linalg.norm(gt_xyz[:, -1] - xyz_data[:, -1]))