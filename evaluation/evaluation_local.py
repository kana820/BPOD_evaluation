import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from math import cos, sin
from statistics import mean

photo_slam = -1

def axis_angle(vec, theta):
    vec_normal = vec / np.linalg.norm(vec)

    R = np.zeros((3, 3))
    theta = theta.item()
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

def computeDrift(time, gt_xyz, result_xyz, interval):
    angle = []  # in degrees
    scale = []  
    distance = []
    for i in range(time.shape[0]):
        if not (photo_slam):
            if i + 2*interval >= time.shape[0]:
                break
            first_time = time[i]
            second_time = time[i + interval] 
            third_time = time[i + 2*interval] 
            j = interval
            k = interval 

        else:
            first_time = time[i]
            # Get Next time index
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
    gt_xyz, result_xyz, result_time, interval
):
    # Find Translation
    translation = (
        gt_xyz[:, 0] - result_xyz[:, 0]
    )
    print("Translation of Alignment:", translation)
    result_xyz_translated = result_xyz + translation.reshape(3, 1)

    angles, scales, distances = computeDrift(
        result_time, gt_xyz, result_xyz_translated, interval
    )

    return angles, scales, distances


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
    parser.add_argument("algo", help="the algorithm that generated the result (Photo-SLAM or DSV-SLAM)")
    args = parser.parse_args()

    if (args.algo == 'photo'):
        photo_slam = 1
    elif (args.algo == 'dsv'):
        photo_slam = 0
    else:
        exit(1)

    if (photo_slam == -1): exit(1)
    if (photo_slam): # Photo-SLAM
        interval = 1
    else:  # DSV-SLAM
        interval = 30

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

    # Read estimated result
    est_data = np.loadtxt(args.data_path)
    est_xyz = est_data[:, 1:4].transpose()
    est_time = (est_data[:, 0]).astype("int64")
    angles, scales, distances = alignTrajectory(
        gt_xyz, est_xyz, est_time, interval
    )

    # Local Drifts
    # Angles
    plt.figure()
    x1 = np.arange(len(angles))
    plt.bar(x1, angles)
    plt.ylabel("Angle [deg]")
    plt.title("Local Drift: Angles [deg]")
    plt.savefig("result_local_angle.png")

    # Distances
    plt.figure()
    x2 = np.arange(len(distances))
    plt.bar(x2, distances)
    plt.ylabel("Distance d_l [m]")
    plt.title("Local Drift: Distance [m]")
    plt.savefig("result_local_distance.png")


    print("Mean Drift in Angles:", mean(angles))  # per second
    print("Mean Drift in Distance:", mean(distances))