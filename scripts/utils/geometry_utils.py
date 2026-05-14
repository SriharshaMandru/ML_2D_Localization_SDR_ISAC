import numpy as np


def angle_distance_to_xy(angle_deg, distance_m):
    theta = np.radians(angle_deg)
    x = distance_m * np.sin(theta)
    y = distance_m * np.cos(theta)
    return x, y


def xy_to_angle_distance(x, y):
    distance = np.sqrt(x * x + y * y)
    angle = np.degrees(np.arctan2(x, y))
    return angle, distance


def feedback_correct_xy(initial_xy, pred_angle, pred_distance, iterations=3, lr=0.25):
    xy = np.array(initial_xy, dtype=float)

    for _ in range(iterations):
        x, y = xy
        est_angle, est_distance = xy_to_angle_distance(x, y)

        angle_error_rad = np.radians(pred_angle - est_angle)
        distance_error = pred_distance - est_distance

        theta = np.radians(est_angle)
        radial = np.array([np.sin(theta), np.cos(theta)])
        tangent = np.array([np.cos(theta), -np.sin(theta)])

        correction = (
            lr * distance_error * radial
            + lr * est_distance * angle_error_rad * tangent
        )

        xy = xy + correction

    return xy
