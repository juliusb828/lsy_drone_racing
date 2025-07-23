"""Custom test cases for mpc_trajectory_tracking, run in micromamba environment with pytest tests/unit/custom_tests."""

import sys
from pathlib import Path

import numpy as np

# Add the project root to sys.path so Python can find the package
sys.path.insert(0, str(Path(__file__).parents[3]))

from types import SimpleNamespace

from lsy_drone_racing.control.mpc_trajectory_following import MPController  # type: ignore


# to check if it works
def test_dummy():
    assert 5 == 5


def test_get_closest_point():
    obs = {
        "pos": np.array([1.0896959, 1.4088244, 0.08456537], dtype=np.float32),
        "quat": np.array([-0.03827618, -0.0140878, -0.02179931, 0.99893], dtype=np.float32),
        "vel": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "ang_vel": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "target_gate": np.array(0, dtype=np.int32),
        "gates_pos": np.array(
            [[0.45, -0.5, 0.56], [1.0, -1.05, 1.11], [0.0, 1.0, 0.56], [-0.5, 0.0, 1.11]],
            dtype=np.float32,
        ),
        "gates_quat": np.array(
            [
                [0.0, 0.0, 0.92268986, 0.38554308],
                [0.0, 0.0, -0.38018841, 0.92490906],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.9999997, 0.00079632673],
            ],
            dtype=np.float32,
        ),
        "gates_visited": np.array([False, False, False, False]),
        "obstacles_pos": np.array(
            [[1.0, 0.0, 1.4], [0.5, -1.0, 1.4], [0.0, 1.5, 1.4], [-0.5, 0.5, 1.4]], dtype=np.float32
        ),
        "obstacles_visited": np.array([False, False, False, False]),
    }
    info = {}
    # Simplified config with only env.freq
    config = {
        "env": {
            "freq": 50
        }
    }

    # Convert config and its nested keys to objects
    config["env"] = SimpleNamespace(**config["env"])
    config_obj = SimpleNamespace(**config)

    # create an instance of the MPController class
    mpc_controller = MPController(obs, info, config_obj)

    x_path = [0, 1, 2, 3, 4, 5]
    y_path = [0, 1, 2, 3, 4, 5]
    z_path = [0, 1, 2, 3, 4, 5]
    trajectory = list(zip(x_path, y_path, z_path))

    x_point = 1.2
    y_point = 1.2
    z_point = 1.5
    #test_point = (x_point, y_point, z_point)

    # call the get_closest_point method
    closest_point_idx_1 = mpc_controller.find_closest_point(x_path, y_path, z_path, x_point, y_point, z_point)

    # Assert the expected closest point
    assert closest_point_idx_1 == 1, (
        f"Expected point at index 1 (x = {trajectory[1]}), but got {closest_point_idx_1}"
    )

    x_point_2 = 4.5
    y_point_2 = 4.2
    z_point_2 = 4.5

    closest_point_idx_2 = mpc_controller.find_closest_point(x_path, y_path, z_path, x_point_2, y_point_2, z_point_2)

    assert closest_point_idx_2 == 4, (
        f"Expected point at index 1 (x = {trajectory[1]}), but got {closest_point_idx_1}"
    )
