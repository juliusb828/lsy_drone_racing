"""
Dynamic Trajectory Controller
Goal: generate the splines adaptively based on the track layout and recompute the trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from lsy_drone_racing.control import Controller
if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryController(Controller):

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        # keep original spline and then adjust if necessary
        initial_drone_pos = obs["pos"]
        #print(f"inital drone pos: {initial_drone_pos}")
        waypoints = np.array(
            [
                [initial_drone_pos[0], initial_drone_pos[1], initial_drone_pos[2]],
                [0.8, 1.0, 0.2],
                [0.6, -0.3, 0.5],
                [0.2, -1.3, 0.68],
                [1.1, -0.85, 1.25],
                [0.25, 0.5, 0.65],
                [0.15, 1.2, 0.525],
                [0.15, 1.2, 1.1],
                [-0.5, 0.0, 1.1],
                [-0.5, -0.5, 1.1],
            ]
        )
        self.t_total = 12
        t = np.linspace(0, self.t_total, len(waypoints))
        self.trajectory = CubicSpline(t, waypoints)
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        print(obs["pos"])
        tau = min(self._tick / self._freq, self.t_total)
        target_pos = self.trajectory(tau)
        if tau == self.t_total:  # Maximum duration reached
            self._finished = True
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished
        """Reset the time step counter."""
        self._tick = 0
