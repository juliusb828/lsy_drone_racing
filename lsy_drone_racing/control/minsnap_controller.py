from __future__ import annotations  # Python 3.10 type hints  # noqa: D100

from typing import TYPE_CHECKING

import minsnap_trajectories as ms  # type: ignore
import numpy as np

#from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MinSnapController(Controller):
    """A controller that uses the minsnap python package to generate a trajectory and follow it."""
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initializes the controller by generating a base trajectory."""
        super().__init__(obs, info, config)

        self.last_known_gates_pos = obs["gates_pos"]
        self.last_known_obstacles_pos = obs["obstacles_pos"]
        self.trajectory_history = []
        self.gates_passed = [False, False, False, False]

        self.t_total = 7.0  # total time duration

        self._prev_yaw = None
        self._prev_tau = None

        self.target_positions, self.target_velocities, self.target_accelerations = self.compute_trajectory(
            obs["pos"],
            obs["gates_pos"],
            obs["gates_quat"],
            obs["obstacles_pos"],
            0,
            obs["target_gate"],
        )

        #print(self.target_positions)

        self._tick = 0
        self._freq = config.env.freq
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone, updates the trajectory if position of gates or obstacles changes.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        tau = min(self._tick / self._freq, self.t_total)
        # frequency is 50 Hz
        target_gate = obs["target_gate"]
        self.gates_passed = np.zeros(4, dtype=bool) 
        self.gates_passed[:target_gate] = True

        current_target_pos = self.target_positions[self._tick]
        current_target_vel = self.target_velocities[self._tick]
        current_target_acc = self.target_accelerations[self._tick]
        #print(obs["pos"])

        yaw = np.array([np.arctan2(current_target_vel[1], current_target_vel[0])])

        if tau == self.t_total:  # Maximum duration reached
            self._finished = True

        return np.concatenate(
            (current_target_pos, current_target_vel, current_target_acc, yaw, np.zeros(3)), dtype=np.float32
        )

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

    def compute_trajectory(
        self,
        pos: NDArray[np.floating],
        gates_pos: NDArray[np.floating],
        gates_quat: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
        tau: float,
        target_gate: int,
    ) -> tuple[ms.PolynomialTrajectory, list[ms.Waypoint]]:
        """Recompute the trajectory based on the current position, gates, and obstacles.

        Args:
            pos: Current position of the drone as a numpy array.
            gates_pos: Positions of the gates as a numpy array.
            gates_quat: Orientations of the gates as a numpy array of quaternions.
            obstacles_pos: Positions of the obstacles as a numpy array.
            tau: Current time in the trajectory, used as starting time.
            target_gate: Index of the target gate.

        Returns:
            A tuple containing the polynomial trajectory and the list of waypoints.
        """
        waypoints = [
            ms.Waypoint(
                time=0.0,
                position=np.array([pos[0], pos[1], pos[2]])
            ),
            ms.Waypoint(
                time=self.t_total/4.0,
                position=np.array([gates_pos[0][0], gates_pos[0][1], gates_pos[0][2]]),
            ),
            ms.Waypoint(
                time=2*(self.t_total/4.0),
                position=np.array([gates_pos[1][0], gates_pos[1][1], gates_pos[1][2]]),
            ),
            ms.Waypoint(
                time=3*(self.t_total/4.0),
                position=np.array([gates_pos[2][0], gates_pos[2][1], gates_pos[2][2]]),
            ),
            ms.Waypoint(
                time=4*(self.t_total/4.0),
                position=np.array([gates_pos[3][0], gates_pos[3][1], gates_pos[3][2]]),
            ),
        ]

        polys = ms.generate_trajectory(
            waypoints,
            degree=8,  # Polynomial degree
            idx_minimized_orders=(3, 4),
            num_continuous_orders=3,
            algorithm="closed-form",  # Or "constrained"
        )

        time_steps = int(self.t_total*50)
        t = np.linspace(0, self.t_total, time_steps)
        #  Sample up to the 3rd order (Jerk) -----v
        pva = ms.compute_trajectory_derivatives(polys, t, 3)
        target_pos = pva[0, ... ]
        target_vel = pva[1, ...]
        target_acc = pva[2, ...]

        safe_distance = 0.8  # Minimum safe distance from obstacles
        self.check_trajectory_collision(target_pos, obstacles_pos, safe_distance)
            

        return target_pos, target_vel, target_acc
    
    def check_trajectory_collision(
        self,
        target_pos: np.ndarray,
        obstacles_pos: NDArray[np.floating],
        safe_distance: float
    ) -> bool:
        """Check if the trajectory intersects or gets too close to obstacles.

        Args:
            target_pos: Array of trajectory positions (shape: [time_steps, 3]).
            obstacles_pos: Array of obstacle positions (shape: [num_obstacles, 3]).
            safe_distance: Minimum safe distance from obstacles.

        Returns:
            True if a collision is detected, False otherwise.
        """
        for pos in target_pos:
            for obstacle in obstacles_pos:
                distance = np.linalg.norm(pos - obstacle)
                if distance < safe_distance:
                    print(f"Warning: at pos {pos}")
                    return True  # Collision detected
        return False  # No collision
