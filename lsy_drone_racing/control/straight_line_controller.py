"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

#added imports
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt

def add_detours_around_obstacles(waypoints, obstacles, safety_radius=0.15):
    def point_line_distance_2d(p, a, b):
        p2d, a2d, b2d = p[:2], a[:2], b[:2]
        ab = b2d - a2d
        ap = p2d - a2d
        if np.allclose(ab, 0):
            return np.linalg.norm(ap), a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        closest_2d = a2d + t * ab
        closest_3d = np.array([closest_2d[0], closest_2d[1], a[2]])
        return np.linalg.norm(p2d - closest_2d), closest_3d

    adjusted_waypoints = [waypoints[0]]

    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        detour_added = False

        path = b - a
        path_norm = np.linalg.norm(path)
        if path_norm < 1e-6:
            adjusted_waypoints.append(b)
            continue

        path_dir = path / path_norm

        for obs in obstacles:
            dist, closest = point_line_distance_2d(obs, a, b)
            if dist < safety_radius:
                # Choose a perpendicular direction
                detour_dir = np.cross(path_dir, np.array([0, 0, 1]))
                if np.linalg.norm(detour_dir) < 1e-6:
                    detour_dir = np.cross(path_dir, np.array([0, 1, 0]))

                detour_dir /= np.linalg.norm(detour_dir)
                detour = closest + detour_dir * (safety_radius + 0.3)

                adjusted_waypoints.append(detour)
                detour_added = True
                break

        adjusted_waypoints.append(b)

    return np.array(adjusted_waypoints)




class TrajectoryController(Controller):
    """Trajectory controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        # keep original spline and then adjust if necessary
        initial_drone_pos = obs["pos"]
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        obstacles_pos = obs["obstacles_pos"]
        waypoints = []
        waypoints.append([initial_drone_pos[0], initial_drone_pos[1], initial_drone_pos[2]])
        for gate_pos, gate_quat in zip(gates_pos, gates_quat):
            rot = R.from_quat(gate_quat).as_matrix()
            fwd = rot[:, 1]
            entry = gate_pos - 0.2 * fwd
            exit = gate_pos + 0.3 * fwd
            waypoints.append(entry)
            #waypoints.append(gate_pos)
            waypoints.append(exit)
        waypoints = np.array(waypoints)
        waypoints = add_detours_around_obstacles(waypoints, obstacles_pos)
        print(waypoints)
        self.t_total = 26  # total time duration
        t = np.linspace(0, self.t_total, len(waypoints))  # parameter vector
        # Create piecewise linear interpolator for each dimension
        self.trajectory = interp1d(t, waypoints, kind='linear', axis=0)
        
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False

        self.last_known_gates_pos = obs["gates_pos"]
        self.last_known_obstacles_pos = obs["obstacles_pos"]

        self.last_extra_point = None
        
        self.trajectory_history = []
        self.trajectory_history.append(waypoints.copy())

        self.gates_passed = [False, False, False, False]

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """ 
        tau = min(self._tick / self._freq, self.t_total)
        #print(obs)
        target_gate = obs["target_gate"]
        self.gates_passed = np.zeros(4, dtype=bool)
        self.gates_passed[:target_gate] = True

        if not np.allclose(obs["gates_pos"], self.last_known_gates_pos, atol=0.001):
            print("GATE POSITION CHANGED, RECOMPUTE TRAJECTORY!!!")
            self.last_known_gates_pos = obs["gates_pos"]
            self.trajectory = self.recompute_trajectory(obs["pos"], obs["gates_pos"], obs["gates_quat"], obs["obstacles_pos"], tau, target_gate)

        if not np.allclose(obs["obstacles_pos"], self.last_known_obstacles_pos, atol=0.001):
            print("OBSTACLE POSITION CHANGED, RECOMPUTE TRAJECTORY!!!")
            self.last_known_obstacles_pos = obs["obstacles_pos"]
            self.trajectory = self.recompute_trajectory(obs["pos"], obs["gates_pos"], obs["gates_quat"], obs["obstacles_pos"], tau, target_gate)

        target_pos = self.trajectory(tau)

        if tau == self.t_total+1:
            self._finished = True

        # Return [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)

    
    def recompute_trajectory(self, pos, gates_pos, gates_quat, obstacles_pos, tau, target_gate):
        waypoints = [[pos[0], pos[1], pos[2]]]
        print("start of recomputation")
        for gate_pos, gate_quat, gate_passed in zip(gates_pos, gates_quat, self.gates_passed):
            if not gate_passed:
                rot = R.from_quat(gate_quat).as_matrix()
                fwd = rot[:, 1]
                sideways = rot[:, 0]
                entry = gate_pos - 0.2 * fwd
                exit = gate_pos + 0.2 * fwd
                pos_close_to_gate_2 = (
                    abs(pos[0] - 0.0) <= 0.3 and
                    abs(pos[1] - 1.0) <= 0.3 and
                    abs(pos[2] - 0.56) <= 0.2
                )
                if target_gate==3 and pos_close_to_gate_2:
                    extra_point = self.last_extra_point
                    print(f"extra_point at {extra_point}")
                    waypoints.append(extra_point)
                waypoints.append(entry)
                waypoints.append(exit)
                is_gate_3 = (
                    abs(gate_pos[0] - 0.0) <= 0.15 and
                    abs(gate_pos[1] - 1.0) <= 0.15 and
                    abs(gate_pos[2] - 0.56) <= 0.1
                )
                if is_gate_3:
                    extra_point = exit - 0.7*sideways
                    self.last_extra_point = extra_point
                    print(f"extra_point at {extra_point}")
                    waypoints.append(extra_point)
                

        waypoints = np.array(waypoints)

        # Add detours
        waypoints = add_detours_around_obstacles(waypoints, obstacles_pos)

        # Defensive: Prevent duplicate time values
        remaining_time = self.t_total - tau
        if remaining_time < 1e-3:
            remaining_time = 1.0  # minimum time window for valid interpolation

        t = np.linspace(tau, tau + remaining_time, len(waypoints))
        if np.any(np.diff(t) <= 0):
            t += np.linspace(0, 1e-3, len(t))  # ensure strict monotonicity

        self.trajectory_history.append(waypoints.copy())
        print("Trajectory recomputed. Plotting...")
        #self.plot_all_trajectories()

        return interp1d(t, waypoints, kind='linear', axis=0, fill_value='extrapolate')

    
#    def plot_all_trajectories(self):
#        plt.figure(figsize=(10, 8))
#        for i, wp in enumerate(self.trajectory_history):
#            if isinstance(wp, dict):
#                wp = wp["waypoints"]
#            x, y = wp[:, 0], wp[:, 1]
#           plt.plot(x, y, '-o', label=f'Trajectory {i+1}')
#
#        if hasattr(self, "last_known_obstacles_pos") and self.last_known_obstacles_pos is not None:
#            obs = self.last_known_obstacles_pos
#            plt.scatter(obs[:, 0], obs[:, 1], color='black', marker='X', s=100, label='Obstacles')
#        plt.title("All Recomputed Trajectories")
#        plt.xlabel("X")
#        plt.ylabel("Y")
#        plt.axis('equal')
#        plt.grid(True)
#        plt.legend()
#        plt.savefig("trajectory_debug_plot.png")
#        plt.close()


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
