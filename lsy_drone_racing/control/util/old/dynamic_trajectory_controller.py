from __future__ import annotations  # Python 3.10 type hints
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from lsy_drone_racing.control import Controller
if TYPE_CHECKING:
    from numpy.typing import NDArray

#added imports
from scipy.spatial.transform import Rotation as R
import minsnap_trajectories as ms  # type: ignore
import matplotlib.pyplot as plt

class DynamicTrajectoryController(Controller):
    
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        
        self.last_known_gates_pos = obs["gates_pos"]
        self.last_known_obstacles_pos = obs["obstacles_pos"]
        self.trajectory_history = []
        self.gates_passed = [False, False, False, False]

        self.t_total = 30  # total time duration
        
        self._prev_yaw = None
        self._prev_tau = None
        
        self.trajectory, self.waypoints = self.recompute_trajectory(obs["pos"], obs["gates_pos"], obs["gates_quat"], obs["obstacles_pos"], 0, obs["target_gate"])
        self.plot_trajectory()
        
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        tau = min(self._tick / self._freq, self.t_total)
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

        t = np.linspace(tau, self.t_total, 100)
        #  Sample up to the 3rd order (Jerk) -----v
        pva = ms.compute_trajectory_derivatives(self.trajectory, [tau], 3)
        target_pos = pva[0, 0]
        target_vel = pva[1, 0]
        target_acc = pva[2, 0]
        yaw = np.array([np.arctan2(target_vel[1], target_vel[0])])
        
        if self._prev_yaw is not None and self._prev_tau is not None:
            dt = tau - self._prev_tau
            if dt > 0:
                def angle_difference(a, b):
                    return (a - b + np.pi) % (2 * np.pi) - np.pi

                yaw_rate = angle_difference(yaw, self._prev_yaw) / dt
            else:
                yaw_rate = 0.0
        else:
            yaw_rate = 0.0

        self._prev_yaw = yaw
        self._prev_tau = tau
        yaw_rate = np.array([yaw_rate]).flatten()

        if tau == self.t_total:  # Maximum duration reached
            self._finished = True
        current_pos = obs["pos"]
        error = target_pos - current_pos
        print(f"error: {error}")
        return np.concatenate((target_pos, target_vel, target_acc, yaw, np.zeros(2), yaw_rate), dtype=np.float32)

    def step_callback(self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]], reward: float, terminated: bool, truncated: bool, info: dict) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished
        """Reset the time step counter."""
        self._tick = 0

    def recompute_trajectory(self, pos, gates_pos, gates_quat, obstacles_pos, tau, target_gate):
        waypoints = [
            ms.Waypoint(time=tau, position=np.array([pos[0], pos[1], pos[2]]))
        ]

        dt = self.t_total/12.0
        t = 0
        for gate_pos, gate_quat, gate_passed in zip(gates_pos, gates_quat, self.gates_passed):
            if gate_passed == False:
                rot = R.from_quat(gate_quat).as_matrix()
                fwd = rot[:, 1]
                entry = gate_pos - 0.2 * fwd
                exit = gate_pos + 0.2 * fwd
                t += 2*dt
                waypoints.append(ms.Waypoint(time=t,position=np.array((entry[0], entry[1], entry[2]))))
                t += dt
                waypoints.append(ms.Waypoint(time=t,position=np.array((exit[0], exit[1], exit[2]))))

        waypoints = self.add_detours_around_obstacles(waypoints, obstacles_pos)

        print(waypoints)
        
        polys = ms.generate_trajectory(
            waypoints,
            degree=8,  # Polynomial degree
            idx_minimized_orders=(3, 4),  
            num_continuous_orders=3,  
            algorithm="closed-form",  # Or "constrained"
        )
        return polys, waypoints

    def plot_trajectory(self):
        # Generate time samples across the full trajectory duration
        t = np.linspace(0, self.t_total, 100)

        # Compute derivatives up to acceleration (order 3)
        pva = ms.compute_trajectory_derivatives(self.trajectory, t, 3)
        position, velocity, acceleration = pva

        # Create 3D plot
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))
        ax.plot(position[:, 0], position[:, 1], position[:, 2], label="Trajectory")

        # Plot waypoints
        position_waypoints = np.array([wp.position for wp in self.waypoints])
        ax.plot(
            position_waypoints[:, 0],
            position_waypoints[:, 1],
            position_waypoints[:, 2],
            "ro",
            label="Waypoints",
        )

        # Optional: plot velocity vector at a waypoint (e.g., the second one)
        if hasattr(self.waypoints[1], 'velocity') and self.waypoints[1].velocity is not None:
            ax.quiver(
                *self.waypoints[1].position,
                *self.waypoints[1].velocity,
                color="g",
                length=1.0,
                normalize=True,
                label="Velocity @ WP1",
            )

        # Labels and legend
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_zlim(8, 12)
        ax.legend(loc="upper right")
        fig.tight_layout()

        # Save or show
        try:
            fig.savefig("example/minsnap_trajectory_plot.png")
        except FileNotFoundError:
            plt.show()

    
    def add_detours_around_obstacles(self, waypoints, obstacles, safety_radius=0.15):
        adjusted_waypoints = [waypoints[0]]

        for i in range(len(waypoints) - 1):
            a = waypoints[i]
            b = waypoints[i + 1]
            
            for obs in obstacles:
                dist, closest = self.point_line_distance_2d(obs, a, b)
                if dist < safety_radius:
                    # Add detour perpendicular to path
                    path_dir = b - a
                    path_dir /= np.linalg.norm(path_dir)
                    # Pick a perpendicular direction (heuristic)
                    detour_dir = np.cross(path_dir, np.array([0, 0, 1]))
                    if np.linalg.norm(detour_dir) == 0:
                        detour_dir = np.cross(path_dir, np.array([0, 1, 0]))
                    detour_dir /= np.linalg.norm(detour_dir)
                    detour = closest + detour_dir * (safety_radius + 0.5)
                    adjusted_waypoints.append(detour)
                    #print("detour added")
                    break
            adjusted_waypoints.append(b)
        return adjusted_waypoints
    
    def point_line_distance_2d(self, p, a, b):
            """
            Calculate the distance between point `p` and the line segment `a`-`b` in 2D (x, y only).
            """
            # Project to 2D
            p2d = p[:2]
            a2d = a[:2]
            b2d = b[:2]
            ab = b2d - a2d
            ap = p2d - a2d
            if np.allclose(ab, 0):
                return np.linalg.norm(ap), a  # Segment is a point
            t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
            closest_2d = a2d + t * ab
            # Rebuild 3D point with same z as a (or interpolate if desired)
            closest_3d = np.array([closest_2d[0], closest_2d[1], a[2]])
            return np.linalg.norm(p2d - closest_2d), closest_3d
