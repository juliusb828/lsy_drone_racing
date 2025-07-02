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

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.plotting import Plotting 

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryController(Controller):
    """Trajectory controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        # Same waypoints as in the trajectory controller. Determined by trial and error.

        # nominal gate positions
        self.do_PID = True
        self.do_plot = True
        self.nominal_gates = config["env.track.gates"]
        print(f"gates at init: {self.nominal_gates}")
        self.gate_positions_nominal = np.array([gate['pos'] for gate in self.nominal_gates])
        self.last_gate_positions = np.copy(self.gate_positions_nominal)
        self.waypoints = np.array(
            [
                np.array(obs['pos']), 
                [1.0, 1.5, 0.05],
                [0.8, 1.0, 0.2],
                [0.55, -0.3, 0.5],
                [0.45, -0.5, 0.56], # gate1
                [0.2, -1.3, 0.65],
                [1, -1.05, 1.11], # gate2
                [1.1, -0.85, 1.1],
                [0.2, 0.5, 0.65],
                [0, 1, 0.56], # gate3
                [0.0, 1.2, 0.525], 
                [0.0, 1.2, 1.1], # here it only goes up 
                [-0.5, 0.0, 1.1], # gate4 
                [-0.5, -0.5, 1.1],
            ]
        )
        # waypoints = np.array(
        #     [
        #         #np.array(obs['pos']),
        #         [1.0, 1.5, 0.05],
        #         [0.8, 1.0, 0.2],
        #         [0.55, -0.3, 0.5],
        #         [0.2, -1.3, 0.65],
        #         [1.1, -0.85, 1.1],
        #         [0.2, 0.5, 0.65],
        #         [0.0, 1.2, 0.525],
        #         [0.0, 1.2, 1.1],
        #         [-0.5, 0.0, 1.1],
        #         [-0.5, -0.5, 1.1],
        #     ]
        # )
        self.gate_to_waypoint_idx = {
        0: 4,
        1: 6,
        2: 9,
        3: 12
        }
        
        # Alternative: constructing trajectory from the waypoints 
        #waypoint = np.array(obs["pos"])
        #waypoints = np.vstack([waypoint, gate_positions])

        # some extra waypoints to avoid obstacle 
        # detour = np.array([[0.4, 0.5, 0.5]])
        #waypoints = np.insert(waypoints, 4, detour, axis=0)

        # obstacles 
        self.obstacles_nominal = config["env.track.obstacles"]
        self.obst_positions_nominal = np.array([obst['pos'] for obst in self.obstacles_nominal])
        self.last_obst_positions = np.copy(self.obst_positions_nominal)
        x_obs, y_obs = self.obst_positions_nominal[:, 0], self.obst_positions_nominal[:, 1]

        self.t_total = 11
        t = np.linspace(0, self.t_total, len(self.waypoints))
        #self.trajectory = CubicSpline(t, self.waypoints)
        self.trajectory = make_interp_spline(t, self.waypoints, k=3)
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False
        self._Kp = 4.0
        #self._Kp = 2.3
        #self._Kp = np.array([2.5, 4.5, 2.0])
        #self._Kd = 0.3
        self._Kd = 2.5
        #self._Kd = np.array([0.3, 0.3, 0.2])
        #self._Ki = np.array([0.3, 0.3, 0.3])
        self._Ki = 0
        #self._Ki = np.array([0.1, 0.3, 0.05])
        self._prev_error_pos = np.zeros(3) 
        self._error_integral = np.zeros(3)

        ## Plotting
        if self.do_plot:
            t_fine = np.linspace(0, self.t_total, 200)  # 200 points = nice and smooth
            trajectory_points = self.trajectory(t_fine)  # shape: (200, 3), # Evaluate the spline at these times
            
            self.plotter = Plotting()
            self.plotter.plot_2d_trajectory(waypoints=self.waypoints, trajectory=trajectory_points,projection='xy', config=config)

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
        current_pos = obs["pos"]
        current_vel = obs["vel"]
        gate_positions = obs['gates_pos']
        obst_positions = obs['obstacles_pos']
        gates_visited = obs['gates_visited']
        #print(f"observations: {obs}")

        # For Level2: once the real gate position becomes available, regenerate trajectory (starting point being the current position) 
        # check for changes
        #print(f"Actual Gate positions = {gate_positions}")
        #print(f"Nominal Gate positions = {self.gate_positions_nominal}")

        # check if any gates position has been update w.r.t nominal value 
        changed_indices = np.where(np.any(~np.isclose(self.last_gate_positions, gate_positions, atol=1e-2), axis=1))[0]

        tau = min(self._tick / self._freq, self.t_total)
        if changed_indices.size > 0:

            gate_idx = changed_indices[0]
            print(f"ADAPT, gate {gate_idx} changed!")

            wp_gate_idx = self.gate_to_waypoint_idx[gate_idx]
            actual_gate = gate_positions[gate_idx] # actual gate position 

            # Construct new waypoints:
            new_waypoints = [current_pos] # start at current drone position 
            new_waypoints.append(actual_gate) # the actual gate position as second way point
            new_waypoints.extend(self.waypoints[wp_gate_idx + 1:]) # Add rest of original waypoints
        
            # Update trajectory
            new_waypoints = np.array(new_waypoints)
            t = np.linspace(0, self.t_total, len(new_waypoints))
            self.trajectory = CubicSpline(t, new_waypoints, bc_type=((1, current_vel/np.linalg.norm(current_vel)), 'natural'))
            print(f"Updated waypoints and trajectory.")

            # update self.last_gate_positions (so we update trajectory only once after the change)
            self.last_gate_positions = np.copy(gate_positions)

            # graph:
            if self.do_plot:
                t_fine = np.linspace(0, self.t_total, 200)  # 200 points = nice and smooth
                trajectory_points = self.trajectory(t_fine)  # shape: (200, 3), # Evaluate the spline at these times
                self.plotter.update_2d_trajectory(trajectory=trajectory_points, obs=obs)
                print("Updated Trajectory graphed")
                input("Press Enter to continue")

        # check if obstacles positions changed 
        changed_indices_obs = np.where(np.any(~np.isclose(self.last_obst_positions, obst_positions, atol=1e-2), axis=1))[0]
        if changed_indices_obs.size > 0:
            print("OBSTACLE SHIFT DETECTED!")

            self.last_obst_positions = np.copy(obst_positions)
            if self.do_plot:
                self.plotter.update_2d_trajectory(obs=obs)
                #self.ax.scatter(obst_positions[:,0], obst_positions[:,1], color='black', s=30)

            # check if trajectory isn't too close to obstacles   
            self.check_obstacles(obst_positions, current_pos, current_vel, gates_visited)
            
        # Control logic
        tau = min(self._tick / self._freq, self.t_total)
        target_pos = self.trajectory(tau)
        target_vel_ff = self.trajectory.derivative()(tau) # feedforward velocity

        error_pos = target_pos - current_pos
        self._error_integral += error_pos / self._freq
        print(f"At tau={tau}: error_pos={error_pos}")

        error_vel = target_vel_ff - current_vel

        # Plotting: Every N steps, plot a new dot for the drone's position
        if self.do_plot:
            self.plotter.plot_drone_pos(tick=self._tick, freq=3, obs=obs)
            

        # Derivative term
        derivative = (error_pos - self._prev_error_pos) * self._freq
        #max_derivative = 0.5
        #derivative = np.clip(derivative, -max_derivative, max_derivative)
        self._prev_error_pos = error_pos

        # PID Control
        #target_vel = self._Kp * error_pos + self._Ki * self._error_integral + self._Kd * derivative
        
        # PID Control (PD + feedforward)
        target_vel = target_vel_ff + self._Kp * error_pos + self._Kd * error_vel

        # output saturation:

        if tau == self.t_total:  # Maximum duration reached
            self._finished = True
        if self.do_PID:
            return np.concatenate((target_pos, target_vel, np.zeros(7)), dtype=np.float32)
        else:
            return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)

    def check_obstacles(self, obst_positions, drone_pos, current_vel, gates_visited):
        safe_dist = 0.15
        too_close = False

        t_fine = np.linspace(0, self.t_total, 200)  # 200 points = nice and smooth
        trajectory_points = self.trajectory(t_fine)
        
        for i, obs_pos in enumerate(obst_positions):
            obs_xy = obs_pos[:2] # center of obstacle
            obst_z_top = obs_pos[2] # top of the obstacle 

            for point in trajectory_points:
                drone_xy = point[:2]
                drone_z = point[2]

                horizontal_dist = np.linalg.norm(drone_xy - obs_xy)
                if horizontal_dist < safe_dist and (0 <= drone_z <= obst_z_top):
                    print("Trajectory too close to obstacle!")
                    obst_num = i
                    too_close = True
                    break
            if too_close:
                break
        
        if True: # not too_close:
            print("Trajectory accepted.")
        elif too_close:
            # replan trajectory
            print(f"Replanning trajectory to avoid obstacle {obst_num}...")

            vec = drone_pos[:2] - obst_positions[i,:2]
            vec_r = np.array([vec[1], -vec[0]]) # rotate
            dist = np.linalg.norm(vec_r)
            direction = vec_r/dist

            # compute detour point
            detour_xy = obst_positions[i,:2] + direction * 2*safe_dist
            detour_z = max(drone_pos[2], obst_positions[i,2]+0.2)
            detour_point = np.array([detour_xy[0], detour_xy[1], detour_z])
            
            # Get indices of gates that have been visited
            visited_indices = np.where(gates_visited)[0]
            current_gate_idx = visited_indices[-1] 
            wp_gate_idx = self.gate_to_waypoint_idx[current_gate_idx]
            #actual_gate = gate_positions[gate_idx] # actual gate position 

            # Construct new waypoints:
            new_waypoints = [drone_pos] # start at current drone position 
            new_waypoints.append(detour_point) # the 
            new_waypoints.extend(self.waypoints[wp_gate_idx + 1:]) # Add rest of original waypoints
        
            # Update trajectory
            new_waypoints = np.array(new_waypoints)
            t = np.linspace(0, self.t_total, len(new_waypoints))
            self.trajectory = CubicSpline(t, new_waypoints, bc_type=((1, current_vel/np.linalg.norm(current_vel)), 'natural'))
            print(f"Updated waypoints and trajectory.")

            # graph:
            if self.do_plot:
                # line, = self.ax.plot([], [], color='cyan', label='Updated_Trajectory')
                # print(self.t_total)
                t_fine = np.linspace(0, self.t_total, 200)  # 200 points = nice and smooth
                trajectory_points = self.trajectory(t_fine)  # shape: (200, 3), # Evaluate the spline at these times
                # x_vals = trajectory_points[:, 0] # Extract X and Y coordinates (ignoring Z)
                # y_vals = trajectory_points[:, 1]
                # line.set_data(x_vals, y_vals)
                self.plotter.update_2d_trajectory(trajectory=trajectory_points)
                print("Updated Trajectory graphed")
                input("Press Enter to continue")
     
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
