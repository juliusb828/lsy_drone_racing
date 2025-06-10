from __future__ import annotations  # Python 3.10 type hints  # noqa: D100

import threading
import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import minsnap_trajectories as ms  # type: ignore
import numpy as np

#from scipy.interpolate import BSpline, make_interp_spline
from scipy.spatial.transform import Rotation as R
from skimage.graph import route_through_array  #type: ignore

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

        self.t_total = 12.0  # total time duration
        self.time = 0 # time passed since start

        self.min_x, self.max_x = -2.0, 2.0
        self.min_y, self.max_y = -2.0, 2.0
        self.resolution = 0.02
        self.target_positions = None
        self.target_positions, self.target_velocities, self.target_accelerations = self.compute_trajectory(obs["pos"], obs["gates_pos"], obs["gates_quat"], obs["obstacles_pos"], 0.0, obs["target_gate"], obs["vel"],)

        self._tick = 0
        # frequency is 50 Hz, so 0.02 ms timesteps
        #self._freq = config.env.freq
        self._finished = False
        self.recompute_flag = False

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
        target_gate = obs["target_gate"]
        self.gates_passed = np.zeros(4, dtype=bool) 
        self.gates_passed[:target_gate] = True

        if not np.allclose(obs["obstacles_pos"], self.last_known_obstacles_pos, atol=0.001):
            self.last_known_obstacles_pos = obs["obstacles_pos"]
            self.trigger_async_recomputation(obs, self.time, target_gate)

        if not np.allclose(obs["gates_pos"], self.last_known_gates_pos, atol=0.001):
            self.last_known_gates_pos = obs["gates_pos"]
            self.trigger_async_recomputation(obs, self.time, target_gate)
        
        #print(tau)
        current_target_pos = self.target_positions[self._tick]
        current_target_vel = self.target_velocities[self._tick]
        #current_target_acc = self.target_accelerations[self._tick]
        yaw = np.array([np.arctan2(current_target_vel[1], current_target_vel[0])])

        if self.time == self.t_total:  # Maximum duration reached
            self._finished = True

        #print(self.time)
        #print(f"current pos {obs["pos"]}")
        #print(f"current target pos: {current_target_pos}")
        return np.concatenate((current_target_pos, np.zeros(6), yaw, np.zeros(3)), dtype=np.float32)
        
    def trigger_async_recomputation(self, obs: dict[str, NDArray[np.floating]], t: float, target_gate: int):
        """Trigger asynchronous recomputation of the trajectory.

        Args:
            obs: The current observation of the environment, including positions of obstacles and gates.
            t: The current time in the trajectory.
            target_gate: The index of the target gate.

        Returns:
            None
        """
        # Start computation in background thread
        self.trajectory_computation_thread = threading.Thread(
            target=self.compute_trajectory, 
            args=(obs["pos"], obs["gates_pos"], obs["gates_quat"], obs["obstacles_pos"], self.time, target_gate, obs["vel"])
        )
        self.trajectory_computation_thread.start()

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
        self.time += 0.02
        return self._finished
        """Reset the time step counter."""
        self._tick = 0

    def to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert coordinates to grid indices.

        Args:
            x: The x-coordinate.
            y: The y-coordinate.

        Returns:
            A tuple containing the grid indices (gx, gy).
        """
        gx = int(np.floor((x - self.min_x) / self.resolution))
        gy = int(np.floor((y - self.min_y) / self.resolution))
        return gx, gy

    def to_coord(self, gx: int, gy: int) -> tuple[float, float]:
        """Convert grid indices to coordinates.

        Args:
            gx: The grid index along the x-axis.
            gy: The grid index along the y-axis.

        Returns:
            A tuple containing the coordinates (x, y).
        """
        x = gx * self.resolution + self.min_x
        y = gy * self.resolution + self.min_y
        return x, y
    
    def construct_cost_map(self, obstacles: list[tuple[float, float]], gates: list[dict]) -> np.ndarray:
        """Construct a cost map based on the positions of obstacles and gates.

        Args:
            obstacles: A list of tuples representing the positions of obstacles (x, y).
            gates: A list of dictionaries containing gate information, including position and orientation.

        Returns:
            A 2D numpy array representing the cost map.
        """
        # Calculate map dimensions
        width = int(np.ceil((self.max_x - self.min_x) / self.resolution))
        height = int(np.ceil((self.max_y - self.min_y) / self.resolution))
        cost_map = np.ones((height, width))
        obstacle_size = 22
        gate_sides_height = 10
        
        # Mark obstacles on the cost map
        for i, (ox, oy) in enumerate(obstacles[:, :2]):
            gx, gy = self.to_grid(ox, oy)
            if i==2 or i==3:
                half_side = int((obstacle_size-2)/2)
            elif i == 1:
                half_side = int((obstacle_size+2)/2)
            elif i == 0:
                half_side = int((obstacle_size+4)/2)
            else:
                half_side = int(obstacle_size / 2)  # Adjust size here
            cost_map[gx - half_side:gx + half_side + 1, gy - half_side:gy + half_side + 1] = 1e6

        # Mark sides of gates on the cost map
        for gate in gates:
            gate_g = self.to_grid(gate["pos"][0], gate["pos"][1])
            # gate is 40 cm wide with 10 cm side
            if gate["rpy"][2] == 0.0 or gate["rpy"][2] == 3.14:
                cost_map[gate_g[0]-24:gate_g[0]-10, gate_g[1]-gate_sides_height:gate_g[1]+gate_sides_height] = 1e6
                cost_map[gate_g[0]+10:gate_g[0]+24, gate_g[1]-gate_sides_height:gate_g[1]+gate_sides_height] = 1e6
            else:
                angle = gate["rpy"][2]
                x_offset = int(np.ceil(10*np.abs(np.cos(angle))))
                y_offset = int(np.ceil(10*np.abs(np.sin(angle))))
                #assuming tilted to the right
                cost_map[gate_g[0]-x_offset-14:gate_g[0]-x_offset, gate_g[1]+y_offset-gate_sides_height:gate_g[1]+y_offset+gate_sides_height] = 1e6
                cost_map[gate_g[0]+x_offset:gate_g[0]+x_offset+14, gate_g[1]-y_offset-gate_sides_height:gate_g[1]-y_offset+gate_sides_height] = 1e6

        cost_map = cost_map.astype(np.int32)
        return cost_map

    def create_gates_dict(self, gates_pos: NDArray[np.floating], gates_quat: NDArray[np.floating]) -> list[dict[str, list[float]]]:
        """Create a dictionary representation of gates with positions and orientations.

        Args:
            gates_pos: A numpy array containing the positions of the gates.
            gates_quat: A numpy array containing the orientations of the gates as quaternions.

        Returns:
            A list of dictionaries, each containing the position and orientation (RPY) of a gate.
        """
        gates = []
        for pos, quat in zip(gates_pos, gates_quat):
            # Convert quaternion to RPY
            rpy = R.from_quat(quat).as_euler('xyz', degrees=False)
            gates.append({'pos': pos.tolist(), 'rpy': rpy.tolist()})
        return gates


    def compute_trajectory(
        self,
        pos: NDArray[np.floating],
        gates_pos: NDArray[np.floating],
        gates_quat: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
        tau: float,
        target_gate: int,
        vel: NDArray[np.floating]
    ) -> tuple[ms.PolynomialTrajectory, list[ms.Waypoint]]:
        """Recompute the trajectory based on the current position, gates, and obstacles.

        Args:
            pos: Current position of the drone as a numpy array.
            gates_pos: Positions of the gates as a numpy array.
            gates_quat: Orientations of the gates as a numpy array of quaternions.
            obstacles_pos: Positions of the obstacles as a numpy array.
            tau: Current time in the trajectory, used as starting time.
            target_gate: Index of the target gate.
            vel: the current velocity

        Returns:
            A tuple containing the polynomial trajectory and the list of waypoints.
        """
        start = time.time()
        gates  = self.create_gates_dict(gates_pos, gates_quat)
        cost_map = self.construct_cost_map(obstacles_pos, gates)

        if self.target_positions is None or self.target_positions.size == 0:
            start_pos = pos
            print(f"start pos set to: {start_pos} (current pos)")
        else:
            start_pos = self.target_positions[self._tick+3]
            print(f"start pos set to: {start_pos} (current target pos)")
        

        start_grid = self.to_grid(x=start_pos[0], y=start_pos[1])
        gate1_grid = self.to_grid(x=gates_pos[0][0]+0.15, y=gates_pos[0][1])
        gate1_artifical_pos_grid = self.to_grid(x=gates_pos[0][0]-0.25, y=gates_pos[0][1]-1.2)
        gate2_grid = self.to_grid(x=gates_pos[1][0]+0.1, y=gates_pos[1][1]+0.05)
        pre_gate3_grid = self.to_grid(x=gates_pos[2][0], y=gates_pos[2][1]-0.15)
        gate3_grid = self.to_grid(x=gates_pos[2][0], y=gates_pos[2][1]+0.1)
        gate3_artificial__pos_grid = self.to_grid(x=gates_pos[2][0]-0.8, y=gates_pos[2][1]+0.3)
        gate4_grid = self.to_grid(x=gates_pos[3][0]+0.1, y=gates_pos[3][1]+0.1)
        past_gate4_grid = self.to_grid(x=gates_pos[3][0]+0.15, y=gates_pos[3][1]-0.7)

        path1, _ = route_through_array(
            cost_map, start_grid, gate1_grid,
            fully_connected=True, geometric=True
        )
        path1 = np.array(path1)
        path1_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path1])
        n_1 = path1_world_coords.shape[0]
        z_1 = np.linspace(start_pos[2], gates_pos[0][2], n_1)
        path1_3d = np.column_stack((path1_world_coords, z_1))

        path2, _ = route_through_array(
            cost_map, gate1_grid, gate1_artifical_pos_grid,
            fully_connected=True, geometric=True
        )
        path2_artificial, _ = route_through_array(
            cost_map, gate1_artifical_pos_grid, gate2_grid,
            fully_connected=True, geometric=True
        )
        path2_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path2]), np.array([self.to_coord(gx, gy) for gx, gy in path2_artificial])])
        n_2 = path2_world_coords.shape[0]
        z_2 = np.linspace(gates[0]["pos"][2], gates[1]["pos"][2]+0.12, n_2)
        path2_3d = np.column_stack((path2_world_coords, z_2))

        path3, _ = route_through_array(
            cost_map, gate2_grid, pre_gate3_grid,
            fully_connected=True, geometric=True
        )
        path3_artificial, _ = route_through_array(
            cost_map, pre_gate3_grid, gate3_grid,
            fully_connected=True, geometric=True
        )
        path3_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path3]), np.array([self.to_coord(gx, gy) for gx, gy in path3_artificial])])
        n_3 = path3_world_coords.shape[0]
        z_3 = np.linspace(gates[1]["pos"][2], gates[2]["pos"][2], n_3)
        path3_3d = np.column_stack((path3_world_coords, z_3))

        path4, _ = route_through_array(
            cost_map, gate3_grid, gate3_artificial__pos_grid,
            fully_connected=True, geometric=True
        )
        path4_artificial, _ = route_through_array(
            cost_map, gate3_artificial__pos_grid, gate4_grid,
            fully_connected=True, geometric=True
        )
        path4_artificial_2, _ = route_through_array(
            cost_map, gate4_grid, past_gate4_grid,
            fully_connected=True, geometric=True
        )
        path4_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path4]), np.array([self.to_coord(gx, gy) for gx, gy in path4_artificial]), np.array([self.to_coord(gx, gy) for gx, gy in path4_artificial_2])])
        n_4 = path4_world_coords.shape[0]
        z_4 = np.linspace(gates[2]["pos"][2], gates[3]["pos"][2]+0.3, n_4)
        path4_3d = np.column_stack((path4_world_coords, z_4))

        if target_gate == 0:
            if np.linalg.norm(pos[:2] - gates[0]["pos"][:2]) < 0.25:
                print("within 25 cm of gate, don't include path to gate 1!")
                combined_3d_path = np.concatenate([path2_3d, path3_3d, path4_3d])
            else:
                combined_3d_path = np.concatenate([path1_3d, path2_3d, path3_3d, path4_3d])
        elif target_gate == 1:
            path2_from_pos, _ = route_through_array(
                cost_map, start_grid, gate2_grid,
                fully_connected=True, geometric=True
            )
            path2_from_pos_to_artificial, _ = route_through_array(
                cost_map, start_grid, gate1_artifical_pos_grid,
                fully_connected=True, geometric=True
            )
            path2_artificial_to_gate2, _ = route_through_array(
                cost_map, gate1_artifical_pos_grid, gate2_grid,
                fully_connected=True, geometric=True
            )
            if np.linalg.norm(pos[:2] - gates[0]['pos'][:2]) < 0.3:
                print("using artificial point")
                path2_from_pos_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path2_from_pos_to_artificial]), np.array([self.to_coord(gx, gy) for gx, gy in path2_artificial_to_gate2])])
            else:
                path2_from_pos_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path2_from_pos])
            n_2 = path2_from_pos_world_coords.shape[0]
            z_2 = np.linspace(start_pos[2], gates[1]["pos"][2]+0.12, n_2)
            path2_from_pos_3d = np.column_stack((path2_from_pos_world_coords, z_2))
            combined_3d_path = np.concatenate([path2_from_pos_3d, path3_3d, path4_3d])
        elif target_gate == 2:
            path3_from_pos, _ = route_through_array(
                cost_map, start_grid, pre_gate3_grid,
                fully_connected=True, geometric=True
            )
            path3_from_pos_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path3_from_pos])
            n_3 = path3_from_pos_world_coords.shape[0]
            z_3 = np.linspace(start_pos[2], gates[2]["pos"][2], n_3)
            path3_from_pos_3d = np.column_stack((path3_from_pos_world_coords, z_3))
            #test: remove artificial point from path 4
            end_of_path3, _ = route_through_array(
                cost_map, pre_gate3_grid, gate3_grid,
                fully_connected=True, geometric=True
            )
            path4, _ = route_through_array(
                cost_map, gate3_grid, gate3_artificial__pos_grid,
                fully_connected=True, geometric=True
            )
            path4_artificial, _ = route_through_array(
                cost_map, gate3_artificial__pos_grid, gate4_grid,
                fully_connected=True, geometric=True
            )
            path4_artificial_2, _ = route_through_array(
                cost_map, gate4_grid, past_gate4_grid,
                fully_connected=True, geometric=True
            )
            path4_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in end_of_path3]), np.array([self.to_coord(gx, gy) for gx, gy in path4]), np.array([self.to_coord(gx, gy) for gx, gy in path4_artificial]), np.array([self.to_coord(gx, gy) for gx, gy in path4_artificial_2])])
            n_4 = path4_world_coords.shape[0]
            z_4 = np.linspace(gates[2]["pos"][2], gates[3]["pos"][2]+0.3, n_4)
            path4_3d = np.column_stack((path4_world_coords, z_4))
            if np.linalg.norm(pos[:2] - gates[2]['pos'][:2]) < 0.5:
                print("no pre gate 3 point!")
                path4, _ = route_through_array(
                    cost_map, start_grid, gate3_artificial__pos_grid,
                    fully_connected=True, geometric=True
                )
                path4_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path4]), np.array([self.to_coord(gx, gy) for gx, gy in path4_artificial]), np.array([self.to_coord(gx, gy) for gx, gy in path4_artificial_2])])
                n_4 = path4_world_coords.shape[0]
                z_4 = np.linspace(gates[2]["pos"][2], gates[3]["pos"][2]+0.3, n_4)
                path4_3d = np.column_stack((path4_world_coords, z_4))
                combined_3d_path = np.concatenate([path3_from_pos_3d, path4_3d])
            else:
                combined_3d_path = np.concatenate([path3_from_pos_3d, path4_3d])
        elif target_gate == 3:
            print("===================================")
            print("TARGET GATE 3")
            path4_start_to_artifical_point_left_of_gate3, _ = route_through_array(
                cost_map, start_grid, gate3_artificial__pos_grid,
                fully_connected=True, geometric=True
            )
            path4_artificial_to_gate4, _ = route_through_array(
                cost_map, gate3_artificial__pos_grid, gate4_grid,
                fully_connected=True, geometric=True
            )
            path4_from_pos, _ = route_through_array(
                cost_map, start_grid, gate4_grid,
                fully_connected=True, geometric=True
            )
            path4_to_past_grid, _ = route_through_array(
                cost_map, gate4_grid, past_gate4_grid,
                fully_connected=True, geometric=True
            )
            print(pos[:2])
            print(gates[3]["pos"][:2])
            if np.linalg.norm(pos[:2] - gates[3]['pos'][:2]) < 1.60:
                print("within 60 cm of final gate!")
                path4_from_pos_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path4_from_pos]), np.array([self.to_coord(gx, gy) for gx, gy in path4_to_past_grid])])
            else:
                path4_from_pos_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path4_start_to_artifical_point_left_of_gate3]), np.array([self.to_coord(gx, gy) for gx, gy in path4_artificial_to_gate4]), np.array([self.to_coord(gx, gy) for gx, gy in path4_to_past_grid])])
            n_4 = path4_from_pos_world_coords.shape[0]
            z_4 = np.linspace(start_pos[2], gates[3]["pos"][2], n_4)
            path4_from_pos_3d = np.column_stack((path4_from_pos_world_coords, z_4))
            combined_3d_path = np.concatenate([path4_from_pos_3d])

        num_combined_points = combined_3d_path.shape[0]
        reduction_rate = int(num_combined_points/20)
        print(f"reduction_rate: {reduction_rate}")
        reduced_3d_path = combined_3d_path[::reduction_rate]
        print(reduced_3d_path.shape)
        num_points = reduced_3d_path.shape[0]
        time_steps = np.linspace(tau, self.t_total, num_points)


        """
        test with BSplines: works for level1, but trajectory is not continous -> level 2 fails
        positions = np.array(reduced_3d_path)  # shape: (N, 3)
        time_steps = np.array(time_steps)      # shape: (N,)
        sampling_rate = 50
        time_samples = np.linspace(tau, self.t_total, int((self.t_total - tau) * sampling_rate))

        # One spline per axis (x, y, z), with smoothing interpolant
        spl_x = make_interp_spline(time_steps, positions[:, 0], k=3)  # k=3 = cubic spline
        spl_y = make_interp_spline(time_steps, positions[:, 1], k=3)
        spl_z = make_interp_spline(time_steps, positions[:, 2], k=3)

        # Evaluate position
        target_pos = np.stack([
            spl_x(time_samples),
            spl_y(time_samples),
            spl_z(time_samples)
        ], axis=1)  # shape: (M, 3)

        # Evaluate velocity (1st derivative)
        target_vel = np.stack([
            spl_x.derivative(1)(time_samples),
            spl_y.derivative(1)(time_samples),
            spl_z.derivative(1)(time_samples)
        ], axis=1)

        # Evaluate acceleration (2nd derivative)
        target_acc = np.stack([
            spl_x.derivative(2)(time_samples),
            spl_y.derivative(2)(time_samples),
            spl_z.derivative(2)(time_samples)
        ], axis=1)

        # Store results
        self.target_positions = target_pos
        self.target_velocities = target_vel
        self.target_accelerations = target_acc
        end = time.time()
        print("trajectory updated!")
        print("Execution time:", (end - start) * 1e3, "ms")
        """
        
        waypoints = [
            ms.Waypoint(
            time=t,
            position=np.array([pos[0], pos[1], pos[2]]),
            velocity=(
            None if np.linalg.norm(pos[:2] - gates[2]["pos"][:2]) < 0.25 or np.linalg.norm(pos[:2] - gates[1]["pos"][:2]) < 0.25
            else np.array([vel[0]*0.5, vel[1]*0.5, vel[2]])
            ) if t == tau else None  # Constrain velocity for the first waypoint unless close to gate[2]
            )
            for t, pos in zip(time_steps, reduced_3d_path)
        ]


        polys = ms.generate_trajectory(
            waypoints,
            degree=8,  # Polynomial degree
            idx_minimized_orders=(3, 4),
            num_continuous_orders=3,
            algorithm="closed-form",
        )
        sampling_rate = 50  # Adjust sampling rate as needed
        time_samples = np.linspace(tau, self.t_total, int((self.t_total-tau) * sampling_rate))
        pva = ms.compute_trajectory_derivatives(polys, time_samples, 3)
        target_pos = pva[0, ...]
        target_vel = pva[1, ...]
        target_acc = pva[2, ...]
        self.target_positions, self.target_velocities, self.target_accelerations = target_pos, target_vel, target_acc
        self._tick = 0
        end = time.time()
        print("trajectory updated!")
        print("Execution time:", (end - start) * 1e3, "ms")
        
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Create two side-by-side subplots
        # First plot: Trajectory and reduced path points
        x = target_pos[:, 0]
        y = target_pos[:, 1]
        reduced_x = reduced_3d_path[:, 0]
        reduced_y = reduced_3d_path[:, 1]
        ax1.plot(x, y, label='Trajectory', color='blue', linewidth=2)
        ax1.scatter(reduced_x, reduced_y, label='Reduced Path Points', color='red', s=20)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid()
        ax1.set_title('Trajectory Plot')

        # Second plot: Cost map and path
        ax2.imshow(cost_map.T, cmap='gray_r', origin='lower')  # Transpose cost_map for correct orientation
        path = np.array(reduced_3d_path[:, :2])  # Ensure path is a numpy array
        grid_coords = np.array([self.to_grid(x, y) for x, y in path])
        rows, cols = grid_coords[:, 0], grid_coords[:, 1]
        ax2.plot(rows, cols, color='red', linewidth=2)
        ax2.set_title('Cost Map with Path')

        # Save and show the combined plot
        plt.tight_layout()
        plt.savefig("trajectory_and_cost_map_changed.png")
        plt.close()
        
        return target_pos, target_vel, target_acc
