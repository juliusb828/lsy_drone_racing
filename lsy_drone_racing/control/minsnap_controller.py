from __future__ import annotations  # Python 3.10 type hints  # noqa: D100

import threading
import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import minsnap_trajectories as ms  # type: ignore
import numpy as np
from scipy.interpolate import CubicSpline

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

        self.t_total = 14.0  # total time duration
        self.time = 0 # time passed since start
        self.plot = False

        self.min_x, self.max_x = -2.0, 2.0
        self.min_y, self.max_y = -2.0, 2.0
        self.resolution = 0.02
        self.target_positions = None
        self.target_velocities = None
        
        self.prev_pos_error = None
        self._tick = 0
        # frequency is 50 Hz, so 0.02 ms timesteps
        self._freq = config.env.freq
        self._finished = False
        self.recompute_flag = False

        self.errors = []
        self.step_count = 0

        #setup once, don't recalculate this every time

        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        gates  = self.create_gates_dict(gates_pos, gates_quat)
        cost_map = self.construct_cost_map(obs["obstacles_pos"], gates)
        pre_gate_grids = []
        pos_gate_grids = []
        for i, (gate_pos, gate_quat) in enumerate(zip(gates_pos, gates_quat)):
            rot = R.from_quat(gate_quat).as_matrix()
            fwd = rot[:, 1]
            pre_gate_grid = self.to_grid(x=gate_pos[0] - 0.2 * fwd[0], y=gate_pos[1] - 0.2 * fwd[1])
            pos_gate_grid = self.to_grid(x=gate_pos[0] + 0.2 * fwd[0], y=gate_pos[1] + 0.2 * fwd[1])
            pre_gate_grids.append(pre_gate_grid)
            pos_gate_grids.append(pos_gate_grid)
            
        self.pre_gate_1_grid, self.pre_gate_2_grid, self.pre_gate_3_grid, self.pre_gate_4_grid = pre_gate_grids
        self.past_gate_1_grid, self.past_gate_2_grid, self.past_gate_3_grid, self.past_gate_4_grid = pos_gate_grids
        self.artifical_pos_1_grid = self.to_grid(x=gates_pos[0][0]-0.35, y=gates_pos[0][1]-0.95)
        self.artificial_pos_2_grid = self.to_grid(x=gates_pos[2][0]-0.8, y=gates_pos[2][1]+0.3)
        self.past_gate_4_grid = self.to_grid(x=gates_pos[3][0]+0.15, y=gates_pos[3][1]-0.7)

        start_grid = self.to_grid(x=obs["pos"][0], y=obs["pos"][1])
        
        path_start_to_pre_gate1, _ = route_through_array(
            cost_map, start_grid, self.pre_gate_1_grid,
            fully_connected=True, geometric=True
        )
        path_start_to_gate1 = np.array(path_start_to_pre_gate1)
        path_start_to_pre_gate1_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_gate1])
        n1 = path_start_to_pre_gate1_world_coords.shape[0]
        z1 = np.linspace(obs["pos"][2], gates_pos[0][2], n1)
        self.path_start_to_pre_gate1_3d = np.column_stack((path_start_to_pre_gate1_world_coords, z1))

        # Path from pre-gate 1 to past gate 1
        path_pre_gate1_to_past_gate1, _ = route_through_array(
            cost_map, self.pre_gate_1_grid, self.past_gate_1_grid,
            fully_connected=True, geometric=True
        )
        path_pre_gate1_to_past_gate1_array = np.array(path_pre_gate1_to_past_gate1)
        path_pre_gate1_to_past_gate1_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_pre_gate1_to_past_gate1_array])
        n1 = path_pre_gate1_to_past_gate1_world_coords.shape[0]
        z1 = np.linspace(gates_pos[0][2], gates_pos[0][2], n1)
        self.path_pre_gate1_to_past_gate1_3d = np.column_stack((path_pre_gate1_to_past_gate1_world_coords, z1))

        # Path from past gate 1 to artificial point 1
        path_past_gate1_to_artificial_point1, _ = route_through_array(
            cost_map, self.past_gate_1_grid, self.artifical_pos_1_grid,
            fully_connected=True, geometric=True
        )
        path_past_gate1_to_artificial_point1_array = np.array(path_past_gate1_to_artificial_point1)
        # Path from artificial point 1 to pre-gate 2
        path_artificial_point1_to_pre_gate2, _ = route_through_array(
            cost_map, self.artifical_pos_1_grid, self.pre_gate_2_grid,
            fully_connected=True, geometric=True
        )
        path_artificial_point1_to_pre_gate2_array = np.array(path_artificial_point1_to_pre_gate2)
        path_artificial_point1_to_pre_gate2_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path_past_gate1_to_artificial_point1_array]), np.array([self.to_coord(gx, gy) for gx, gy in path_artificial_point1_to_pre_gate2_array])])
        n1 = path_artificial_point1_to_pre_gate2_world_coords.shape[0]
        z1 = np.linspace(gates_pos[0][2], gates_pos[1][2], n1)
        self.path_past_gate1_3d_to_pre_gate2_3d = np.column_stack((path_artificial_point1_to_pre_gate2_world_coords, z1))

        # Path from pre-gate 2 to past gate 2
        path_pre_gate2_to_past_gate2, _ = route_through_array(
            cost_map, self.pre_gate_2_grid, self.past_gate_2_grid,
            fully_connected=True, geometric=True
        )
        path_pre_gate2_to_past_gate2_array = np.array(path_pre_gate2_to_past_gate2)
        path_pre_gate2_to_past_gate2_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_pre_gate2_to_past_gate2_array])
        n2 = path_pre_gate2_to_past_gate2_world_coords.shape[0]
        z2 = np.linspace(gates_pos[1][2], gates_pos[1][2], n2)
        self.path_pre_gate2_to_past_gate2_3d = np.column_stack((path_pre_gate2_to_past_gate2_world_coords, z2))

        # Path from past gate 2 to pre-gate 3
        path_past_gate2_to_pre_gate3, _ = route_through_array(
            cost_map, self.past_gate_2_grid, self.pre_gate_3_grid,
            fully_connected=True, geometric=True
        )
        path_past_gate2_to_pre_gate3_array = np.array(path_past_gate2_to_pre_gate3)
        path_past_gate2_to_pre_gate3_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_past_gate2_to_pre_gate3_array])
        n2_to_3 = path_past_gate2_to_pre_gate3_world_coords.shape[0]
        z2_to_3 = np.linspace(gates_pos[1][2], gates_pos[2][2], n2_to_3)
        self.path_past_gate2_to_pre_gate3_3d = np.column_stack((path_past_gate2_to_pre_gate3_world_coords, z2_to_3))

        # Path from pre-gate 3 to past gate 3
        path_pre_gate3_to_past_gate3, _ = route_through_array(
            cost_map, self.pre_gate_3_grid, self.past_gate_3_grid,
            fully_connected=True, geometric=True
        )
        path_pre_gate3_to_past_gate3_array = np.array(path_pre_gate3_to_past_gate3)
        path_pre_gate3_to_past_gate3_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_pre_gate3_to_past_gate3_array])
        n3 = path_pre_gate3_to_past_gate3_world_coords.shape[0]
        z3 = np.linspace(gates_pos[2][2], gates_pos[2][2], n3)
        self.path_pre_gate3_to_past_gate3_3d = np.column_stack((path_pre_gate3_to_past_gate3_world_coords, z3))

        path_past_gate3_to_artificial_point2, _ = route_through_array(
            cost_map, self.past_gate_3_grid, self.artificial_pos_2_grid,
            fully_connected=True, geometric=True
        )
        path_past_gate3_to_artificial_point3_array = np.array(path_past_gate3_to_artificial_point2)
        # Path from artificial point 1 to pre-gate 2
        path_artificial_point2_to_pre_gate4, _ = route_through_array(
            cost_map, self.artificial_pos_2_grid, self.pre_gate_4_grid,
            fully_connected=True, geometric=True
        )
        self.path_artificial_point2_to_pre_gate4_array = np.array(path_artificial_point2_to_pre_gate4)
        path_artificial_point2_to_pre_gate4_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path_past_gate3_to_artificial_point3_array]), np.array([self.to_coord(gx, gy) for gx, gy in self.path_artificial_point2_to_pre_gate4_array])])
        n1 = path_artificial_point2_to_pre_gate4_world_coords.shape[0]
        z1 = np.linspace(gates_pos[0][2], gates_pos[1][2], n1)
        self.path_past_gate3_to_pre_gate4_3d = np.column_stack((path_artificial_point2_to_pre_gate4_world_coords, z1))

        # Path from pre-gate 4 to past gate 4
        path_pre_gate4_to_past_gate4, _ = route_through_array(
            cost_map, self.pre_gate_4_grid, self.past_gate_4_grid,
            fully_connected=True, geometric=True
        )
        path_pre_gate4_to_past_gate4_array = np.array(path_pre_gate4_to_past_gate4)
        path_pre_gate4_to_past_gate4_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_pre_gate4_to_past_gate4_array])
        n4 = path_pre_gate4_to_past_gate4_world_coords.shape[0]
        z4 = np.linspace(gates_pos[3][2], gates_pos[3][2], n4)
        self.path_pre_gate4_to_past_gate4_3d = np.column_stack((path_pre_gate4_to_past_gate4_world_coords, z4))

        # compute trajectory last, doesn't work if paths aren't calculated 
        self.target_positions, self.target_velocities, self.target_accelerations = self.compute_trajectory(obs["pos"], obs["gates_pos"], obs["gates_quat"], obs["obstacles_pos"], 0.0, obs["target_gate"], obs["vel"], False, False)


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
            distances = np.linalg.norm(obs["obstacles_pos"] - self.last_known_obstacles_pos, axis=-1)
            max_distance = np.max(distances)
            self.last_known_obstacles_pos = obs["obstacles_pos"]
            if max_distance > 0.08:
                print("recomputing trajectory")
                self.trigger_async_recomputation(obs, self.time, target_gate, True, False)
            else:
                print("no recomputation necessary")

        if not np.allclose(obs["gates_pos"], self.last_known_gates_pos, atol=0.001):
            distances = np.linalg.norm(obs["gates_pos"] - self.last_known_gates_pos, axis=-1)
            max_distance = np.max(distances)
            self.last_known_gates_pos = obs["gates_pos"]
            if max_distance > 0.08:
                print("recomputing trajectory")
                self.trigger_async_recomputation(obs, self.time, target_gate, False, True)
            else:
                print("no recomputation necessary")
        
        #print(tau)
        current_target_pos = self.target_positions[self._tick]
        current_target_vel = self.target_velocities[self._tick]
        #current_target_acc = self.target_accelerations[self._tick]
        yaw = np.array([np.arctan2(current_target_vel[1], current_target_vel[0])])

        if self.time == self.t_total:  # Maximum duration reached
            self._finished = True

        error_3d = np.linalg.norm(current_target_pos - obs["pos"])
        self.errors.append(error_3d)
        self.step_count += 1
        print(f"average error {sum(self.errors)/self.step_count}")

        return np.concatenate((current_target_pos, current_target_vel, np.zeros(3), yaw, np.zeros(3)), dtype=np.float32)
        
    def trigger_async_recomputation(self, obs: dict[str, NDArray[np.floating]], t: float, target_gate: int, obsDetected: bool, gateDetected: bool):
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
            args=(obs["pos"], obs["gates_pos"], obs["gates_quat"], obs["obstacles_pos"], self.time, target_gate, obs["vel"], obsDetected, gateDetected)
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
        obstacle_size = 20
        gate_sides_height = 6
        
        # Mark obstacles on the cost map
        for i, (ox, oy) in enumerate(obstacles[:, :2]):
            gx, gy = self.to_grid(ox, oy)
            half_side = int(obstacle_size / 2)  # Adjust size here
            cost_map[gx - half_side:gx + half_side + 1, gy - half_side:gy + half_side + 1] = 1e6

        # Mark sides of gates on the cost map
        for gate in gates:
            gate_g = self.to_grid(gate["pos"][0], gate["pos"][1])
            # gate is 40 cm wide with 10 cm side
            if gate["rpy"][2] == 0.0 or gate["rpy"][2] == 3.14:
                cost_map[gate_g[0]-24:gate_g[0]-8, gate_g[1]-gate_sides_height:gate_g[1]+gate_sides_height] = 1e6
                cost_map[gate_g[0]+8:gate_g[0]+24, gate_g[1]-gate_sides_height:gate_g[1]+gate_sides_height] = 1e6
            else:
                angle = gate["rpy"][2]
                x_offset = int(np.ceil(8*np.abs(np.cos(angle))))
                y_offset = int(np.ceil(8*np.abs(np.sin(angle))))
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
        vel: NDArray[np.floating],
        obsDetected: bool,
        gateDetected: bool
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
        else:
            start_pos = self.target_positions[self._tick]

        start_grid = self.to_grid(x=start_pos[0], y=start_pos[1])
        
        path_start_to_pre_gate1, _ = route_through_array(
            cost_map, start_grid, self.pre_gate_1_grid,
            fully_connected=True, geometric=True
        )
        path_start_to_gate1 = np.array(path_start_to_pre_gate1)
        path_start_to_pre_gate1_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_gate1])
        n1 = path_start_to_pre_gate1_world_coords.shape[0]
        z1 = np.linspace(start_pos[2], gates_pos[0][2], n1)
        self.path_start_to_pre_gate1_3d = np.column_stack((path_start_to_pre_gate1_world_coords, z1))

        if target_gate == 0 and not gateDetected and not obsDetected:
            print("trajectory calculation pre gate 1")
            combined_3d_path = np.concatenate([
                self.path_start_to_pre_gate1_3d, self.path_pre_gate1_to_past_gate1_3d, self.path_past_gate1_3d_to_pre_gate2_3d,
                self.path_pre_gate2_to_past_gate2_3d, self.path_past_gate2_to_pre_gate3_3d, self.path_pre_gate3_to_past_gate3_3d,
                self.path_past_gate3_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
            ])
        elif target_gate == 0 and gateDetected:
            print("trajectory calculation at gate 1 (detected gate 1)")
            path_start_to_past_gate1, _ = route_through_array(
                cost_map, start_grid, self.past_gate_1_grid,
                fully_connected=True, geometric=True
            )
            path_start_to_past_gate1_array = np.array(path_start_to_past_gate1)
            path_start_to_past_gate1_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_past_gate1_array])
            n1 = path_start_to_past_gate1_world_coords.shape[0]
            z1 = np.linspace(start_pos[2], gates_pos[0][2], n1)
            path_start_to_past_gate1_3d = np.column_stack((path_start_to_past_gate1_world_coords, z1))
            combined_3d_path = np.concatenate([
                path_start_to_past_gate1_3d, self.path_past_gate1_3d_to_pre_gate2_3d,
                self.path_pre_gate2_to_past_gate2_3d, self.path_past_gate2_to_pre_gate3_3d, self.path_pre_gate3_to_past_gate3_3d,
                self.path_past_gate3_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
            ])
        elif (obsDetected and target_gate == 0) or (obsDetected and target_gate == 1):
            # check to make sure it isn't because of the first obstacle
            if np.linalg.norm(pos[:2] - obstacles_pos[1][:2]) < 0.5:
                print("trajectory calculation because of second obstacle")
                path_start_to_artificial_point1, _ = route_through_array(
                    cost_map, start_grid, self.artifical_pos_1_grid,
                    fully_connected=True, geometric=True
                )
                path_start_to_artificial_point1_array = np.array(path_start_to_artificial_point1)
                path_artificial_point1_to_pre_gate2, _ = route_through_array(
                    cost_map, self.artifical_pos_1_grid, self.pre_gate_2_grid,
                    fully_connected=True, geometric=True
                )
                path_artificial_point1_to_pre_gate2_array = np.array(path_artificial_point1_to_pre_gate2)
                path_start_to_pre_gate2_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_artificial_point1_array]), np.array([self.to_coord(gx, gy) for gx, gy in path_artificial_point1_to_pre_gate2_array])])
                n1 = path_start_to_pre_gate2_world_coords.shape[0]
                z1 = np.linspace(start_pos[2], gates_pos[1][2], n1)
                path_start_to_pre_gate2_3d = np.column_stack((path_start_to_pre_gate2_world_coords, z1))
                combined_3d_path = np.concatenate([
                    path_start_to_pre_gate2_3d, self.path_pre_gate2_to_past_gate2_3d, self.path_past_gate2_to_pre_gate3_3d, 
                    self.path_pre_gate3_to_past_gate3_3d, self.path_past_gate3_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
                ])
            else:
                #first obstacle
                combined_3d_path = np.concatenate([
                    self.path_start_to_pre_gate1_3d, self.path_pre_gate1_to_past_gate1_3d, self.path_past_gate1_3d_to_pre_gate2_3d,
                    self.path_pre_gate2_to_past_gate2_3d, self.path_past_gate2_to_pre_gate3_3d, self.path_pre_gate3_to_past_gate3_3d,
                    self.path_past_gate3_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
                ])
        elif target_gate == 1 and gateDetected:
            print("trajectory calculation because of second gate")
            path_start_to_past_gate2, _ = route_through_array(
                cost_map, start_grid, self.past_gate_2_grid,
                fully_connected=True, geometric=True
            )
            path_start_to_past_gate2_array = np.array(path_start_to_past_gate2)
            path_start_to_past_gate2_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_past_gate2_array])
            n1 = path_start_to_past_gate2_world_coords.shape[0]
            z1 = np.linspace(start_pos[2], gates_pos[1][2], n1)
            path_start_to_past_gate2_3d = np.column_stack((path_start_to_past_gate2_world_coords, z1))
            combined_3d_path = np.concatenate([
                path_start_to_past_gate2_3d, self.path_past_gate2_to_pre_gate3_3d, self.path_pre_gate3_to_past_gate3_3d,
                self.path_past_gate3_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
            ])
        elif target_gate == 2 and gateDetected:
            print("trajectory calculation because of third gate")
            path_start_to_past_gate3, _ = route_through_array(
                cost_map, start_grid, self.past_gate_3_grid,
                fully_connected=True, geometric=True
            )
            path_start_to_past_gate3_array = np.array(path_start_to_past_gate3)
            path_start_to_past_gate3_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_past_gate3_array])
            n1 = path_start_to_past_gate3_world_coords.shape[0]
            z1 = np.linspace(start_pos[2], gates_pos[2][2], n1)
            path_start_to_past_gate3_3d = np.column_stack((path_start_to_past_gate3_world_coords, z1))
            combined_3d_path = np.concatenate([
                path_start_to_past_gate3_3d, self.path_past_gate3_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
            ])
        elif (obsDetected and target_gate == 2):
            path_start_to_artificial_point2, _ = route_through_array(
                cost_map, start_grid, self.artificial_pos_2_grid,
                fully_connected=True, geometric=True
            )
            path_start_to_artificial_point2_array = np.array(path_start_to_artificial_point2)
            path_start_to_pre_gate_4_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_artificial_point2_array]), np.array([self.to_coord(gx, gy) for gx, gy in self.path_artificial_point2_to_pre_gate4_array])])
            n1 = path_start_to_pre_gate_4_world_coords.shape[0]
            z1 = np.linspace(start_pos[2], gates_pos[3][2], n1)
            path_start_to_pre_gate4_3d = np.column_stack((path_start_to_pre_gate_4_world_coords, z1))
            combined_3d_path = np.concatenate([
                path_start_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
            ])
        elif obsDetected and target_gate == 3:
            #two cases: either obstacle 3 or 4, if obs 3 go to artificial point, if obs 4 -> don't use artificial point
            if np.linalg.norm(pos[:2] - obstacles_pos[3][:2]) < 0.5:
                path_start_to_pre_gate4, _ = route_through_array(
                    cost_map, start_grid, self.pre_gate_4_grid,
                    fully_connected=True, geometric=True
                )
                path_start_to_pre_gate_4_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_pre_gate4])
                n1 = path_start_to_pre_gate_4_world_coords.shape[0]
                z1 = np.linspace(start_pos[2], gates_pos[3][2], n1)
                path_start_to_pre_gate4_3d = np.column_stack((path_start_to_pre_gate_4_world_coords, z1))
                combined_3d_path = np.concatenate([
                    path_start_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
                ])
            else:
                path_start_to_artificial_point2, _ = route_through_array(
                    cost_map, start_grid, self.artificial_pos_2_grid,
                    fully_connected=True, geometric=True
                )
                path_start_to_artificial_point2_array = np.array(path_start_to_artificial_point2)
                path_start_to_pre_gate_4_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_artificial_point2_array]), np.array([self.to_coord(gx, gy) for gx, gy in self.path_artificial_point2_to_pre_gate4_array])])
                n1 = path_start_to_pre_gate_4_world_coords.shape[0]
                z1 = np.linspace(start_pos[2], gates_pos[3][2], n1)
                path_start_to_pre_gate4_3d = np.column_stack((path_start_to_pre_gate_4_world_coords, z1))
                combined_3d_path = np.concatenate([
                    path_start_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
                ])
        elif target_gate == 3 and gateDetected:
            path_start_to_pre_gate4, _ = route_through_array(
                cost_map, start_grid, self.pre_gate_4_grid,
                fully_connected=True, geometric=True
            )
            path_start_to_pre_gate_4_world_coords = np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_pre_gate4])
            n1 = path_start_to_pre_gate_4_world_coords.shape[0]
            z1 = np.linspace(start_pos[2], gates_pos[3][2], n1)
            path_start_to_pre_gate4_3d = np.column_stack((path_start_to_pre_gate_4_world_coords, z1))
            combined_3d_path = np.concatenate([
                path_start_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
            ])
        
        reduced_3d_path = combined_3d_path[::25]
        num_points = reduced_3d_path.shape[0]
        time_steps = np.linspace(tau, self.t_total, num_points)
        
        if self.target_velocities is not None:
            target_velocity_at_tau = self.target_velocities[self._tick]
        else:
            target_velocity_at_tau = np.array([vel[0], vel[1], vel[2]])
        waypoints = []
        for t, pos in zip(time_steps, reduced_3d_path):
            location = np.array([pos[0], pos[1], pos[2]])
            if t == tau:
                velocity = np.array([target_velocity_at_tau[0], target_velocity_at_tau[1], target_velocity_at_tau[2]])
            else:
                velocity = None
            waypoint = ms.Waypoint(
            time=t,
            position=location,
            velocity=velocity
            )
            waypoints.append(waypoint)


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
        
        
        if self.plot:
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

            # Plot additional points on the cost map
            points_to_plot = [
                self.pre_gate_1_grid, self.pre_gate_2_grid, self.pre_gate_3_grid, self.pre_gate_4_grid,
                self.past_gate_1_grid, self.past_gate_2_grid, self.past_gate_3_grid, self.past_gate_4_grid,
                self.artifical_pos_1_grid, self.artificial_pos_2_grid
            ]
            point_labels = [
                "Pre Gate 1", "Pre Gate 2", "Pre Gate 3", "Pre Gate 4",
                "Past Gate 1", "Past Gate 2", "Past Gate 3", "Past Gate 4",
                "Artificial Pos 1", "Artificial Pos 2"
            ]
            for (gx, gy), label in zip(points_to_plot, point_labels):
                ax2.scatter(gx, gy, label=label, s=50)  # Plot each point
            ax2.legend()

            # Save and show the combined plot
            plt.tight_layout()
            plt.savefig("trajectory_and_cost_map_changed.png")
            plt.close()

        return self.target_positions, self.target_velocities, self.target_accelerations
