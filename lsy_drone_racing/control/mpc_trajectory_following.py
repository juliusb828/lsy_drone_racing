"""This module implements MPC trajectory tracking on a dynamically generated path to avoid obstacles and pass thorugh gates.

It is based on the example MPC code in attitude_mpc.py and uses the same MPC settings to track a trajectory. It utilizes
the collective thrust interface for drone control to compute control commands based on current state observations and
desired waypoints. The waypoints are calculated at the start of the program, and recalculated based on the locations of
the gates and obstacles. The code works for Levels 0-2, but struggles with reliability at high speeds in Level 2.
"""

from __future__ import annotations  # Python 3.10 type hints

import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver  # type: ignore
from casadi import MX, cos, sin, vertcat  # type: ignore
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation as R
from skimage.graph import route_through_array  # type: ignore

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# code taken from attitude_mpc.py
def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "lsy_example_mpc"

    # Define Gravitational Acceleration
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    """Model setting"""
    # define basic variables in state and input vector
    px = MX.sym("px")  # 0
    py = MX.sym("py")  # 1
    pz = MX.sym("pz")  # 2
    vx = MX.sym("vx")  # 3
    vy = MX.sym("vy")  # 4
    vz = MX.sym("vz")  # 5
    roll = MX.sym("r")  # 6
    pitch = MX.sym("p")  # 7
    yaw = MX.sym("y")  # 8
    f_collective = MX.sym("f_collective")

    f_collective_cmd = MX.sym("f_collective_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")

    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")

    # define state and input vector
    states = vertcat(
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        roll,
        pitch,
        yaw,
        f_collective,
        f_collective_cmd,
        r_cmd,
        p_cmd,
        y_cmd,
    )
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd)

    # Define nonlinear system dynamics
    f = vertcat(
        vx,
        vy,
        vz,
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)),
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)),
        (params_acc[0] * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY,
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd,
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd,
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd,
        10.0 * (f_collective_cmd - f_collective),
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd,
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs

    return model


# code taken from attitude_mpc.py
def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model

    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    Q = np.diag(
        [
            10.0,
            10.0,
            15.0,  # Position
            0.01,
            0.01,
            0.01,  # Velocity
            0.1,
            0.1,
            0.1,  # rpy
            0.01,
            0.01,  # f_collective, f_collective_cmd
            0.01,
            0.01,
            0.01,
        ]
    )  # rpy_cmd

    R = np.diag([0.01, 0.01, 0.01, 0.01])

    Q_e = Q.copy()

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[:nx, :] = np.eye(nx)  # Only select position states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)  # Only select position states
    ocp.cost.Vx_e = Vx_e

    ocp.cost.yref = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    ocp.cost.yref_e = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0]
    )

    # Set State Constraints
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 40
    ocp.solver_options.nlp_solver_max_iter = 100

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc.json", verbose=verbose)

    return acados_ocp_solver, ocp


class MPController(Controller):
    """MPC trajectory tracking controller using the collective thrust and attitude interface and a dynamically recomputed trajectory."""

    # parts of this code also taken from attitude_mpc.py
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        # frequency is 50 Hz, so 0.02 ms timesteps
        self._tick = 0
        self.des_completion_time = 7.8
        self.enable_smoothing = False
        self.transition_length = 5
        self.slowdown_factor = 1.5
        self.N = 40
        self.T_HORIZON = 2.0
        self.dt = self.T_HORIZON / self.N
        self.prev_x_des = None
        self.prev_y_des = None
        self.prev_z_des = None
        self.first_gate_recomputed = False
        self.test_x = np.zeros(self.N + 1)
        self.test_y = np.zeros(self.N + 1)
        self.test_z = np.zeros(self.N + 1)

        self.last_known_gates_pos = obs["gates_pos"]
        self.last_known_obstacles_pos = obs["obstacles_pos"]
        self.gates_passed = [False, False, False, False]

        self.time = 0  # time passed since start

        self.min_x, self.max_x = -2.0, 2.0
        self.min_y, self.max_y = -2.0, 2.0
        self.resolution = 0.02

        # compute trajectory last, doesn't work if paths aren't calculated
        self.waypoints = self.compute_trajectory(
            obs["pos"],
            obs["gates_pos"],
            obs["gates_quat"],
            obs["obstacles_pos"],
            0.0,
            obs["target_gate"],
            obs["vel"],
            False,
            False,
        )

        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False

    # parts of this code also taken from attitude_mpc.py
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        target_gate = obs["target_gate"]
        self.gates_passed = np.zeros(4, dtype=bool)
        self.gates_passed[:target_gate] = True

        if not np.allclose(obs["obstacles_pos"], self.last_known_obstacles_pos, atol=0.001):
            distances = np.linalg.norm(
                obs["obstacles_pos"] - self.last_known_obstacles_pos, axis=-1
            )
            max_distance = np.max(distances)
            self.last_known_obstacles_pos = obs["obstacles_pos"]
            if max_distance > 0.02:
                # recompute waypoints because of a detected deviation in obstacle position
                self.trigger_async_recomputation(obs, self.time, target_gate, True, False)

        if not np.allclose(obs["gates_pos"], self.last_known_gates_pos, atol=0.001):
            distances = np.linalg.norm(obs["gates_pos"] - self.last_known_gates_pos, axis=-1)
            max_distance = np.max(distances)
            self.last_known_gates_pos = obs["gates_pos"]
            if max_distance > 0.02:
                # recompute waypoints because of a detected deviation in obstacle position
                self.trigger_async_recomputation(obs, self.time, target_gate, False, True)

        i = min(self._tick, len(self.x_des) - 1)
        if self._tick > i:
            self.finished = True

        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)

        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
            )
        )
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        for j in range(self.N):
            self.test_x[j] = self.x_des[i + j]
            self.test_y[j] = self.y_des[i + j]
            self.test_z[j] = self.z_des[i + j]
            yref = np.zeros(18)
            yref[:3] = [self.x_des[i + j], self.y_des[i + j], self.z_des[i + j]]
            yref[9:11] = 0.35
            self.acados_ocp_solver.set(j, "yref", yref)
        
        self.test_x[self.N] = self.x_des[i + self.N]
        self.test_y[self.N] = self.y_des[i + self.N]
        self.test_z[self.N] = self.z_des[i + self.N]
        
        yref_N = np.zeros(14)
        yref_N[:3] = [self.x_des[i + self.N], self.y_des[i + self.N], self.z_des[i + self.N]]
        yref_N[9:11] = 0.35
        self.acados_ocp_solver.set(self.N, "yref", yref_N)

        self.acados_ocp_solver.solve()

        mpc_states = []
        for j in range(self.N + 1):
            x = self.acados_ocp_solver.get(j, "x")
            mpc_states.append(x[:3])  # Extract position (x, y, z) from state
        self.mpc_horizon = np.array(mpc_states)  # Save the extracted MPC trajectory as mpc_horizon

        x1 = self.acados_ocp_solver.get(1, "x")
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        cmd = x1[10:14]

        return cmd

    # do asynchronous computation with a separate thread to allow the mpc to continue running
    def trigger_async_recomputation(
        self,
        obs: dict[str, NDArray[np.floating]],
        t: float,
        target_gate: int,
        obsDetected: bool,
        gateDetected: bool,
    ):
        """Trigger asynchronous recomputation of the trajectory.

        Args:
            obs: The current observation of the environment, including positions of obstacles and gates.
            t: The current time in the trajectory.
            target_gate: The index of the target gate.
            obsDetected: True if the recalculation is done because of an obstacle
            gateDetected: True if the recalculation is done because of a gate

        Returns:
            None
        """
        self.trajectory_computation_thread = threading.Thread(
            target=self.compute_trajectory,
            args=(
                obs["pos"],
                obs["gates_pos"],
                obs["gates_quat"],
                obs["obstacles_pos"],
                self.time,
                target_gate,
                obs["vel"],
                obsDetected,
                gateDetected,
            ),
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
        """Increment the tick counter."""
        self._tick += 1
        self.time += 1 / self.freq
        return self.finished

    def episode_callback(self):
        """Reset the integral error."""
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

    def construct_cost_map(
        self, obstacles: list[tuple[float, float]], gates: list[dict]
    ) -> np.ndarray:
        """Construct a cost map based on the positions of obstacles and gates.

        Args:
            obstacles: A list of tuples representing the positions of obstacles (x, y).
            gates: A list of dictionaries containing gate information, including position and orientation.

        Returns:
            A 2D numpy array representing the cost map.
        """
        # calculate map dimensions
        width = int(np.ceil((self.max_x - self.min_x) / self.resolution))
        height = int(np.ceil((self.max_y - self.min_y) / self.resolution))
        cost_map = np.ones((height, width))
        obstacle_size = 20
        gate_sides_height = 6

        # mark obstacles on the cost map
        for i, (ox, oy) in enumerate(obstacles[:, :2]):
            gx, gy = self.to_grid(ox, oy)
            if i == 2:
                half_side = int((obstacle_size - 4) / 2)
            elif i == 3:
                half_side = int((obstacle_size + 2) / 2)
            else:
                half_side = int(obstacle_size / 2)
            cost_map[gx - half_side : gx + half_side + 1, gy - half_side : gy + half_side + 1] = 1e6

        # mark sides of gates on the cost map
        for gate in gates:
            gate_g = self.to_grid(gate["pos"][0], gate["pos"][1])
            if gate["rpy"][2] == 0.0 or gate["rpy"][2] == 3.14:
                cost_map[
                    gate_g[0] - 24 : gate_g[0] - 8,
                    gate_g[1] - gate_sides_height : gate_g[1] + gate_sides_height,
                ] = 1e6
                cost_map[
                    gate_g[0] + 8 : gate_g[0] + 24,
                    gate_g[1] - gate_sides_height : gate_g[1] + gate_sides_height,
                ] = 1e6
            else:
                angle = gate["rpy"][2]
                x_offset = int(np.ceil(8 * np.abs(np.cos(angle))))
                y_offset = int(np.ceil(8 * np.abs(np.sin(angle))))
                cost_map[
                    gate_g[0] - x_offset - 14 : gate_g[0] - x_offset,
                    gate_g[1] + y_offset - gate_sides_height : gate_g[1]
                    + y_offset
                    + gate_sides_height,
                ] = 1e6
                cost_map[
                    gate_g[0] + x_offset : gate_g[0] + x_offset + 14,
                    gate_g[1] - y_offset - gate_sides_height : gate_g[1]
                    - y_offset
                    + gate_sides_height,
                ] = 1e6

        cost_map = cost_map.astype(np.int32)
        return cost_map

    def create_gates_dict(
        self, gates_pos: NDArray[np.floating], gates_quat: NDArray[np.floating]
    ) -> list[dict[str, list[float]]]:
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
            rpy = R.from_quat(quat).as_euler("xyz", degrees=False)
            gates.append({"pos": pos.tolist(), "rpy": rpy.tolist()})
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
        gateDetected: bool,
    ) -> list:
        """Compute a list of waypoints based on the current position, gates, and obstacles.

        Args:
            pos: Current position of the drone as a numpy array.
            gates_pos: Positions of the gates as a numpy array.
            gates_quat: Orientations of the gates as a numpy array of quaternions.
            obstacles_pos: Positions of the obstacles as a numpy array.
            tau: Current time in the trajectory, used as starting time.
            target_gate: Index of the target gate.
            vel: The current velocity.
            obsDetected: True if the recalculation is done because of an obstacle
            gateDetected: True if the recalculation is done because of a gate

        Returns:
            A list of waypoints.
        """
        start = time.time()
        gates = self.create_gates_dict(gates_pos, gates_quat)
        cost_map = self.construct_cost_map(obstacles_pos, gates)
        z_adjustment = 0.05

        pre_gate_grids = []
        pos_gate_grids = []
        for i, (gate_pos, gate_quat) in enumerate(zip(gates_pos, gates_quat)):
            rot = R.from_quat(gate_quat).as_matrix()
            fwd = rot[:, 1]
            pre_gate_grid = self.to_grid(
                x=gate_pos[0] - 0.25 * fwd[0], y=gate_pos[1] - 0.25 * fwd[1]
            )
            pos_gate_grid = self.to_grid(
                x=gate_pos[0] + 0.25 * fwd[0], y=gate_pos[1] + 0.25 * fwd[1]
            )
            pre_gate_grids.append(pre_gate_grid)
            pos_gate_grids.append(pos_gate_grid)

        self.pre_gate_1_grid, self.pre_gate_2_grid, self.pre_gate_3_grid, self.pre_gate_4_grid = (
            pre_gate_grids
        )
        (
            self.past_gate_1_grid,
            self.past_gate_2_grid,
            self.past_gate_3_grid,
            self.past_gate_4_grid,
        ) = pos_gate_grids
        self.artifical_pos_1_grid = self.to_grid(x=gates_pos[0][0] - 0.25, y=gates_pos[0][1] - 0.95)
        self.artificial_pos_2_grid = self.to_grid(x=gates_pos[2][0] - 0.6, y=gates_pos[2][1] + 0.15)
        self.past_gate_4_grid = self.to_grid(x=gates_pos[3][0], y=gates_pos[3][1] - 0.7)
        start_pos = pos
        start_grid = self.to_grid(x=start_pos[0], y=start_pos[1])

        # test: adjust post gate 3 to stop it from crashing into the left side of the gate
        # self.pre_gate_3_grid = self.to_grid(x=gates_pos[2][0]+0.12, y=gates_pos[2][1]-0.2)
        # self.past_gate_3_grid = self.to_grid(x=gates_pos[2][0]+0.16, y=gates_pos[2][1]+0.2)

        path_start_to_pre_gate1, _ = route_through_array(
            cost_map, start_grid, self.pre_gate_1_grid, fully_connected=True, geometric=True
        )
        path_start_to_gate1 = np.array(path_start_to_pre_gate1)
        path_start_to_pre_gate1_world_coords = np.array(
            [self.to_coord(gx, gy) for gx, gy in path_start_to_gate1]
        )
        n1 = path_start_to_pre_gate1_world_coords.shape[0]
        z1 = np.linspace(start_pos[2], gates_pos[0][2] - z_adjustment, n1)
        self.path_start_to_pre_gate1_3d = np.column_stack(
            (path_start_to_pre_gate1_world_coords, z1)
        )
        # Path from pre-gate 1 to past gate 1
        path_pre_gate1_to_past_gate1, _ = route_through_array(
            cost_map,
            self.pre_gate_1_grid,
            self.past_gate_1_grid,
            fully_connected=True,
            geometric=True,
        )
        path_pre_gate1_to_past_gate1_array = np.array(path_pre_gate1_to_past_gate1)
        path_pre_gate1_to_past_gate1_world_coords = np.array(
            [self.to_coord(gx, gy) for gx, gy in path_pre_gate1_to_past_gate1_array]
        )
        n1 = path_pre_gate1_to_past_gate1_world_coords.shape[0]
        z1 = np.linspace(gates_pos[0][2] - z_adjustment, gates_pos[0][2] - z_adjustment, n1)
        self.path_pre_gate1_to_past_gate1_3d = np.column_stack(
            (path_pre_gate1_to_past_gate1_world_coords, z1)
        )
        # Path from past gate 1 to artificial point 1
        path_past_gate1_to_artificial_point1, _ = route_through_array(
            cost_map,
            self.past_gate_1_grid,
            self.artifical_pos_1_grid,
            fully_connected=True,
            geometric=True,
        )
        path_past_gate1_to_artificial_point1_array = np.array(path_past_gate1_to_artificial_point1)
        # Path from artificial point 1 to pre-gate 2
        path_artificial_point1_to_pre_gate2, _ = route_through_array(
            cost_map,
            self.artifical_pos_1_grid,
            self.pre_gate_2_grid,
            fully_connected=True,
            geometric=True,
        )
        path_artificial_point1_to_pre_gate2_array = np.array(path_artificial_point1_to_pre_gate2)
        path_artificial_point1_to_pre_gate2_world_coords = np.concatenate(
            [
                np.array(
                    [self.to_coord(gx, gy) for gx, gy in path_past_gate1_to_artificial_point1_array]
                ),
                np.array(
                    [self.to_coord(gx, gy) for gx, gy in path_artificial_point1_to_pre_gate2_array]
                ),
            ]
        )
        n1 = path_artificial_point1_to_pre_gate2_world_coords.shape[0]
        z1 = np.linspace(gates_pos[0][2] - z_adjustment, gates_pos[1][2], n1)
        self.path_past_gate1_3d_to_pre_gate2_3d = np.column_stack(
            (path_artificial_point1_to_pre_gate2_world_coords, z1)
        )
        # Path from pre-gate 2 to past gate 2
        path_pre_gate2_to_past_gate2, _ = route_through_array(
            cost_map,
            self.pre_gate_2_grid,
            self.past_gate_2_grid,
            fully_connected=True,
            geometric=True,
        )
        path_pre_gate2_to_past_gate2_array = np.array(path_pre_gate2_to_past_gate2)
        path_pre_gate2_to_past_gate2_world_coords = np.array(
            [self.to_coord(gx, gy) for gx, gy in path_pre_gate2_to_past_gate2_array]
        )
        n2 = path_pre_gate2_to_past_gate2_world_coords.shape[0]
        z2 = np.linspace(gates_pos[1][2], gates_pos[1][2], n2)
        self.path_pre_gate2_to_past_gate2_3d = np.column_stack(
            (path_pre_gate2_to_past_gate2_world_coords, z2)
        )
        # Path from past gate 2 to pre-gate 3
        path_past_gate2_to_pre_gate3, _ = route_through_array(
            cost_map,
            self.past_gate_2_grid,
            self.pre_gate_3_grid,
            fully_connected=True,
            geometric=True,
        )
        path_past_gate2_to_pre_gate3_array = np.array(path_past_gate2_to_pre_gate3)
        path_past_gate2_to_pre_gate3_world_coords = np.array(
            [self.to_coord(gx, gy) for gx, gy in path_past_gate2_to_pre_gate3_array]
        )
        n2_to_3 = path_past_gate2_to_pre_gate3_world_coords.shape[0]
        z2_to_3 = np.linspace(gates_pos[1][2], gates_pos[2][2] - z_adjustment, n2_to_3)
        self.path_past_gate2_to_pre_gate3_3d = np.column_stack(
            (path_past_gate2_to_pre_gate3_world_coords, z2_to_3)
        )
        # Path from pre-gate 3 to past gate 3
        path_pre_gate3_to_past_gate3, _ = route_through_array(
            cost_map,
            self.pre_gate_3_grid,
            self.past_gate_3_grid,
            fully_connected=True,
            geometric=True,
        )
        path_pre_gate3_to_past_gate3_array = np.array(path_pre_gate3_to_past_gate3)
        path_pre_gate3_to_past_gate3_world_coords = np.array(
            [self.to_coord(gx, gy) for gx, gy in path_pre_gate3_to_past_gate3_array]
        )
        n3 = path_pre_gate3_to_past_gate3_world_coords.shape[0]
        z3 = np.linspace(gates_pos[2][2] - z_adjustment, gates_pos[2][2] - z_adjustment, n3)
        self.path_pre_gate3_to_past_gate3_3d = np.column_stack(
            (path_pre_gate3_to_past_gate3_world_coords, z3)
        )
        path_past_gate3_to_artificial_point2, _ = route_through_array(
            cost_map,
            self.past_gate_3_grid,
            self.artificial_pos_2_grid,
            fully_connected=True,
            geometric=True,
        )
        path_past_gate3_to_artificial_point3_array = np.array(path_past_gate3_to_artificial_point2)
        # Path from artificial point 1 to pre-gate 2
        path_artificial_point2_to_pre_gate4, _ = route_through_array(
            cost_map,
            self.artificial_pos_2_grid,
            self.pre_gate_4_grid,
            fully_connected=True,
            geometric=True,
        )
        self.path_artificial_point2_to_pre_gate4_array = np.array(
            path_artificial_point2_to_pre_gate4
        )
        path_artificial_point2_to_pre_gate4_world_coords = np.concatenate(
            [
                np.array(
                    [self.to_coord(gx, gy) for gx, gy in path_past_gate3_to_artificial_point3_array]
                ),
                np.array(
                    [
                        self.to_coord(gx, gy)
                        for gx, gy in self.path_artificial_point2_to_pre_gate4_array
                    ]
                ),
            ]
        )
        n1 = path_artificial_point2_to_pre_gate4_world_coords.shape[0]
        z1 = np.linspace(gates_pos[0][2] - z_adjustment, gates_pos[1][2], n1)
        self.path_past_gate3_to_pre_gate4_3d = np.column_stack(
            (path_artificial_point2_to_pre_gate4_world_coords, z1)
        )

        # Path from pre-gate 4 to past gate 4
        path_pre_gate4_to_past_gate4, _ = route_through_array(
            cost_map,
            self.pre_gate_4_grid,
            self.past_gate_4_grid,
            fully_connected=True,
            geometric=True,
        )
        path_pre_gate4_to_past_gate4_array = np.array(path_pre_gate4_to_past_gate4)
        path_pre_gate4_to_past_gate4_world_coords = np.array(
            [self.to_coord(gx, gy) for gx, gy in path_pre_gate4_to_past_gate4_array]
        )
        n4 = path_pre_gate4_to_past_gate4_world_coords.shape[0]
        z4 = np.linspace(gates_pos[3][2], gates_pos[3][2], n4)
        self.path_pre_gate4_to_past_gate4_3d = np.column_stack(
            (path_pre_gate4_to_past_gate4_world_coords, z4)
        )

        if target_gate == 0 and not gateDetected and not obsDetected:
            self.combined_3d_path = np.concatenate(
                [
                    self.path_start_to_pre_gate1_3d,
                    self.path_pre_gate1_to_past_gate1_3d,
                    self.path_past_gate1_3d_to_pre_gate2_3d,
                    self.path_pre_gate2_to_past_gate2_3d,
                    self.path_past_gate2_to_pre_gate3_3d,
                    self.path_pre_gate3_to_past_gate3_3d,
                    self.path_past_gate3_to_pre_gate4_3d,
                    self.path_pre_gate4_to_past_gate4_3d,
                ]
            )
        elif target_gate == 0 and gateDetected:
            # check whether it is the first or second gate (it can be both!)
            if not self.first_gate_recomputed:
                path_start_to_past_gate1, _ = route_through_array(
                    cost_map,
                    start_grid,
                    self.past_gate_1_grid,
                    fully_connected=True,
                    geometric=True,
                )
                path_start_to_past_gate1_array = np.array(path_start_to_past_gate1)
                path_start_to_past_gate1_world_coords = np.array(
                    [self.to_coord(gx, gy) for gx, gy in path_start_to_past_gate1_array]
                )
                n1 = path_start_to_past_gate1_world_coords.shape[0]
                z1 = np.linspace(start_pos[2], gates_pos[0][2] - z_adjustment, n1)
                path_start_to_past_gate1_3d = np.column_stack(
                    (path_start_to_past_gate1_world_coords, z1)
                )
                self.combined_3d_path = np.concatenate(
                    [
                        path_start_to_past_gate1_3d,
                        self.path_past_gate1_3d_to_pre_gate2_3d,
                        self.path_pre_gate2_to_past_gate2_3d,
                        self.path_past_gate2_to_pre_gate3_3d,
                        self.path_pre_gate3_to_past_gate3_3d,
                        self.path_past_gate3_to_pre_gate4_3d,
                        self.path_pre_gate4_to_past_gate4_3d,
                    ]
                )
                self.first_gate_recomputed = True
            else:
                path_start_to_artificial_point_1, _ = route_through_array(
                    cost_map,
                    start_grid,
                    self.artifical_pos_1_grid,
                    fully_connected=True,
                    geometric=True,
                )
                path_start_to_artificial_point_1_array = np.array(path_start_to_artificial_point_1)
                # Path from artificial point 1 to pre-gate 2
                path_artificial_point1_to_pre_gate2, _ = route_through_array(
                    cost_map,
                    self.artifical_pos_1_grid,
                    self.pre_gate_2_grid,
                    fully_connected=True,
                    geometric=True,
                )
                path_artificial_point1_to_pre_gate2_array = np.array(
                    path_artificial_point1_to_pre_gate2
                )
                path_start_to_pre_gate2_world_coords = np.concatenate(
                    [
                        np.array(
                            [
                                self.to_coord(gx, gy)
                                for gx, gy in path_start_to_artificial_point_1_array
                            ]
                        ),
                        np.array(
                            [
                                self.to_coord(gx, gy)
                                for gx, gy in path_artificial_point1_to_pre_gate2_array
                            ]
                        ),
                    ]
                )
                n1 = path_start_to_pre_gate2_world_coords.shape[0]
                z1 = np.linspace(start_pos[2], gates_pos[1][2], n1)
                path_start_to_pre_gate2_3d = np.column_stack(
                    (path_start_to_pre_gate2_world_coords, z1)
                )
                # Path from pre-gate 2 to past gate 2
                path_pre_gate2_to_past_gate2, _ = route_through_array(
                    cost_map,
                    self.pre_gate_2_grid,
                    self.past_gate_2_grid,
                    fully_connected=True,
                    geometric=True,
                )
                path_pre_gate2_to_past_gate2_array = np.array(path_pre_gate2_to_past_gate2)
                path_pre_gate2_to_past_gate2_world_coords = np.array(
                    [self.to_coord(gx, gy) for gx, gy in path_pre_gate2_to_past_gate2_array]
                )
                n2 = path_pre_gate2_to_past_gate2_world_coords.shape[0]
                z2 = np.linspace(gates_pos[1][2], gates_pos[1][2], n2)
                self.path_pre_gate2_to_past_gate2_3d = np.column_stack(
                    (path_pre_gate2_to_past_gate2_world_coords, z2)
                )
                self.combined_3d_path = np.concatenate(
                    [
                        path_start_to_pre_gate2_3d,
                        self.path_pre_gate2_to_past_gate2_3d,
                        self.path_past_gate2_to_pre_gate3_3d,
                        self.path_pre_gate3_to_past_gate3_3d,
                        self.path_past_gate3_to_pre_gate4_3d,
                        self.path_pre_gate4_to_past_gate4_3d,
                    ]
                )
        elif target_gate == 0 and obsDetected:
            path_start_to_past_gate1, _ = route_through_array(
                cost_map, start_grid, self.past_gate_1_grid, fully_connected=True, geometric=True
            )
            path_start_to_past_gate1_array = np.array(path_start_to_past_gate1)
            path_start_to_past_gate1_world_coords = np.array(
                [self.to_coord(gx, gy) for gx, gy in path_start_to_past_gate1_array]
            )
            n1 = path_start_to_past_gate1_world_coords.shape[0]
            z1 = np.linspace(start_pos[2], gates_pos[0][2] - z_adjustment, n1)
            path_start_to_past_gate1_3d = np.column_stack(
                (path_start_to_past_gate1_world_coords, z1)
            )
            self.combined_3d_path = np.concatenate(
                [
                    path_start_to_past_gate1_3d,
                    self.path_past_gate1_3d_to_pre_gate2_3d,
                    self.path_pre_gate2_to_past_gate2_3d,
                    self.path_past_gate2_to_pre_gate3_3d,
                    self.path_pre_gate3_to_past_gate3_3d,
                    self.path_past_gate3_to_pre_gate4_3d,
                    self.path_pre_gate4_to_past_gate4_3d,
                ]
            )
        elif target_gate == 1 and gateDetected:
            if np.linalg.norm(pos[:2] - gates_pos[0][:2]) < 0.2:
                # drone is still close to the first gate, include artificial point
                path_start_to_artificial_point_1, _ = route_through_array(
                    cost_map,
                    start_grid,
                    self.artifical_pos_1_grid,
                    fully_connected=True,
                    geometric=True,
                )
                path_start_to_artificial_point_1_array = np.array(path_start_to_artificial_point_1)
                # Path from artificial point 1 to pre-gate 2
                path_artificial_point1_to_pre_gate2, _ = route_through_array(
                    cost_map,
                    self.artifical_pos_1_grid,
                    self.pre_gate_2_grid,
                    fully_connected=True,
                    geometric=True,
                )
                path_artificial_point1_to_pre_gate2_array = np.array(
                    path_artificial_point1_to_pre_gate2
                )
                path_start_to_pre_gate2_world_coords = np.concatenate(
                    [
                        np.array(
                            [
                                self.to_coord(gx, gy)
                                for gx, gy in path_start_to_artificial_point_1_array
                            ]
                        ),
                        np.array(
                            [
                                self.to_coord(gx, gy)
                                for gx, gy in path_artificial_point1_to_pre_gate2_array
                            ]
                        ),
                    ]
                )
                n1 = path_start_to_pre_gate2_world_coords.shape[0]
                z1 = np.linspace(start_pos[2], gates_pos[1][2], n1)
                path_start_to_pre_gate2_3d = np.column_stack(
                    (path_start_to_pre_gate2_world_coords, z1)
                )
                # Path from pre-gate 2 to past gate 2
                path_pre_gate2_to_past_gate2, _ = route_through_array(
                    cost_map,
                    self.pre_gate_2_grid,
                    self.past_gate_2_grid,
                    fully_connected=True,
                    geometric=True,
                )
                path_pre_gate2_to_past_gate2_array = np.array(path_pre_gate2_to_past_gate2)
                path_pre_gate2_to_past_gate2_world_coords = np.array(
                    [self.to_coord(gx, gy) for gx, gy in path_pre_gate2_to_past_gate2_array]
                )
                n2 = path_pre_gate2_to_past_gate2_world_coords.shape[0]
                z2 = np.linspace(gates_pos[1][2], gates_pos[1][2], n2)
                self.path_pre_gate2_to_past_gate2_3d = np.column_stack(
                    (path_pre_gate2_to_past_gate2_world_coords, z2)
                )
                self.combined_3d_path = np.concatenate(
                    [
                        path_start_to_pre_gate2_3d,
                        self.path_pre_gate2_to_past_gate2_3d,
                        self.path_past_gate2_to_pre_gate3_3d,
                        self.path_pre_gate3_to_past_gate3_3d,
                        self.path_past_gate3_to_pre_gate4_3d,
                        self.path_pre_gate4_to_past_gate4_3d,
                    ]
                )
            else:
                path_start_to_pre_gate2, _ = route_through_array(
                    cost_map, start_grid, self.pre_gate_2_grid, fully_connected=True, geometric=True
                )
                path_start_to_pre_gate2_array = np.array(path_start_to_pre_gate2)
                path_start_to_pre_gate2_world_coords = np.array(
                    [self.to_coord(gx, gy) for gx, gy in path_start_to_pre_gate2_array]
                )
                n1 = path_start_to_pre_gate2_world_coords.shape[0]
                z1 = np.linspace(start_pos[2], gates_pos[1][2], n1)
                path_start_to_pre_gate2_3d = np.column_stack(
                    (path_start_to_pre_gate2_world_coords, z1)
                )
                self.combined_3d_path = np.concatenate(
                    [
                        path_start_to_pre_gate2_3d,
                        self.path_pre_gate2_to_past_gate2_3d,
                        self.path_past_gate2_to_pre_gate3_3d,
                        self.path_pre_gate3_to_past_gate3_3d,
                        self.path_past_gate3_to_pre_gate4_3d,
                        self.path_pre_gate4_to_past_gate4_3d,
                    ]
                )
        elif target_gate == 2 and gateDetected or target_gate == 2 and obsDetected:
            if obsDetected:
                path_start_to_past_gate3, _ = route_through_array(
                    cost_map,
                    start_grid,
                    self.past_gate_3_grid,
                    fully_connected=True,
                    geometric=True,
                )
                path_start_to_past_gate3_array = np.array(path_start_to_past_gate3)
                path_start_to_past_gate3_world_coords = np.array(
                    [self.to_coord(gx, gy) for gx, gy in path_start_to_past_gate3_array]
                )
                n1 = path_start_to_past_gate3_world_coords.shape[0]
                z1 = np.linspace(start_pos[2], gates_pos[2][2] - z_adjustment, n1)
                path_start_to_past_gate3_3d = np.column_stack(
                    (path_start_to_past_gate3_world_coords, z1)
                )
                self.combined_3d_path = np.concatenate(
                    [
                        path_start_to_past_gate3_3d,
                        self.path_past_gate3_to_pre_gate4_3d,
                        self.path_pre_gate4_to_past_gate4_3d,
                    ]
                )
            else:
                path_start_to_pre_gate3, _ = route_through_array(
                    cost_map, start_grid, self.pre_gate_3_grid, fully_connected=True, geometric=True
                )
                path_start_to_pre_gate3_array = np.array(path_start_to_pre_gate3)
                path_start_to_past_gate3_world_coords = np.concatenate(
                    [
                        np.array(
                            [self.to_coord(gx, gy) for gx, gy in path_start_to_pre_gate3_array]
                        ),
                        np.array(
                            [self.to_coord(gx, gy) for gx, gy in path_pre_gate3_to_past_gate3]
                        ),
                    ]
                )
                n1 = path_start_to_past_gate3_world_coords.shape[0]
                z1 = np.linspace(start_pos[2], gates_pos[2][2] - z_adjustment, n1)
                path_start_to_past_gate3_3d = np.column_stack(
                    (path_start_to_past_gate3_world_coords, z1)
                )
                self.combined_3d_path = np.concatenate(
                    [
                        path_start_to_past_gate3_3d,
                        self.path_past_gate3_to_pre_gate4_3d,
                        self.path_pre_gate4_to_past_gate4_3d,
                    ]
                )
        elif target_gate == 3:
            if obsDetected:
                # obstacle 4 detected
                if obstacles_pos[3][0] < gates_pos[3][0]:
                    # enough distance between obstacle and gate to go directly
                    path_start_to_pre_gate4, _ = route_through_array(
                        cost_map,
                        start_grid,
                        self.pre_gate_4_grid,
                        fully_connected=True,
                        geometric=True,
                    )
                    path_start_to_pre_gate_4_world_coords = np.array(
                        [self.to_coord(gx, gy) for gx, gy in path_start_to_pre_gate4]
                    )
                    n1 = path_start_to_pre_gate_4_world_coords.shape[0]
                    z1 = np.linspace(start_pos[2], gates_pos[3][2], n1)
                    path_start_to_pre_gate4_3d = np.column_stack(
                        (path_start_to_pre_gate_4_world_coords, z1)
                    )
                    self.combined_3d_path = np.concatenate(
                        [path_start_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d]
                    )
                else:
                    # use artifiicial point 2
                    path_start_to_artificial_point_2, _ = route_through_array(
                        cost_map,
                        start_grid,
                        self.artificial_pos_2_grid,
                        fully_connected=True,
                        geometric=True,
                    )
                    path_artificial_point2_to_pre_gate4, _ = route_through_array(
                        cost_map,
                        self.artificial_pos_2_grid,
                        self.pre_gate_4_grid,
                        fully_connected=True,
                        geometric=True,
                    )
                    self.path_start_to_artificial_point2_array = np.array(
                        path_start_to_artificial_point_2
                    )
                    self.path_artificial_point2_to_pre_gate4_array = np.array(
                        path_artificial_point2_to_pre_gate4
                    )
                    path_start_to_pre_gate4_world_coords = np.concatenate(
                        [
                            np.array(
                                [
                                    self.to_coord(gx, gy)
                                    for gx, gy in self.path_artificial_point2_to_pre_gate4_array
                                ]
                            ),
                            np.array(
                                [
                                    self.to_coord(gx, gy)
                                    for gx, gy in self.path_artificial_point2_to_pre_gate4_array
                                ]
                            ),
                        ]
                    )
                    n1 = path_start_to_pre_gate4_world_coords.shape[0]
                    z1 = np.linspace(gates_pos[0][2] - z_adjustment, gates_pos[1][2], n1)
                    self.path_start_to_pre_gate4_3d = np.column_stack(
                        (path_start_to_pre_gate4_world_coords, z1)
                    )
            else:
                path_start_to_pre_gate4, _ = route_through_array(
                    cost_map, start_grid, self.pre_gate_4_grid, fully_connected=True, geometric=True
                )
                path_start_to_pre_gate_4_world_coords = np.array(
                    [self.to_coord(gx, gy) for gx, gy in path_start_to_pre_gate4]
                )
                n1 = path_start_to_pre_gate_4_world_coords.shape[0]
                z1 = np.linspace(start_pos[2], gates_pos[3][2], n1)
                path_start_to_pre_gate4_3d = np.column_stack(
                    (path_start_to_pre_gate_4_world_coords, z1)
                )
                self.combined_3d_path = np.concatenate(
                    [path_start_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d]
                )

        # sample waypoints to avoid jagged behavior, using every 4th waypoint yields the smoothest behavior 
        # while ensuring that the trajectory doesn't deviate too much from the original waypoints
        self.waypoints = self.combined_3d_path[::4]

        # use b splines on reduced waypoints
        tck, u = splprep(
            [self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2]],
            s=0.1,  # s = smoothing factor
            k=3,
        )

        ts = np.linspace(0, 1, len(self.waypoints) * 5)
        x_full, y_full, z_full = splev(ts, tck)
        self.full_traj = np.stack((x_full, y_full, z_full), axis=-1)

        ts = np.linspace(0, 1, int(self.freq * (self.des_completion_time - tau)))

        self.x_des, self.y_des, self.z_des = splev(ts, tck)

        # extend trajectory for MPC horizon
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * (2 * self.N + 1)))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * (2 * self.N + 1)))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * (2 * self.N + 1)))

        if obsDetected or gateDetected:
            self.next_x_in_prev_traj = self.prev_x_des[self._tick + 1]
            self.next_y_in_prev_traj = self.prev_y_des[self._tick + 1]
            self.next_z_in_prev_traj = self.prev_z_des[self._tick + 1]

            closest_idx = self.find_closest_point(self.x_des, self.y_des, self.z_des, self.next_x_in_prev_traj, self.next_y_in_prev_traj, self.next_z_in_prev_traj)

            # Remove the first 'lookahead' values from the trajectory
            lookahead = min(len(self.x_des) - 1, closest_idx)
            self.x_des = self.x_des[lookahead:]
            self.y_des = self.y_des[lookahead:]
            self.z_des = self.z_des[lookahead:]

        # smooth the points of the old and new trajectory to ensure that the trajectory is smooth
        # no significant improvement, so turned off via self.enable_smoothing (set to False in init)
        if self.prev_x_des is not None and self.enable_smoothing:
            num_transition_points = min(self.transition_length, len(self.prev_x_des) - self._tick)
            if num_transition_points > 0:
                # get the remaining portion of the old trajectory from current MPC position + 1
                old_x_remaining = self.prev_x_des[
                    self._tick + 1 : self._tick + 1 + num_transition_points
                ]
                old_y_remaining = self.prev_y_des[
                    self._tick + 1 : self._tick + 1 + num_transition_points
                ]
                old_z_remaining = self.prev_z_des[
                    self._tick + 1 : self._tick + 1 + num_transition_points
                ]

                # get the corresponding section from the new trajectory
                new_x_for_blend = self.x_des[:num_transition_points]
                new_y_for_blend = self.y_des[:num_transition_points]
                new_z_for_blend = self.z_des[:num_transition_points]

                extended_length = int(num_transition_points * self.slowdown_factor)
                extended_weights = np.linspace(0, 1, extended_length)

                # interpolate both old and new trajectories to the extended length
                old_interp_x = np.interp(
                    extended_weights, np.linspace(0, 1, num_transition_points), old_x_remaining
                )
                old_interp_y = np.interp(
                    extended_weights, np.linspace(0, 1, num_transition_points), old_y_remaining
                )
                old_interp_z = np.interp(
                    extended_weights, np.linspace(0, 1, num_transition_points), old_z_remaining
                )

                new_interp_x = np.interp(
                    extended_weights, np.linspace(0, 1, num_transition_points), new_x_for_blend
                )
                new_interp_y = np.interp(
                    extended_weights, np.linspace(0, 1, num_transition_points), new_y_for_blend
                )
                new_interp_z = np.interp(
                    extended_weights, np.linspace(0, 1, num_transition_points), new_z_for_blend
                )

                # blend the extended trajectories
                blended_x = (1 - extended_weights) * old_interp_x + extended_weights * new_interp_x
                blended_y = (1 - extended_weights) * old_interp_y + extended_weights * new_interp_y
                blended_z = (1 - extended_weights) * old_interp_z + extended_weights * new_interp_z

                # combine with rest of new trajectory
                smooth_x_des = np.concatenate([blended_x, self.x_des[num_transition_points:]])
                smooth_y_des = np.concatenate([blended_y, self.y_des[num_transition_points:]])
                smooth_z_des = np.concatenate([blended_z, self.z_des[num_transition_points:]])

                self.x_des = smooth_x_des
                self.y_des = smooth_y_des
                self.z_des = smooth_z_des
        

        self._tick = 0
        self.finished = False

        self.prev_x_des = self.x_des
        self.prev_y_des = self.y_des
        self.prev_z_des = self.z_des

        end = time.time()
        print("trajectory updated! Execution time:", (end - start) * 1e3, "ms")
        return self.waypoints

    def find_closest_point(
        self,
        x_path: NDArray[np.floating],
        y_path: NDArray[np.floating],
        z_path: NDArray[np.floating],
        x_point: float,
        y_point: float,
        z_point: float,
    ) -> int:
        """Finds the closest point on a path to a given point.

        Args:
            x_path: The x-coordinates of the path.
            y_path: The y-coordinates of the path.
            z_path: The z-coordinates of the path.
            x_point: The x-coordinate of the point to compare.
            y_point: The y-coordinate of the point to compare.
            z_point: The z-coordinate of the point to compare.

        Returns:
            The index of the closest point on the path to the given point.
        """
        # Goal: find the closest point to [self.next_x_in_prev_traj, self.next_y_in_prev_traj, self.next_z_in_prev_traj]
        # and remove all previous values from self.x_des, self.y_des and self.z_des
        search_window = min(50, len(x_path))
        patience = 10
        closest_idx = 0
        no_improvement_count = 0
        min_dist = np.inf

        for i in range(search_window):
            dist = (
                (x_path[i] - x_point) ** 2
                + (y_path[i] - y_point) ** 2
                + (z_path[i] - z_point) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                break
        return closest_idx
        
    def get_trajectory_and_mpc_horizon(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve the full trajectory, the MPC horizon and the part of the trajectory fed into the MPC for drawing in sim.py.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the full trajectory, the part of the trajectory fed into the MPC and the MPC's horizon.
        """
        self.test = np.stack((self.test_x, self.test_y, self.test_z), axis=-1)
        return self.full_traj, self.mpc_horizon, self.test
