"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from skimage.graph import route_through_array  #type: ignore

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    Q = np.diag(
        [
            10.0,
            10.0,
            10.0,  # Position
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

    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    # ocp.cost.yref = np.zeros((ny, ))
    # ocp.cost.yref_e = np.zeros((ny_e, ))
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
    """
    # Set State Constraints with velocity constraints
    ocp.constraints.lbx = np.array([-2.0, -2.0, -2.0, 0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([2.0, 2.0, 2.0, 0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([3, 4, 5, 9, 10, 11, 12, 13])
    """    

    # Set Input Constraints
    # ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0. -10.0])
    # ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0])
    # ocp.constraints.idxbu = np.array([0, 1, 2, 3])

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

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc.json", verbose=verbose)

    return acados_ocp_solver, ocp


class MPController(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

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

        self.des_completion_time = 8
        self.slowdown_factor = 1.5
        self.N = 20
        self.T_HORIZON = 1.0
        self.dt = self.T_HORIZON / self.N
        self.transition_length = 10
        self.prev_x_des = None
        self.prev_y_des = None
        self.prev_z_des = None

        self.last_known_gates_pos = obs["gates_pos"]
        self.last_known_obstacles_pos = obs["obstacles_pos"]
        self.trajectory_history = []
        self.gates_passed = [False, False, False, False]

        self.time = 0 # time passed since start
        self.plot = False

        self.min_x, self.max_x = -2.0, 2.0
        self.min_y, self.max_y = -2.0, 2.0
        self.resolution = 0.02

        self._finished = False
        self.recompute_flag = False

        #setup once, don't recalculate this every time
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        gates = self.create_gates_dict(gates_pos, gates_quat)
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
        self.artifical_pos_1_grid = self.to_grid(x=gates_pos[0][0]-0.25, y=gates_pos[0][1]-0.95)
        self.artificial_pos_2_grid = self.to_grid(x=gates_pos[2][0]-0.5, y=gates_pos[2][1]+0.15)
        self.past_gate_4_grid = self.to_grid(x=gates_pos[3][0], y=gates_pos[3][1]-0.7)

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
        self.waypoints = self.compute_waypoints(obs["pos"], obs["gates_pos"], obs["gates_quat"], obs["obstacles_pos"], 0.0, obs["target_gate"], obs["vel"], False, False)

        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False

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
        #print(f"dist to 1st obstacle: {np.sqrt((obs['obstacles_pos'][0][0]-obs['pos'][0])**2 + (obs['obstacles_pos'][0][1]-obs['pos'][1])**2)}")
        target_gate = obs["target_gate"]
        self.gates_passed = np.zeros(4, dtype=bool) 
        self.gates_passed[:target_gate] = True

        if not np.allclose(obs["obstacles_pos"], self.last_known_obstacles_pos, atol=0.001):
            distances = np.linalg.norm(obs["obstacles_pos"] - self.last_known_obstacles_pos, axis=-1)
            max_distance = np.max(distances)
            self.last_known_obstacles_pos = obs["obstacles_pos"]
            if max_distance > 0.02:
                print("recomputing waypoints")
                self.trigger_async_recomputation(obs, self.time, target_gate, True, False)

        if not np.allclose(obs["gates_pos"], self.last_known_gates_pos, atol=0.001):
            distances = np.linalg.norm(obs["gates_pos"] - self.last_known_gates_pos, axis=-1)
            max_distance = np.max(distances)
            self.last_known_gates_pos = obs["gates_pos"]
            if max_distance > 0.02:
                print("recomputing waypoints")
                self.trigger_async_recomputation(obs, self.time, target_gate, False, True)

        
        i = min(self._tick, len(self.x_des) - 1)
        if self._tick > i:
            self.finished = True

        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

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
            yref = np.array(
                [
                    self.x_des[i + j],
                    self.y_des[i + j],
                    self.z_des[i + j],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.35,
                    0.35,
                    0.0,
                    0.0,
                    0.0,    
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            self.acados_ocp_solver.set(j, "yref", yref)
        yref_N = np.array(
            [
                self.x_des[i + self.N],
                self.y_des[i + self.N],
                self.z_des[i + self.N],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.35,
                0.35,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.acados_ocp_solver.set(self.N, "yref", yref_N)

        self.acados_ocp_solver.solve()

        # Extract the MPC trajectory from the solver and save it
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
            target=self.compute_waypoints, 
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
        """Increment the tick counter."""
        self._tick += 1
        self.time += 1/self.freq
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
        gate_sides_height = 6
        
        # Mark obstacles on the cost map
        for i, (ox, oy) in enumerate(obstacles[:, :2]):
            gx, gy = self.to_grid(ox, oy)
            half_side = int(obstacle_size / 2)
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

    def compute_waypoints(
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

        Returns:
            A list of waypoints.
        """
        start = time.time()
        gates  = self.create_gates_dict(gates_pos, gates_quat)
        cost_map = self.construct_cost_map(obstacles_pos, gates)

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
        self.artifical_pos_1_grid = self.to_grid(x=gates_pos[0][0]-0.25, y=gates_pos[0][1]-0.95)
        self.artificial_pos_2_grid = self.to_grid(x=gates_pos[2][0]-0.5, y=gates_pos[2][1]+0.15)
        self.past_gate_4_grid = self.to_grid(x=gates_pos[3][0], y=gates_pos[3][1]-0.7)
        start_pos = pos
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
        elif (obsDetected and target_gate == 0):
            print("trajectory calculation because of second obstacle")
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
            #else:
            #    print("trajectory calculation because of first obstacle")
            #    #first obstacle
            #    combined_3d_path = np.concatenate([
            #        self.path_start_to_pre_gate1_3d, self.path_pre_gate1_to_past_gate1_3d, self.path_past_gate1_3d_to_pre_gate2_3d,
            #        self.path_pre_gate2_to_past_gate2_3d, self.path_past_gate2_to_pre_gate3_3d, self.path_pre_gate3_to_past_gate3_3d,
            #        self.path_past_gate3_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
            #    ])
        elif (target_gate == 1 and gateDetected):
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
        elif target_gate == 2 and gateDetected or target_gate == 2 and obsDetected:
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
        elif target_gate == 3:
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
        #elif (obsDetected and target_gate == 2):
        #    path_start_to_artificial_point2, _ = route_through_array(
        #        cost_map, start_grid, self.artificial_pos_2_grid,
        #        fully_connected=True, geometric=True
        #    )
        #    path_start_to_artificial_point2_array = np.array(path_start_to_artificial_point2)
        #    path_start_to_pre_gate_4_world_coords = np.concatenate([np.array([self.to_coord(gx, gy) for gx, gy in path_start_to_artificial_point2_array]), np.array([self.to_coord(gx, gy) for gx, gy in self.path_artificial_point2_to_pre_gate4_array])])
        #    n1 = path_start_to_pre_gate_4_world_coords.shape[0]
        #    z1 = np.linspace(start_pos[2], gates_pos[3][2], n1)
        #    path_start_to_pre_gate4_3d = np.column_stack((path_start_to_pre_gate_4_world_coords, z1))
        #    combined_3d_path = np.concatenate([
        #        path_start_to_pre_gate4_3d, self.path_pre_gate4_to_past_gate4_3d
        #    ])
        """
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
        """
        
        reduced_3d_path = combined_3d_path[::4]
        
        #num_points = reduced_3d_path.shape[0]
        self.waypoints = reduced_3d_path

        ts = np.linspace(0, 1, np.shape(self.waypoints)[0])
        cs_x = CubicSpline(ts, self.waypoints[:, 0])
        cs_y = CubicSpline(ts, self.waypoints[:, 1])
        cs_z = CubicSpline(ts, self.waypoints[:, 2])
        
        # Store points for plotting
        self.full_traj = np.stack((cs_x(ts), cs_y(ts), cs_z(ts)), axis=-1)  # shape (T, 3)
        
        # Generate trajectory at controller frequency
        print(f"desired completion time is: {self.des_completion_time}")
        print(f"remaining time is: {self.des_completion_time- tau}")
        ts = np.linspace(0, 1, int(self.freq * (self.des_completion_time-tau)))
        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        
        if self.prev_x_des is not None:
            print("applying smoothing")
            num_transition_points = min(self.transition_length, len(self.prev_x_des) - self._tick)
            if num_transition_points > 0:
                """
                # Get the remaining portion of old trajectory from current MPC position
                old_x_remaining = self.prev_x_des[self._tick:self._tick + num_transition_points]
                old_y_remaining = self.prev_y_des[self._tick:self._tick + num_transition_points]
                old_z_remaining = self.prev_z_des[self._tick:self._tick + num_transition_points]
                
                # Get corresponding section from new trajectory
                new_x_for_blend = self.x_des[:num_transition_points]
                new_y_for_blend = self.y_des[:num_transition_points]
                new_z_for_blend = self.z_des[:num_transition_points]
                
                # Create blending weights (0 = old, 1 = new)
                blend_weights = np.linspace(0, 1, num_transition_points)
                
                # Blend the trajectories
                blended_x = np.zeros_like(old_x_remaining)
                blended_y = np.zeros_like(old_y_remaining)
                blended_z = np.zeros_like(old_z_remaining)
                
                for i in range(num_transition_points):
                    w = blend_weights[i]
                    blended_x[i] = (1 - w) * old_x_remaining[i] + w * new_x_for_blend[i]
                    blended_y[i] = (1 - w) * old_y_remaining[i] + w * new_y_for_blend[i]
                    blended_z[i] = (1 - w) * old_z_remaining[i] + w * new_z_for_blend[i]
                
                # Combine blended section with rest of new trajectory
                smooth_x_des = np.concatenate([blended_x, self.x_des[num_transition_points:]])
                smooth_y_des = np.concatenate([blended_y, self.y_des[num_transition_points:]])
                smooth_z_des = np.concatenate([blended_z, self.z_des[num_transition_points:]])
                """
                # Get the remaining portion of old trajectory from current MPC position
                old_x_remaining = self.prev_x_des[self._tick:self._tick + num_transition_points]
                old_y_remaining = self.prev_y_des[self._tick:self._tick + num_transition_points]
                old_z_remaining = self.prev_z_des[self._tick:self._tick + num_transition_points]
                
                # Get corresponding section from new trajectory
                new_x_for_blend = self.x_des[:num_transition_points]
                new_y_for_blend = self.y_des[:num_transition_points]
                new_z_for_blend = self.z_des[:num_transition_points]

                extended_length = int(num_transition_points * self.slowdown_factor)
                extended_weights = np.linspace(0, 1, extended_length)
                
                # Interpolate both old and new trajectories to extended length
                old_interp_x = np.interp(extended_weights, np.linspace(0, 1, num_transition_points), old_x_remaining)
                old_interp_y = np.interp(extended_weights, np.linspace(0, 1, num_transition_points), old_y_remaining)
                old_interp_z = np.interp(extended_weights, np.linspace(0, 1, num_transition_points), old_z_remaining)
                
                new_interp_x = np.interp(extended_weights, np.linspace(0, 1, num_transition_points), new_x_for_blend)
                new_interp_y = np.interp(extended_weights, np.linspace(0, 1, num_transition_points), new_y_for_blend)
                new_interp_z = np.interp(extended_weights, np.linspace(0, 1, num_transition_points), new_z_for_blend)
                
                # Blend the extended trajectories
                blended_x = (1 - extended_weights) * old_interp_x + extended_weights * new_interp_x
                blended_y = (1 - extended_weights) * old_interp_y + extended_weights * new_interp_y
                blended_z = (1 - extended_weights) * old_interp_z + extended_weights * new_interp_z
                
                # Combine with rest of new trajectory
                smooth_x_des = np.concatenate([blended_x, self.x_des[num_transition_points:]])
                smooth_y_des = np.concatenate([blended_y, self.y_des[num_transition_points:]])
                smooth_z_des = np.concatenate([blended_z, self.z_des[num_transition_points:]])
                
                self.x_des = smooth_x_des
                self.y_des = smooth_y_des
                self.z_des = smooth_z_des
        else:
            print("no smoothing (first calculation)")
        
        
        # Extend trajectory for MPC horizon
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * (2 * self.N + 1)))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * (2 * self.N + 1)))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * (2 * self.N + 1)))

        self.prev_x_des = self.x_des
        self.prev_y_des = self.y_des
        self.prev_z_des = self.z_des

        # Reset trajectory tracking
        self._tick = 0
        self.finished = False

        end = time.time()
        print("trajectory updated!")
        print("Execution time:", (end - start) * 1e3, "ms")
        return reduced_3d_path
    
    def get_trajectory_and_mpc_horizon(self) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve the full trajectory and the MPC horizon for drawing in sim.py.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the full trajectory and the MPC horizon.
        """
        return self.full_traj, self.mpc_horizon

    