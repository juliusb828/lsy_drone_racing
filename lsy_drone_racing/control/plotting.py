import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class Plotting:
    """Plotting class."""

    def __init__(
            self,
            save_dir:str="lsy_drone_racing/control/plots"
            )->None:
        """Initialization of the plotter.

        Args:
            save_dir: directory in which the plot should be saved. 
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True) # create folder for plots if it doesn't exist
        self.fig_2d, self.ax_2d = None, None # the 2d trajectory plot 
        self.line_actual = None
        self.scatter_waypoints = None 


    def plot_2d_trajectory(
            self,
            waypoints:NDArray[np.floating], 
            trajectory:NDArray[np.floating],
            projection:str='xy', 
            show_plot:bool = True, 
            config: dict=None,
            show_gates:bool=True, 
            show_obstacles:bool=True,
            save:bool=True):
        """Plots the planned 2D trajectory.

        Args:
            waypoints: Waypoints used for planning.
            trajectory: The planned trajectory.
            projection: Which 2D plane to project onto ('xy', 'xz', or 'yz').
            show_plot: whether the plot should be shown live
            config: The race configuration, see the config files for details, with nominal gate and obstacle positions
            show_gates: Whether to plot gate positions
            show_obstacles: Whether to plot the nominal obstacle positions
            save: Whether to save the plot to a file
        """
        if self.fig_2d is None or self.ax_2d is None:
            self.fig_2d, self.ax_2d = plt.subplots()

        ax = self.ax_2d
        ax.clear()

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if projection not in ["xy", "xz", "yz"]:
            raise ValueError("Projection must be one of 'xy', 'xz', or 'yz'.")

        i, j = axis_map[projection[0]], axis_map[projection[1]]

        if config is not None:
            # plot gates
            if show_gates:
                gates = config["env.track.gates"]
                gates_pos = np.array([gate['pos'] for gate in gates])
                ax.scatter(gates_pos[:,0], gates_pos[:,1], color='green', label='Gates')
            # plot obstacles
            if show_obstacles:
                obstacles = config["env.track.obstacles"]
                obstacles_pos = np.array([obstacle['pos'] for obstacle in obstacles])
                ax.scatter(obstacles_pos[:, i], obstacles_pos[:, j], color='black', marker='x', label='Obstacles')

        if waypoints is not None:
            self.scatter_waypoints = ax.scatter(waypoints[:, i], waypoints[:, j], color='red', label='Waypoints')
        if trajectory is not None:
            self.line_actual, = ax.plot(trajectory[:, i], trajectory[:, j], 'b-', label='Trajectory')

        #ax.plot([0, 1, 2], [0, 1, 0], 'g--', label='Debug Line')
        ax.set_xlabel(projection[0].upper())
        ax.set_ylabel(projection[1].upper())
        ax.set_title(f"{projection.upper()} Trajectory")
        ax.legend()
        ax.axis('equal')
        plt.grid(True)

        if show_plot:
            plt.show()

        # save into a PNG
        if save:
            filename = f"{projection}_trajectory.png"
            filepath = os.path.join(self.save_dir, filename)
            self.fig_2d.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")


    def update_2d_trajectory(self):
        """Given a position of the drone, new waypoints, or a new trajectory, updates the trajectory plot."""
        pass

    def plot_position_error(self, trajectory, obs):
        # ||actual - desired|| over time?
        pass

