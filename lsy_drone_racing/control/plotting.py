import os

import itertools
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
        self.fig_2d, self.ax_2d = plt.subplots() # the 2d trajectory plot 
        self.line_actual = None
        self.scatter_waypoints = None 
        self.projection = 'xy'
        self.save = True
        self.axis_map = {'x': 0, 'y': 1, 'z': 2}
        self.drone_positions = []  # to store (x, y) of every position  
        self.drone_dots = None     # the scatter object for all dots
        self.trajectory_count = 0
        self.color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


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

        Args:plt.subplots()
            waypoints: Waypoints used for planning.
            trajectory: The planned trajectory.
            projection: Which 2D plane to project onto ('xy', 'xz', or 'yz').
            show_plot: whether the plot should be shown live
            config: The race configuration, see the config files for details, with nominal gate and obstacle positionsgates_rpy = np.array([gate['rpy'] for gate in gates])
            show_gates: Whether to plot gate positions
            show_obstacles: Whether to plot the nominal obstacle positions
            save: Whether to save the plot to a file
        """

        ax = self.ax_2d
        #ax.clear()

        self.projection = projection
        if projection not in ["xy", "xz", "yz"]:
            raise ValueError("Projection must be one of 'xy', 'xz', or 'yz'.")

        i, j = self.axis_map[projection[0]], self.axis_map[projection[1]]

        if config is not None:
            # plot gates
            if show_gates:
                gates = config["env.track.gates"]
                gates_pos = np.array([gate['pos'] for gate in gates])
                gates_rpy = np.array([gate['rpy'] for gate in gates])        
                ax.scatter(gates_pos[:,i], gates_pos[:,j], color='grey', label='Gates')

                length = 0.3  # total gate length
                half_len = length / 2

                if projection == "xy":
                    for pos, rpy in zip(gates_pos, gates_rpy):
                        yaw = rpy[2]
                        dx = np.cos(yaw) * half_len
                        dy = np.sin(yaw) * half_len

                        x0, y0 = pos[0] - dx, pos[1] - dy
                        x1, y1 = pos[0] + dx, pos[1] + dy

                        ax.plot([x0, x1], [y0, y1], color='grey', linewidth=2)

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
        #self.fig_2d.canvas.draw()
        #self.fig_2d.canvas.flush_events()
        plt.ion()
        #plt.show(block=False)

        if show_plot:
            plt.show()

        # save into a PNG
        self.save = save
        if save:
            filename = f"{projection}_trajectory.png"
            filepath = os.path.join(self.save_dir, filename)
            self.fig_2d.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")


    def plot_drone_pos(self, 
            tick:int,
            freq:int=20, 
            obs: dict[str, NDArray[np.floating]]=None
            ):
        """Given a position of the drone, new waypoints, or a new trajectory, updates the trajectory plot.

        Args:
            tick: current tick
            frequency: how often to plot the drone position
            obs: observation of the environment's state (gates, obstacles positions, drone positions)
        """

        i, j = self.axis_map[self.projection[0]], self.axis_map[self.projection[1]]

        if obs is not None:
            if tick % freq == 0:
                # plot current drone position
                current_pos = obs["pos"]
                self.drone_positions.append([current_pos[i], current_pos[j]])

                pos_array = np.array(self.drone_positions)
        
                # Clear the old scatter only once, then replot all positions
                if self.drone_dots is not None:
                    self.drone_dots.remove()
            
                self.drone_dots = self.ax_2d.scatter(
                    pos_array[:, 0], pos_array[:, 1], marker='.', s=8, color='darkblue', label='Drone positions'
                )

                self.fig_2d.canvas.draw_idle()
                plt.pause(0.001)
                print(f'Drone position @ {current_pos} graphed.')

        # save into a PNG
        if self.save and tick % freq*10 == 0: 
            filename = f"{self.projection}_trajectory.png"
            filepath = os.path.join(self.save_dir, filename)
            self.fig_2d.savefig(filepath, dpi=100, bbox_inches='tight')
            print(f"Updated plot saved to {filepath}")
            

    def update_2d_trajectory(self,
        waypoints:NDArray[np.floating]=None, 
        trajectory:NDArray[np.floating]=None, 
        obs: dict[str, NDArray[np.floating]]=None,
        remove_previous:bool=False
        ):
        """
        Given new waypoints, or a new trajectory, updates the trajectory plot.

        Args:
            waypoints: (updated) waypoints used for planning.
            trajectory: a (re)planned trajectory.
            obs: observation of the environment's state (gates, obstacles positions)
            remove_previous: should the previous planned trajectory be removed
        """
        ax = self.ax_2d

        i, j = self.axis_map[self.projection[0]], self.axis_map[self.projection[1]]

        if waypoints is not None:
            self.scatter_waypoints = ax.scatter(waypoints[:, i], waypoints[:, j], color='red', label='Updated Waypoints')
        
        if trajectory is not None:
            self.trajectory_count += 1 
            color = next(self.color_cycle)
            label = f'Replanned Trajectory #{self.trajectory_count}'
            if remove_previous:
                self.line_actual.remove()
            self.line_actual, = ax.plot(trajectory[:, i], trajectory[:, j], '-', color=color, label=label)


    def plot_position_error(self, trajectory, obs):
        # ||actual - desired|| over time?
        pass

