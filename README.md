# Autonomous Drone Racing Project Course
<p align="center">
  <img width="460" height="300" src="docs/img/banner.jpeg">
</p>
<sub><sup>AI generated image</sup></sub>

[![Python Version]][Python Version URL] [![Ruff Check]][Ruff Check URL] [![Documentation Status]][Documentation Status URL] [![Tests]][Tests URL]

[Python Version]: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue
[Python Version URL]: https://www.python.org

[Ruff Check]: https://github.com/utiasDSL/lsy_drone_racing/actions/workflows/ruff.yml/badge.svg?style=flat-square
[Ruff Check URL]: https://github.com/utiasDSL/lsy_drone_racing/actions/workflows/ruff.yml

[Documentation Status]: https://readthedocs.org/projects/lsy-drone-racing/badge/?version=latest
[Documentation Status URL]: https://lsy-drone-racing.readthedocs.io/en/latest/?badge=latest

[Tests]: https://github.com/utiasDSL/lsy_drone_racing/actions/workflows/testing.yml/badge.svg
[Tests URL]: https://github.com/utiasDSL/lsy_drone_racing/actions/workflows/testing.yml


## Approach
In our approach, we focus primarily on the trajectory generation and use the MPC trajectory tracking provided in 'attitude_mpc.py'. The trajectory is generated as follows: (1) a 2D (xy-plane) cost map is generated, which marks obstacles and sides of gates with a high cost. (2) The 'route_through_array' method is used to find a path through the cost map using points before and after each gate as well as additional waypoints. (3) The path generated from 'route_through_array' is sampled and B-Splines are used to generate a trajectory. 
When recomputing the trajectory, the closest point on the new trajectory to the next point on the old one is chosen to ensure smoothness.


## How to use:
The controller described in our report is found under 'control/mpc_trajectory_following.py'. It uses the attitude interface. This controller works in both sim and real, but struggles with reliability.

Additionally, there is an old version of a controller called 'minsnap_controller.py', which uses minsnap to generate the trajectories. It uses the state interface. This controller does not work reliably in sim and was not tested in real.
