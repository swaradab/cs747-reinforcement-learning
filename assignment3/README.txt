CONTENTS
--------
* Introduction
* Requirements
* Installation
* Execution
  * Training and generating plots
  * Testing

INTRODUCTION:
-------------

This program will solve the Mountain Car Task in less than 160 timesteps (using linear Sarsa), or 130 timesteps (using Tile Coding with Linear Function Approximation). 
For this, it will carry out both the training and testing phases, depending on the command line arguments passed. 
A description of the Mountain Car problem can be found here: https://gym.openai.com/envs/MountainCar-v0/

REQUIREMENTS:
-------------

This program requires the following modules and libraries to be installed:
gym
pyglet
numpy 
matplotlib
Other than these, it uses Python in-built libraries math, time and argparse.

INSTALLATION:
-------------

To install gym: use package manager pip to install gym by using command 'pip install gym'.
To install pyglet: use package manager pip to install pyglet by using command 'pip install pyglet'.
To install numpy: use package manager pip to install numpy by using command 'pip install numpy'.
To install matplotlib: use package manager pip to install matplotlib by using command 'pip install matplotlib'.

EXECUTION:
----------

To carry out training for Task T1 (Linear Sarsa), use the command: python mountain_car.py --task T1 --train 1

To carry out training for Task T2 (Tile Coding with Linear Function Approximation), use the command: python mountain_car.py --task T2 --train 1

Wait for the training to finish before proceeding. There will be no printed output. The files T1/T2.npy and T1/T2.jpg (depending on the task) will be automatically generated within the same directory as the code file when training is finished. 
The .npy file contains the trained weights, while the .jpg file contains the image of the training plot.
In order to carry out testing, the .npy file produced during training must be present in the same directory as the code file.

To carry out testing for Task T1 (Linear Sarsa), use the command: python mountain_car.py --task T1 --train 0

To carry out testing for Task T2 (Tile Coding with Linear Function Approximation), use the command: python mountain_car.py --task T2 --train 0

After the completion of testing, the reward obtained upon testing will be printed as output. If the reward is above -199, then the Mountain Car problem is successfully solved.