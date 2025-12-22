"""
config.py
---------
Configurazione globale del progetto.
"""

import os
import numpy as np

# ============== PATH ==============

SDF_FILE_PATH = "/Users/matteopaolini/maho_mp/project_07_object_handover.sdf"
FIGURES_PATH = "/Users/matteopaolini/maho_mp/figures"

# ============== SIMULATION ==============

VISUALIZE_ONLY = False
SIM_TIME_STEP = 0.001
SIM_DURATION = 8.0

# ============== ROBOT ==============

PANDA1_NAME = "panda_1"
PANDA2_NAME = "panda_2"

Q_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]

# Joint limits (7 giunti del braccio)
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# ============== GRIPPER ==============

P1_GRIPPER_RPY = [-np.pi, np.pi/2, 0]
#P2_GRIPPER_RPY = [-np.pi, -np.pi/2, -3*np.pi/2]
P2_GRIPPER_RPY = [np.pi, np.pi/2, np.pi]


GRIPPER_OFFSET_Y = 0.1

# ============== WORKSPACE ANALYSIS ==============

HANDOVER_X_RANGE = (0.1, 0.5)
HANDOVER_Y_RANGE = (-0.2, 0.2)
HANDOVER_Z_RANGE = (0.5, 0.8)
HANDOVER_STEP = 0.05

HANDOVER_CONFIG_FILE = "handover_config.py"

# ============== VISUALIZATION ==============

HANDOVER_POINT_RADIUS = 0.015
BEST_HANDOVER_POINT_RADIUS = 0.03
HANDOVER_POINT_ALPHA = 0.7
BEST_HANDOVER_POINT_ALPHA = 0.9


# ============== TRAJECTORY GENERATOR ==============

# Velocità massima giunti (rad/s)
TRAJ_V_MAX = 0.5

# Accelerazione massima giunti (rad/s²)
TRAJ_A_MAX = 2.0