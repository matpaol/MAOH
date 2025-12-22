"""
ik_solver.py
============
Funzioni IK per la state machine.

PREREQUISITO: Eseguire prima workspace_analysis.py per generare handover_config.py

Funzioni:
- get_handover_config(): configurazioni di handover pre-calcolate
- solve_ik(): IK generico
- solve_ik_for_pickup(): IK per afferrare
- solve_ik_for_place(): IK per posare
"""

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.all import InverseKinematics, Solve

from config01 import Q_HOME, P1_GRIPPER_RPY, P2_GRIPPER_RPY

# Import configurazione handover
try:
    from handover_config import (
        HANDOVER_CENTER,
        HANDOVER_P1,
        HANDOVER_P2,
        HANDOVER_Q1,
        HANDOVER_Q2,
    )
    _CONFIG_LOADED = True
except ImportError:
    print("⚠️  handover_config.py non trovato!")
    print("   Esegui prima: python workspace_analysis.py")
    _CONFIG_LOADED = False


def get_handover_config():
    """
    Ritorna la configurazione di handover pre-calcolata.
    
    Returns:
        dict con 'center', 'p1', 'p2', 'q1', 'q2' oppure None
    """
    if not _CONFIG_LOADED:
        return None
    
    return {
        'center': HANDOVER_CENTER,
        'p1': HANDOVER_P1,
        'p2': HANDOVER_P2,
        'q1': HANDOVER_Q1,
        'q2': HANDOVER_Q2,
    }


def solve_ik(plant, context, robot_model, position, rpy):
    """
    Inverse Kinematics generico - basato su tutorial_04_ik.py
    """
    frame_E = plant.GetFrameByName("panda_hand", robot_model)
    
    X_desired = RigidTransform(
        RotationMatrix(RollPitchYaw(rpy[0], rpy[1], rpy[2])),
        position
    )
    
    ik = InverseKinematics(plant, context)
    prog = ik.prog()
    q_vars = ik.q()
    
    # Position
    p = X_desired.translation().reshape((3, 1))
    ik.AddPositionConstraint(
        frameB=frame_E,
        p_BQ=np.zeros((3, 1)),
        frameA=plant.world_frame(),
        p_AQ_lower=p,
        p_AQ_upper=p
    )
    
    # Orientation
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=X_desired.rotation(),
        frameBbar=frame_E,
        R_BbarB=RotationMatrix(),
        theta_bound=0.01
    )
    
    # Limits
    prog.AddBoundingBoxConstraint(
        plant.GetPositionLowerLimits(),
        plant.GetPositionUpperLimits(),
        q_vars
    )
    
    # Cost
    q_nom = plant.GetPositions(context).reshape((-1, 1))
    prog.AddQuadraticErrorCost(np.eye(len(q_nom)), q_nom, q_vars)
    
    result = Solve(prog, q_nom)
    
    if result.is_success():
        return plant.GetPositionsFromArray(robot_model, result.GetSolution(q_vars))
    return None


def solve_ik_for_pickup(plant, context, robot_model, cube_position, approach_height=0.1):
    """
    IK per afferrare il cubo.
    
    Returns:
        dict con 'approach_q' e 'grasp_q' oppure None
    """
    robot_name = plant.GetModelInstanceName(robot_model)
    rpy = P1_GRIPPER_RPY if "panda_1" in robot_name else P2_GRIPPER_RPY
    
    # Approach (sopra il cubo)
    approach_pos = [cube_position[0], cube_position[1], cube_position[2] + approach_height]
    approach_q = solve_ik(plant, context, robot_model, approach_pos, rpy)
    if approach_q is None:
        return None
    
    plant.SetPositions(context, robot_model, approach_q)
    
    # Grasp (sul cubo)
    grasp_q = solve_ik(plant, context, robot_model, cube_position, rpy)
    if grasp_q is None:
        return None
    
    return {'approach_q': approach_q, 'grasp_q': grasp_q}


def solve_ik_for_place(plant, context, robot_model, target_position, approach_height=0.1):
    """
    IK per posare il cubo.
    
    Returns:
        dict con 'approach_q' e 'place_q' oppure None
    """
    robot_name = plant.GetModelInstanceName(robot_model)
    rpy = P1_GRIPPER_RPY if "panda_1" in robot_name else P2_GRIPPER_RPY
    
    # Approach
    approach_pos = [target_position[0], target_position[1], target_position[2] + approach_height]
    approach_q = solve_ik(plant, context, robot_model, approach_pos, rpy)
    if approach_q is None:
        return None
    
    plant.SetPositions(context, robot_model, approach_q)
    
    # Place
    place_q = solve_ik(plant, context, robot_model, target_position, rpy)
    if place_q is None:
        return None
    
    return {'approach_q': approach_q, 'place_q': place_q}