"""
workspace_analysis.py
=====================
Trova il punto ottimale di handover tra i due robot Panda.

Usa le funzioni helper di visualize.py:
- create_meshcat()
- create_scene()
- add_visualization()

Esegui per generare handover_config.py:
    python workspace_analysis.py
"""

import numpy as np
import time
from itertools import product

from pydrake.geometry import Rgba, Sphere
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.all import InverseKinematics, Solve

from config01 import (
    Q_HOME,
    SIM_TIME_STEP,
    PANDA1_NAME, PANDA2_NAME,
    HANDOVER_X_RANGE, HANDOVER_Y_RANGE, HANDOVER_Z_RANGE, HANDOVER_STEP,
    GRIPPER_OFFSET_Y,
    P1_GRIPPER_RPY, P2_GRIPPER_RPY,
    JOINT_LOWER, JOINT_UPPER,
    HANDOVER_CONFIG_FILE,
    HANDOVER_POINT_RADIUS, BEST_HANDOVER_POINT_RADIUS,
    HANDOVER_POINT_ALPHA, BEST_HANDOVER_POINT_ALPHA,
)
from visualize import create_meshcat, create_scene, add_visualization


# ============================================================
# SETUP
# ============================================================

def setup_scene():
    """
    Crea la scena usando le funzioni di visualize.py
    """
    meshcat = create_meshcat()
    
    builder = DiagramBuilder()
    plant, scene_graph = create_scene(builder, SIM_TIME_STEP)
    
    panda1 = plant.GetModelInstanceByName(PANDA1_NAME)
    panda2 = plant.GetModelInstanceByName(PANDA2_NAME)
    
    plant.SetDefaultPositions(panda1, Q_HOME)
    plant.SetDefaultPositions(panda2, Q_HOME)
    
    add_visualization(builder, meshcat)
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.Initialize()
    diagram_context = simulator.get_mutable_context()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    
    # Context separato per IK
    ik_context = plant.CreateDefaultContext()
    plant.SetPositions(ik_context, panda1, Q_HOME)
    plant.SetPositions(ik_context, panda2, Q_HOME)
    
    return {
        'plant': plant,
        'diagram': diagram,
        'diagram_context': diagram_context,
        'plant_context': plant_context,
        'ik_context': ik_context,
        'panda1': panda1,
        'panda2': panda2,
        'meshcat': meshcat,
    }


# ============================================================
# IK
# ============================================================

def solve_ik(plant, context, robot_model, position, rpy):
    """Inverse Kinematics - basato su tutorial_04_ik.py"""
    frame_E = plant.GetFrameByName("panda_hand", robot_model)
    
    X_desired = RigidTransform(
        RotationMatrix(RollPitchYaw(rpy[0], rpy[1], rpy[2])),
        position
    )
    
    ik = InverseKinematics(plant, context)
    prog = ik.prog()
    q_vars = ik.q()
    
    p = X_desired.translation().reshape((3, 1))
    ik.AddPositionConstraint(
        frameB=frame_E,
        p_BQ=np.zeros((3, 1)),
        frameA=plant.world_frame(),
        p_AQ_lower=p,
        p_AQ_upper=p
    )
    
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=X_desired.rotation(),
        frameBbar=frame_E,
        R_BbarB=RotationMatrix(),
        theta_bound=0.01
    )
    
    prog.AddBoundingBoxConstraint(
        plant.GetPositionLowerLimits(),
        plant.GetPositionUpperLimits(),
        q_vars
    )
    
    q_nom = plant.GetPositions(context).reshape((-1, 1))
    prog.AddQuadraticErrorCost(np.eye(len(q_nom)), q_nom, q_vars)
    
    result = Solve(prog, q_nom)
    
    if result.is_success():
        return plant.GetPositionsFromArray(robot_model, result.GetSolution(q_vars))
    return None


def calc_score(q1, q2):
    """Score = margine minimo dai joint limits."""
    q1_arm = np.array(q1)[:7]
    q2_arm = np.array(q2)[:7]
    
    margin1 = np.minimum(q1_arm - JOINT_LOWER, JOINT_UPPER - q1_arm)
    margin2 = np.minimum(q2_arm - JOINT_LOWER, JOINT_UPPER - q2_arm)
    
    return float(min(np.min(margin1), np.min(margin2)))


# ============================================================
# RICERCA
# ============================================================

def find_valid_points(scene):
    """Testa la griglia e trova i punti raggiungibili."""
    plant = scene['plant']
    ik_context = scene['ik_context']
    panda1 = scene['panda1']
    panda2 = scene['panda2']
    
    x_vals = np.arange(HANDOVER_X_RANGE[0], HANDOVER_X_RANGE[1] + HANDOVER_STEP, HANDOVER_STEP)
    y_vals = np.arange(HANDOVER_Y_RANGE[0], HANDOVER_Y_RANGE[1] + HANDOVER_STEP, HANDOVER_STEP)
    z_vals = np.arange(HANDOVER_Z_RANGE[0], HANDOVER_Z_RANGE[1] + HANDOVER_STEP, HANDOVER_STEP)
    grid = list(product(x_vals, y_vals, z_vals))
    
    print(f"Testing {len(grid)} punti...")
    
    valid = []
    
    for i, (x, y, z) in enumerate(grid):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(grid)}")
        
        p1 = [x, y + GRIPPER_OFFSET_Y, z]
        p2 = [x, y - GRIPPER_OFFSET_Y, z]
        
        plant.SetPositions(ik_context, panda1, Q_HOME)
        plant.SetPositions(ik_context, panda2, Q_HOME)
        
        q1 = solve_ik(plant, ik_context, panda1, p1, P1_GRIPPER_RPY)
        if q1 is None:
            continue
        
        plant.SetPositions(ik_context, panda1, q1)
        
        q2 = solve_ik(plant, ik_context, panda2, p2, P2_GRIPPER_RPY)
        if q2 is None:
            continue
        
        score = calc_score(q1, q2)
        valid.append({
            'center': [x, y, z],
            'p1': p1, 'p2': p2,
            'q1': q1, 'q2': q2,
            'score': score,
        })
    
    return valid


# ============================================================
# VISUALIZZAZIONE
# ============================================================

def visualize_points(meshcat, points, scores) : 
    """Visualizza punti colorati per score."""
    if not points:
        return
    
    min_s, max_s = min(scores), max(scores)
    range_s = max_s - min_s if max_s > min_s else 1.0
    
    for i, (point, score) in enumerate(zip(points, scores)):
        norm = (score - min_s) / range_s
        color = Rgba(1 - norm, norm, 0, HANDOVER_POINT_ALPHA)
        meshcat.SetObject(f"points/{i}", Sphere(HANDOVER_POINT_RADIUS), color)
        meshcat.SetTransform(f"points/{i}", RigidTransform(point))


def visualize_best_point(meshcat, point):
    """Visualizza il punto migliore in blu."""
    color = Rgba(0, 0.5, 1, BEST_HANDOVER_POINT_ALPHA)
    meshcat.SetObject("points/BEST", Sphere(BEST_HANDOVER_POINT_RADIUS), color)
    meshcat.SetTransform("points/BEST", RigidTransform(point))


def show_robot_config(scene, q1, q2):
    """Mostra i robot in una configurazione."""
    plant = scene['plant']
    plant_context = scene['plant_context']
    
    plant.SetPositions(plant_context, scene['panda1'], q1)
    plant.SetPositions(plant_context, scene['panda2'], q2)
    scene['diagram'].ForcedPublish(scene['diagram_context'])


# ============================================================
# SALVATAGGIO
# ============================================================

def save_config(best):
    """Salva in handover_config.py"""
    content = f'''"""
handover_config.py
==================
AUTO-GENERATO da workspace_analysis.py
"""

import numpy as np

HANDOVER_CENTER = {list(np.round(best['center'], 4))}
HANDOVER_P1 = {list(np.round(best['p1'], 4))}
HANDOVER_P2 = {list(np.round(best['p2'], 4))}
HANDOVER_Q1 = {list(np.round(best['q1'], 4))}
HANDOVER_Q2 = {list(np.round(best['q2'], 4))}
HANDOVER_SCORE = {round(best['score'], 4)}
'''
    with open(HANDOVER_CONFIG_FILE, 'w') as f:
        f.write(content)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 50)
    print("WORKSPACE ANALYSIS")
    print("=" * 50)
    
    print("\n[1/4] Setup...")
    scene = setup_scene()
    print("🌐 http://localhost:7001")
    time.sleep(1)
    
    print("\n[2/4] Ricerca punti validi...")
    valid = find_valid_points(scene)
    print(f" Trovati {len(valid)} punti")
    
    if not valid:
        print("Nessun punto trovato!")
        return
    
    print("\n[3/4] Selezione migliore...")
    valid.sort(key=lambda x: x['score'], reverse=True)
    best = valid[0]
    print(f"🏆 Migliore: {np.round(best['center'], 3)}")
    print(f"   Score: {best['score']:.4f}")
    
    print("\n[4/4] Visualizzazione...")
    meshcat = scene['meshcat']
    #visualize_points(meshcat, [v['center'] for v in valid], [v['score'] for v in valid])
    visualize_best_point(meshcat, best['center'])
    show_robot_config(scene, best['q1'], best['q2'])
    
    save_config(best)
    print(f"\n💾 Salvato: {HANDOVER_CONFIG_FILE}")
    
    print("\n" + "=" * 50)
    print("✅ COMPLETATO!")
    print("=" * 50)
    
    input("\nENTER per uscire...")


if __name__ == "__main__":
    main()