"""
test_handover_demo.py
=====================
Demo completa del movimento di handover tra i due robot.

Mostra la sequenza:
  Robot 1: HOME → APPROACH → PICK → HANDOVER
  Robot 2: HOME → HANDOVER → APPROACH_PLACE → PLACE

Usa:
- ik_solver.py per calcolare le configurazioni
- trajectory_system.py per generare movimenti smooth
- handover_config.py per il punto di scambio (se disponibile)
"""

import numpy as np
import time

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput, Demultiplexer
from pydrake.systems.analysis import Simulator
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw

from config01 import (
    Q_HOME, SIM_TIME_STEP, 
    PANDA1_NAME, PANDA2_NAME,
    P1_GRIPPER_RPY, P2_GRIPPER_RPY,
)
from visualize import create_meshcat, create_scene, add_visualization
from controller import Controller
from traj_generator import MultiSegmentTrajectory
from ik_solver import solve_ik

# Prova a importare handover_config (generato da trovapunti_evo.py)
try:
    from handover_config import HANDOVER_Q1, HANDOVER_Q2, HANDOVER_CENTER
    HANDOVER_CONFIG_AVAILABLE = True
    print("✅ handover_config.py trovato!")
except ImportError:
    HANDOVER_CONFIG_AVAILABLE = False
    print("⚠️  handover_config.py non trovato, uso valori di default")
    # Valori di default per il punto di handover
    HANDOVER_CENTER = [0.3, 0.0, 0.6]
    HANDOVER_Q1 = None
    HANDOVER_Q2 = None


# ============================================================
# CONFIGURAZIONE TASK
# ============================================================

# Posizione cubo (dal SDF)
CUBE_POS = [0.5, 0.5, 0.55]

# Posizione target (tavolo robot 2)
PLACE_POS = [0.4, -0.4, 0.55]  # Leggermente più vicino al robot 2

# Altezza di approach (sopra l'oggetto)
APPROACH_HEIGHT = 0.10  # Ridotto per stare nel workspace

# Velocità e accelerazione
V_MAX = 0.4  # rad/s
A_MAX = 1.5  # rad/s²

# Orientamento alternativo per Robot 2 (gripper verso il basso)
P2_PLACE_RPY = [np.pi, 0, np.pi]  # Funziona meglio per la fase place


# ============================================================
# CALCOLO WAYPOINTS CON IK
# ============================================================

def compute_robot1_waypoints(plant, context, panda1):
    """
    Calcola i waypoints per Robot 1 (pick e vai a handover).
    
    Sequenza: HOME → APPROACH → PICK → LIFT → HANDOVER
    """
    print("\n[Robot 1] Calcolo waypoints...")
    
    waypoints = [np.array(Q_HOME)]
    
    # 1. APPROACH: sopra il cubo
    approach_pos = [CUBE_POS[0], CUBE_POS[1], CUBE_POS[2] + APPROACH_HEIGHT]
    print(f"   Solving IK for APPROACH: {approach_pos}")
    q_approach = solve_ik(plant, context, panda1, approach_pos, P1_GRIPPER_RPY)
    if q_approach is None:
        print("   ❌ IK fallito per APPROACH")
        return None
    waypoints.append(np.array(q_approach))
    print(f"   ✅ APPROACH: {q_approach[:3]}...")
    
    # Reset context per prossimo IK
    plant.SetPositions(context, panda1, q_approach)
    
    # 2. PICK: sul cubo
    pick_pos = [CUBE_POS[0], CUBE_POS[1], CUBE_POS[2] + 0.02]  # +2cm per gripper
    print(f"   Solving IK for PICK: {pick_pos}")
    q_pick = solve_ik(plant, context, panda1, pick_pos, P1_GRIPPER_RPY)
    if q_pick is None:
        print("   ❌ IK fallito per PICK")
        return None
    waypoints.append(np.array(q_pick))
    print(f"   ✅ PICK: {q_pick[:3]}...")
    
    # Reset context
    plant.SetPositions(context, panda1, q_pick)
    
    # 3. LIFT: solleva il cubo
    lift_pos = [CUBE_POS[0], CUBE_POS[1], CUBE_POS[2] + APPROACH_HEIGHT]
    print(f"   Solving IK for LIFT: {lift_pos}")
    q_lift = solve_ik(plant, context, panda1, lift_pos, P1_GRIPPER_RPY)
    if q_lift is None:
        print("   ❌ IK fallito per LIFT")
        return None
    waypoints.append(np.array(q_lift))
    print(f"   ✅ LIFT: {q_lift[:3]}...")
    
    # Reset context
    plant.SetPositions(context, panda1, q_lift)
    
    # 4. HANDOVER: punto di scambio
    if HANDOVER_Q1 is not None:
        q_handover = np.array(HANDOVER_Q1)
        print(f"   ✅ HANDOVER (da config): {q_handover[:3]}...")
    else:
        # Calcola con IK
        handover_p1 = [HANDOVER_CENTER[0], HANDOVER_CENTER[1] + 0.1, HANDOVER_CENTER[2]]
        print(f"   Solving IK for HANDOVER: {handover_p1}")
        q_handover = solve_ik(plant, context, panda1, handover_p1, P1_GRIPPER_RPY)
        if q_handover is None:
            print("   ❌ IK fallito per HANDOVER")
            return None
        print(f"   ✅ HANDOVER: {q_handover[:3]}...")
    waypoints.append(np.array(q_handover))
    
    print(f"   Totale waypoints Robot 1: {len(waypoints)}")
    return np.array(waypoints)


def compute_robot2_waypoints(plant, context, panda2):
    """
    Calcola i waypoints per Robot 2 (ricevi e place).
    
    Sequenza: HOME → HANDOVER → LIFT → APPROACH_PLACE → PLACE
    """
    print("\n[Robot 2] Calcolo waypoints...")
    
    waypoints = [np.array(Q_HOME)]
    
    # 1. HANDOVER: punto di scambio
    if HANDOVER_Q2 is not None:
        q_handover = np.array(HANDOVER_Q2)
        print(f"   ✅ HANDOVER (da config): {q_handover[:3]}...")
    else:
        handover_p2 = [HANDOVER_CENTER[0], HANDOVER_CENTER[1] - 0.1, HANDOVER_CENTER[2]]
        print(f"   Solving IK for HANDOVER: {handover_p2}")
        q_handover = solve_ik(plant, context, panda2, handover_p2, P2_GRIPPER_RPY)
        if q_handover is None:
            print("   ❌ IK fallito per HANDOVER")
            return None
        print(f"   ✅ HANDOVER: {q_handover[:3]}...")
    waypoints.append(np.array(q_handover))
    
    # Reset context
    plant.SetPositions(context, panda2, q_handover)
    
    # 2. LIFT dopo handover - usa orientamento per place
    lift_pos = [HANDOVER_CENTER[0], HANDOVER_CENTER[1] - 0.15, HANDOVER_CENTER[2]]
    print(f"   Solving IK for LIFT: {lift_pos}")
    q_lift = solve_ik(plant, context, panda2, lift_pos, P2_PLACE_RPY)
    if q_lift is None:
        print("   ⚠️  IK fallito per LIFT, provo con altro orientamento...")
        q_lift = solve_ik(plant, context, panda2, lift_pos, P2_GRIPPER_RPY)
        if q_lift is None:
            print("   ⚠️  IK fallito per LIFT, skip")
            q_lift = q_handover
        else:
            waypoints.append(np.array(q_lift))
            print(f"   ✅ LIFT: {q_lift[:3]}...")
    else:
        waypoints.append(np.array(q_lift))
        print(f"   ✅ LIFT: {q_lift[:3]}...")
    
    # Reset context
    plant.SetPositions(context, panda2, q_lift)
    
    # 3. APPROACH PLACE: sopra la posizione target
    approach_place_pos = [PLACE_POS[0], PLACE_POS[1], PLACE_POS[2] + APPROACH_HEIGHT]
    print(f"   Solving IK for APPROACH_PLACE: {approach_place_pos}")
    q_approach_place = solve_ik(plant, context, panda2, approach_place_pos, P2_PLACE_RPY)
    if q_approach_place is None:
        print("   ❌ IK fallito per APPROACH_PLACE")
        return None
    waypoints.append(np.array(q_approach_place))
    print(f"   ✅ APPROACH_PLACE: {q_approach_place[:3]}...")
    
    # Reset context
    plant.SetPositions(context, panda2, q_approach_place)
    
    # 4. PLACE: posizione finale
    place_pos = [PLACE_POS[0], PLACE_POS[1], PLACE_POS[2] + 0.02]
    print(f"   Solving IK for PLACE: {place_pos}")
    q_place = solve_ik(plant, context, panda2, place_pos, P2_PLACE_RPY)
    if q_place is None:
        print("   ❌ IK fallito per PLACE")
        return None
    waypoints.append(np.array(q_place))
    print(f"   ✅ PLACE: {q_place[:3]}...")
    
    print(f"   Totale waypoints Robot 2: {len(waypoints)}")
    return np.array(waypoints)


# ============================================================
# MAIN DEMO
# ============================================================

def run_handover_demo():
    """
    Esegue la demo completa dell'handover.
    """
    print("=" * 60)
    print("       HANDOVER DEMO - MOVIMENTO COMPLETO")
    print("=" * 60)
    
    # ===== FASE 1: Setup =====
    print("\n[FASE 1] Setup scena...")
    meshcat = create_meshcat()
    
    # Creiamo prima un plant temporaneo per calcolare IK
    temp_builder = DiagramBuilder()
    temp_plant, _ = create_scene(temp_builder, SIM_TIME_STEP)
    temp_diagram = temp_builder.Build()
    temp_context = temp_diagram.CreateDefaultContext()
    temp_plant_context = temp_diagram.GetMutableSubsystemContext(temp_plant, temp_context)
    
    panda1 = temp_plant.GetModelInstanceByName(PANDA1_NAME)
    panda2 = temp_plant.GetModelInstanceByName(PANDA2_NAME)
    
    # Imposta posizioni iniziali
    temp_plant.SetPositions(temp_plant_context, panda1, Q_HOME)
    temp_plant.SetPositions(temp_plant_context, panda2, Q_HOME)
    
    print("🌐 Meshcat: http://localhost:7001")
    
    # ===== FASE 2: Calcola Waypoints =====
    print("\n[FASE 2] Calcolo waypoints con IK...")
    
    # Robot 1 waypoints
    wp1 = compute_robot1_waypoints(temp_plant, temp_plant_context, panda1)
    if wp1 is None:
        print("❌ Impossibile calcolare waypoints per Robot 1")
        return
    
    # Reset per Robot 2
    temp_plant.SetPositions(temp_plant_context, panda1, Q_HOME)
    temp_plant.SetPositions(temp_plant_context, panda2, Q_HOME)
    
    # Robot 2 waypoints
    wp2 = compute_robot2_waypoints(temp_plant, temp_plant_context, panda2)
    if wp2 is None:
        print("❌ Impossibile calcolare waypoints per Robot 2")
        return
    
    # ===== FASE 3: Crea Trajectories =====
    print("\n[FASE 3] Creazione traiettorie...")
    
    # Crea trajectory generators
    traj1 = MultiSegmentTrajectory(wp1, v_max=V_MAX, a_max=A_MAX)
    traj2 = MultiSegmentTrajectory(wp2, v_max=V_MAX, a_max=A_MAX)
    
    duration1 = traj1.get_duration()
    duration2 = traj2.get_duration()
    
    print(f"   Robot 1: {len(wp1)} waypoints, durata {duration1:.2f}s")
    print(f"   Robot 2: {len(wp2)} waypoints, durata {duration2:.2f}s")
    
    # ===== FASE 4: Build Diagram =====
    print("\n[FASE 4] Costruzione diagram simulazione...")
    
    builder = DiagramBuilder()
    plant, scene_graph = create_scene(builder, SIM_TIME_STEP)
    add_visualization(builder, meshcat)
    
    panda1 = plant.GetModelInstanceByName(PANDA1_NAME)
    panda2 = plant.GetModelInstanceByName(PANDA2_NAME)
    
    plant.SetDefaultPositions(panda1, Q_HOME)
    plant.SetDefaultPositions(panda2, Q_HOME)
    
    # Controllers
    ctrl1 = builder.AddNamedSystem("Controller1", Controller(plant, panda1))
    ctrl2 = builder.AddNamedSystem("Controller2", Controller(plant, panda2))
    
    # Trajectory systems
    traj_sys1 = builder.AddNamedSystem("Traj1", MultiSegmentTrajectory(wp1, v_max=V_MAX, a_max=A_MAX))
    traj_sys2 = builder.AddNamedSystem("Traj2", MultiSegmentTrajectory(wp2, v_max=V_MAX, a_max=A_MAX))
    
    # Demux (estrae solo q_ref, ignora qd_ref)
    demux1 = builder.AddNamedSystem("Demux1", Demultiplexer([9, 9]))
    demux2 = builder.AddNamedSystem("Demux2", Demultiplexer([9, 9]))
    
    # Connessioni Robot 1
    builder.Connect(traj_sys1.get_output_port(), demux1.get_input_port())
    builder.Connect(demux1.get_output_port(0), ctrl1.GetInputPort("Desired_state"))
    builder.Connect(plant.GetOutputPort("panda_1_state"), ctrl1.GetInputPort("Current_state"))
    builder.Connect(ctrl1.GetOutputPort("tau_u"), plant.GetInputPort("panda_1_actuation"))
    
    # Connessioni Robot 2
    builder.Connect(traj_sys2.get_output_port(), demux2.get_input_port())
    builder.Connect(demux2.get_output_port(0), ctrl2.GetInputPort("Desired_state"))
    builder.Connect(plant.GetOutputPort("panda_2_state"), ctrl2.GetInputPort("Current_state"))
    builder.Connect(ctrl2.GetOutputPort("tau_u"), plant.GetInputPort("panda_2_actuation"))
    
    diagram = builder.Build()
    
    # ===== FASE 5: Simulazione =====
    print("\n[FASE 5] Simulazione...")
    
    sim_duration = max(duration1, duration2) + 2.0
    print(f"   Durata simulazione: {sim_duration:.2f}s")
    print("\n" + "=" * 60)
    print("   ▶️  APRI MESHCAT E GUARDA L'ANIMAZIONE!")
    print("   📍 http://localhost:7001")
    print("=" * 60)
    
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    
    meshcat.StartRecording()
    simulator.AdvanceTo(sim_duration)
    meshcat.PublishRecording()
    
    print("\n✅ Simulazione completata!")
    print("\n📝 Sequenza eseguita:")
    print("   Robot 1: HOME → APPROACH → PICK → LIFT → HANDOVER")
    print("   Robot 2: HOME → HANDOVER → LIFT → APPROACH_PLACE → PLACE")
    
    print("\n" + "=" * 60)
    print("   Premi ▶️ in Meshcat per rivedere l'animazione")
    print("=" * 60)
    
    input("\nENTER per uscire...")


if __name__ == "__main__":
    run_handover_demo()