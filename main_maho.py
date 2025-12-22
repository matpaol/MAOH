"""
main.py
=======
File principale per eseguire il task di handover multi-robot.

Integra tutti i moduli:
- config01.py: configurazione
- visualize.py: setup scena
- controller.py: PD+G controller
- ik_solver.py: inverse kinematics
- handover_config.py: punti handover pre-calcolati
- gripper_manager.py: gestione gripper
- state_machine.py: coordinamento task

Esegui:
    python main.py
"""

import numpy as np

from pydrake.systems.framework import DiagramBuilder, LeafSystem, BasicVector
from pydrake.systems.analysis import Simulator

# ============================================================
# IMPORT MODULI PROGETTO
# ============================================================

from config01 import (
    SIM_TIME_STEP,
    PANDA1_NAME, PANDA2_NAME,
    Q_HOME,
    P1_GRIPPER_RPY, P2_GRIPPER_RPY,
)

from visualize import create_meshcat, create_scene, add_visualization
from controller import Controller
from gripper_manager import DualGripperManager
from state_machine import HandoverStateMachine, compute_ik_configs

from handover_config import (
    HANDOVER_Q1, HANDOVER_Q2,
    HANDOVER_CENTER,
)


# ============================================================
# CONFIGURAZIONE TASK
# ============================================================

# Posizione cubo iniziale (dal SDF)
CUBE_INITIAL_POS = [0.5, 0.5, 0.55]

# Posizione target (tavolo Robot 2)
CUBE_TARGET_POS = [0.5, -0.5, 0.52]

# Orientamento gripper R2 per place (diverso da handover)
P2_PLACE_RPY = [np.pi, 0, np.pi]

# Parametri geometrici
APPROACH_HEIGHT = 0.12
GRASP_Z_OFFSET = 0.04

# Simulazione
SIM_DURATION = 30.0
SIM_DT = 0.01


# ============================================================
# TARGET SOURCE (fornisce target dalla state machine ai controller)
# ============================================================

class TargetSource(LeafSystem):
    """
    LeafSystem che fornisce i target dalla state machine ai controller.
    
    Legge i target dalla HandoverStateMachine e li espone come output port.
    """
    
    def __init__(self, state_machine, robot_id):
        """
        Args:
            state_machine: HandoverStateMachine instance
            robot_id: 1 per Robot 1, 2 per Robot 2
        """
        super().__init__()
        self.sm = state_machine
        self.robot_id = robot_id
        
        self.DeclareVectorOutputPort(
            "target",
            BasicVector(9),
            self._output_target
        )
    
    def _output_target(self, context, output):
        """Legge il target corrente dalla state machine."""
        q1, q2 = self.sm.get_targets()
        if self.robot_id == 1:
            output.SetFromVector(q1)
        else:
            output.SetFromVector(q2)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("       MULTI-AGENT HANDOVER")
    print("=" * 60)
    
    # ===== FASE 1: Setup Scena =====
    print("\n[1/5] Setup scena...")
    meshcat = create_meshcat()
    
    builder = DiagramBuilder()
    plant, scene_graph = create_scene(builder, SIM_TIME_STEP)
    add_visualization(builder, meshcat)
    
    # Ottieni riferimenti
    panda1 = plant.GetModelInstanceByName(PANDA1_NAME)
    panda2 = plant.GetModelInstanceByName(PANDA2_NAME)
    cube_model = plant.GetModelInstanceByName("cube")
    cube_body = plant.GetBodyByName("link", cube_model)
    
    # Posizioni iniziali
    plant.SetDefaultPositions(panda1, Q_HOME)
    plant.SetDefaultPositions(panda2, Q_HOME)
    
    print(f"    ✅ Scena caricata")
    print(f"    🌐 Meshcat: http://localhost:7001")
    
    # ===== FASE 2: Calcola IK =====
    print("\n[2/5] Calcolo configurazioni IK...")
    
    # Crea context temporaneo per IK
    temp_diagram = builder.Build()
    temp_context = temp_diagram.CreateDefaultContext()
    temp_plant_context = temp_diagram.GetMutableSubsystemContext(plant, temp_context)
    
    # Calcola tutte le configurazioni
    ik_configs = compute_ik_configs(
        plant=plant,
        context=temp_plant_context,
        panda1=panda1,
        panda2=panda2,
        cube_pos=CUBE_INITIAL_POS,
        place_pos=CUBE_TARGET_POS,
        handover_q1=HANDOVER_Q1,
        handover_q2=HANDOVER_Q2,
        gripper_rpy_r1=P1_GRIPPER_RPY,
        gripper_rpy_r2=P2_GRIPPER_RPY,
        place_rpy_r2=P2_PLACE_RPY,
        approach_height=APPROACH_HEIGHT,
        grasp_z_offset=GRASP_Z_OFFSET,
    )
    
    # ===== FASE 3: Crea Gripper Manager e State Machine =====
    print("[3/5] Creazione gripper manager e state machine...")
    
    # NOTA: Dobbiamo ricreare il diagram perché l'abbiamo già buildato
    builder2 = DiagramBuilder()
    plant2, scene_graph2 = create_scene(builder2, SIM_TIME_STEP)
    add_visualization(builder2, meshcat)
    
    panda1 = plant2.GetModelInstanceByName(PANDA1_NAME)
    panda2 = plant2.GetModelInstanceByName(PANDA2_NAME)
    cube_model = plant2.GetModelInstanceByName("cube")
    cube_body = plant2.GetBodyByName("link", cube_model)
    
    plant2.SetDefaultPositions(panda1, Q_HOME)
    plant2.SetDefaultPositions(panda2, Q_HOME)
    
    # Crea gripper manager (lo useremo dopo con il context reale)
    gripper_manager = DualGripperManager(plant2, panda1, panda2)
    
    # Crea state machine
    state_machine = HandoverStateMachine(gripper_manager, ik_configs)
    
    print(f"    ✅ State machine creata")
    
    # ===== FASE 4: Crea Controller e Connessioni =====
    print("\n[4/5] Costruzione sistema di controllo...")
    
    # Controller
    ctrl1 = builder2.AddNamedSystem("Controller_R1", Controller(plant2, panda1))
    ctrl2 = builder2.AddNamedSystem("Controller_R2", Controller(plant2, panda2))
    
    # Target sources (leggono dalla state machine)
    target1 = builder2.AddNamedSystem("Target_R1", TargetSource(state_machine, 1))
    target2 = builder2.AddNamedSystem("Target_R2", TargetSource(state_machine, 2))
    
    # Connessioni Robot 1
    builder2.Connect(plant2.GetOutputPort("panda_1_state"), 
                    ctrl1.GetInputPort("Current_state"))
    builder2.Connect(ctrl1.GetOutputPort("tau_u"), 
                    plant2.GetInputPort("panda_1_actuation"))
    builder2.Connect(target1.get_output_port(), 
                    ctrl1.GetInputPort("Desired_state"))
    
    # Connessioni Robot 2
    builder2.Connect(plant2.GetOutputPort("panda_2_state"), 
                    ctrl2.GetInputPort("Current_state"))
    builder2.Connect(ctrl2.GetOutputPort("tau_u"), 
                    plant2.GetInputPort("panda_2_actuation"))
    builder2.Connect(target2.get_output_port(), 
                    ctrl2.GetInputPort("Desired_state"))
    
    # Build diagram
    diagram = builder2.Build()
    print(f"    ✅ Diagram costruito")
    
    # ===== FASE 5: Simulazione =====
    print("\n[5/5] Simulazione...")
    
    simulator = Simulator(diagram)
    sim_context = simulator.get_mutable_context()
    plant_context = diagram.GetMutableSubsystemContext(plant2, sim_context)
    
    # Apri gripper all'inizio
    gripper_manager.open_all(plant_context)
    
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    
    print("\n" + "=" * 60)
    print("   ▶️  SIMULAZIONE IN CORSO")
    print("   🌐 Apri http://localhost:7001 per vedere")
    print("=" * 60 + "\n")
    
    meshcat.StartRecording()
    
    # Loop di simulazione
    t = 0.0
    while t < SIM_DURATION:
        # Aggiorna state machine
        done = state_machine.update(t, plant_context, cube_body)
        
        if done:
            print(f"\n✅ Task completato a t = {t:.1f}s")
            # Continua un po' per vedere il risultato
            simulator.AdvanceTo(t + 2.0)
            break
        
        # Avanza simulazione
        try:
            simulator.AdvanceTo(t + SIM_DT)
        except Exception as e:
            print(f"⚠️  Errore simulazione a t={t:.1f}s: {e}")
            break
        
        t += SIM_DT
    
    meshcat.PublishRecording()
    
    # ===== RISULTATO =====
    print("\n" + "=" * 60)
    print("   RISULTATO")
    print("=" * 60)
    
    # Posizione finale del cubo
    cube_pose = plant2.GetFreeBodyPose(plant_context, cube_body)
    cube_final_pos = cube_pose.translation()
    
    print(f"\n   Posizione iniziale cubo: {CUBE_INITIAL_POS}")
    print(f"   Posizione target:        {CUBE_TARGET_POS}")
    print(f"   Posizione finale:        [{cube_final_pos[0]:.3f}, {cube_final_pos[1]:.3f}, {cube_final_pos[2]:.3f}]")
    
    # Calcola errore
    error = np.linalg.norm(np.array(cube_final_pos[:2]) - np.array(CUBE_TARGET_POS[:2]))
    print(f"\n   Errore XY: {error:.3f} m")
    
    if error < 0.1:
        print("\n   🎉 HANDOVER RIUSCITO!")
    else:
        print("\n   ⚠️  Handover non completato correttamente")
    
    print("\n" + "=" * 60)
    print("   ▶️  Premi PLAY in Meshcat per rivedere l'animazione")
    print("=" * 60)
    
    input("\nENTER per uscire...")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
