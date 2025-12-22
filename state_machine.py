"""
state_machine.py
================
Macchina a stati per coordinare il task di handover.

Stati:
    IDLE → R1_APPROACH → R1_GRASP → R1_CLOSE_GRIPPER → R1_LIFT → 
    R1_TO_HANDOVER → R2_TO_HANDOVER → HANDOVER_TRANSFER → 
    R1_RETREAT → R2_LIFT → R2_APPROACH_PLACE → R2_PLACE → 
    R2_OPEN_GRIPPER → R2_RETREAT → DONE

Uso:
    from state_machine import HandoverStateMachine, State
    from gripper_manager import DualGripperManager
    
    # Setup
    gripper_mgr = DualGripperManager(plant, panda1, panda2)
    sm = HandoverStateMachine(gripper_mgr, ik_configs)
    
    # Nel loop di simulazione:
    done = sm.update(t, plant_context, cube_body)
    q1_target, q2_target = sm.get_targets()
"""

import numpy as np
from enum import Enum

from config01 import Q_HOME


# ============================================================
# STATI
# ============================================================

class State(Enum):
    """Stati della macchina a stati per l'handover."""
    IDLE = 0
    
    # Robot 1 - Pick
    R1_APPROACH = 1
    R1_GRASP = 2
    R1_CLOSE_GRIPPER = 3
    R1_LIFT = 4
    R1_TO_HANDOVER = 5
    
    # Handover
    R2_TO_HANDOVER = 6
    HANDOVER_TRANSFER = 7
    
    # Robot 2 - Place
    R1_RETREAT = 8
    R2_LIFT = 9
    R2_APPROACH_PLACE = 10
    R2_PLACE = 11
    R2_OPEN_GRIPPER = 12
    R2_RETREAT = 13
    
    DONE = 14


# ============================================================
# CONFIGURAZIONE TIMING
# ============================================================

PHASE_DURATION = 2.5    # Secondi per movimento
GRIPPER_WAIT = 0.5      # Secondi per gripper open/close


# ============================================================
# STATE MACHINE
# ============================================================

class HandoverStateMachine:
    """
    Macchina a stati per il task di handover.
    
    Gestisce:
    - Transizioni tra stati basate sul tempo
    - Target joint per i controller di entrambi i robot
    - Comandi gripper (attach/detach cubo)
    
    Attributes:
        state: Stato corrente (State enum)
        gripper: DualGripperManager per controllare i gripper
        configs: Dict con configurazioni IK pre-calcolate
    """
    
    def __init__(self, gripper_manager, ik_configs):
        """
        Args:
            gripper_manager: DualGripperManager instance
            ik_configs: dict con le configurazioni IK pre-calcolate:
                {
                    'r1_approach': np.array (9,),
                    'r1_grasp': np.array (9,),
                    'r1_lift': np.array (9,),
                    'r1_handover': np.array (9,),
                    'r2_handover': np.array (9,),
                    'r2_lift': np.array (9,),
                    'r2_approach_place': np.array (9,),
                    'r2_place': np.array (9,),
                }
        """
        self.gripper = gripper_manager
        self.configs = ik_configs
        
        # Stato
        self.state = State.IDLE
        self.state_start_time = 0.0
        
        # Target correnti per i controller
        self.q_target_r1 = np.array(Q_HOME)
        self.q_target_r2 = np.array(Q_HOME)
        
        # Flag per azioni one-shot (evita ripetizioni)
        self._action_done = False
        
    def get_state(self):
        """Ritorna lo stato corrente."""
        return self.state
    
    def get_state_name(self):
        """Ritorna il nome dello stato corrente."""
        return self.state.name
    
    def get_targets(self):
        """
        Ritorna i target correnti per i controller.
        
        Returns:
            tuple: (q_target_r1, q_target_r2) entrambi np.array (9,)
        """
        return self.q_target_r1.copy(), self.q_target_r2.copy()
    
    def is_done(self):
        """Ritorna True se il task è completato."""
        return self.state == State.DONE
    
    def update(self, t, context, cube_body=None):
        """
        Aggiorna la state machine.
        
        Chiamare ad ogni step di simulazione.
        
        Args:
            t: tempo simulazione corrente (secondi)
            context: plant context per gripper
            cube_body: Body del cubo (serve per attach al primo grasp)
            
        Returns:
            bool: True se il task è completato
        """
        # Aggiorna posizione cubo se attaccato a un gripper
        self.gripper.update(context)
        
        # Tempo nella fase corrente
        phase_time = t - self.state_start_time
        
        # ===== IDLE =====
        if self.state == State.IDLE:
            self._transition(State.R1_APPROACH, t)
            
        # ===== R1: PICK SEQUENCE =====
        elif self.state == State.R1_APPROACH:
            self.q_target_r1 = np.array(self.configs['r1_approach'])
            if phase_time > PHASE_DURATION:
                self._transition(State.R1_GRASP, t)
                
        elif self.state == State.R1_GRASP:
            self.q_target_r1 = np.array(self.configs['r1_grasp'])
            if phase_time > PHASE_DURATION:
                self._transition(State.R1_CLOSE_GRIPPER, t)
                
        elif self.state == State.R1_CLOSE_GRIPPER:
            # Azione one-shot: attacca il cubo
            if not self._action_done and cube_body is not None:
                self.gripper.gripper1.attach(context, cube_body)
                print(f"    📦 Cubo attaccato a R1")
                self._action_done = True
            if phase_time > GRIPPER_WAIT:
                self._transition(State.R1_LIFT, t)
                
        elif self.state == State.R1_LIFT:
            self.q_target_r1 = np.array(self.configs['r1_lift'])
            if phase_time > PHASE_DURATION:
                self._transition(State.R1_TO_HANDOVER, t)
                
        elif self.state == State.R1_TO_HANDOVER:
            self.q_target_r1 = np.array(self.configs['r1_handover'])
            if phase_time > PHASE_DURATION:
                self._transition(State.R2_TO_HANDOVER, t)
                
        # ===== HANDOVER =====
        elif self.state == State.R2_TO_HANDOVER:
            self.q_target_r2 = np.array(self.configs['r2_handover'])
            if phase_time > PHASE_DURATION:
                self._transition(State.HANDOVER_TRANSFER, t)
                
        elif self.state == State.HANDOVER_TRANSFER:
            # Azione one-shot: trasferisci cubo da R1 a R2
            if not self._action_done:
                self.gripper.handover(context)
                print(f"    🤝 Handover completato")
                self._action_done = True
            if phase_time > GRIPPER_WAIT:
                self._transition(State.R1_RETREAT, t)
                
        # ===== R2: PLACE SEQUENCE =====
        elif self.state == State.R1_RETREAT:
            self.q_target_r1 = np.array(Q_HOME)
            if phase_time > PHASE_DURATION:
                self._transition(State.R2_LIFT, t)
                
        elif self.state == State.R2_LIFT:
            self.q_target_r2 = np.array(self.configs['r2_lift'])
            if phase_time > PHASE_DURATION:
                self._transition(State.R2_APPROACH_PLACE, t)
                
        elif self.state == State.R2_APPROACH_PLACE:
            self.q_target_r2 = np.array(self.configs['r2_approach_place'])
            if phase_time > PHASE_DURATION:
                self._transition(State.R2_PLACE, t)
                
        elif self.state == State.R2_PLACE:
            self.q_target_r2 = np.array(self.configs['r2_place'])
            if phase_time > PHASE_DURATION:
                self._transition(State.R2_OPEN_GRIPPER, t)
                
        elif self.state == State.R2_OPEN_GRIPPER:
            # Azione one-shot: rilascia il cubo
            if not self._action_done:
                self.gripper.gripper2.detach(context)
                print(f"    📦 Cubo rilasciato da R2")
                self._action_done = True
            if phase_time > GRIPPER_WAIT:
                self._transition(State.R2_RETREAT, t)
                
        elif self.state == State.R2_RETREAT:
            self.q_target_r2 = np.array(Q_HOME)
            if phase_time > PHASE_DURATION:
                self._transition(State.DONE, t)
                
        # ===== DONE =====
        elif self.state == State.DONE:
            return True
            
        return False
    
    def _transition(self, new_state, t):
        """
        Esegue transizione a nuovo stato.
        
        Args:
            new_state: State enum del nuovo stato
            t: tempo corrente
        """
        print(f"[{t:5.1f}s] {self.state.name} → {new_state.name}")
        self.state = new_state
        self.state_start_time = t
        self._action_done = False  # Reset flag per prossima azione
    
    def reset(self):
        """Resetta la state machine allo stato iniziale."""
        self.state = State.IDLE
        self.state_start_time = 0.0
        self.q_target_r1 = np.array(Q_HOME)
        self.q_target_r2 = np.array(Q_HOME)
        self._action_done = False


# ============================================================
# HELPER: CALCOLO CONFIGURAZIONI IK
# ============================================================

def compute_ik_configs(plant, context, panda1, panda2, 
                       cube_pos, place_pos, handover_q1, handover_q2,
                       gripper_rpy_r1, gripper_rpy_r2, place_rpy_r2,
                       approach_height=0.12, grasp_z_offset=0.04):
    """
    Calcola tutte le configurazioni IK necessarie per la state machine.
    
    Args:
        plant: MultibodyPlant
        context: plant context
        panda1, panda2: ModelInstanceIndex dei robot
        cube_pos: [x, y, z] posizione iniziale cubo
        place_pos: [x, y, z] posizione target dove posare
        handover_q1, handover_q2: Configurazioni handover pre-calcolate
        gripper_rpy_r1: [r, p, y] orientamento gripper R1
        gripper_rpy_r2: [r, p, y] orientamento gripper R2 per handover
        place_rpy_r2: [r, p, y] orientamento gripper R2 per place
        approach_height: Altezza sopra il target per approach
        grasp_z_offset: Offset Z del gripper sopra il centro oggetto
        
    Returns:
        dict: Configurazioni IK per la state machine
    """
    from ik_solver import solve_ik
    
    configs = {}
    
    print("\n[IK] Calcolo configurazioni...")
    
    # ===== ROBOT 1 =====
    plant.SetPositions(context, panda1, Q_HOME)
    
    # R1 Approach (sopra il cubo)
    approach_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + approach_height]
    q = solve_ik(plant, context, panda1, approach_pos, gripper_rpy_r1)
    configs['r1_approach'] = np.array(q) if q is not None else np.array(Q_HOME)
    print(f"    {'✅' if q is not None else '❌'} R1 Approach: {approach_pos}")
    
    if q is not None:
        plant.SetPositions(context, panda1, q)
    
    # R1 Grasp (sul cubo)
    grasp_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + grasp_z_offset]
    q = solve_ik(plant, context, panda1, grasp_pos, gripper_rpy_r1)
    configs['r1_grasp'] = np.array(q) if q is not None else configs['r1_approach']
    print(f"    {'✅' if q is not None else '❌'} R1 Grasp: {grasp_pos}")
    
    # R1 Lift (= approach)
    configs['r1_lift'] = configs['r1_approach'].copy()
    print(f"    ✅ R1 Lift: (= approach)")
    
    # R1 Handover (da handover_config)
    configs['r1_handover'] = np.array(handover_q1)
    print(f"    ✅ R1 Handover: (da config)")
    
    # ===== ROBOT 2 =====
    plant.SetPositions(context, panda2, Q_HOME)
    
    # R2 Handover (da handover_config)
    configs['r2_handover'] = np.array(handover_q2)
    print(f"    ✅ R2 Handover: (da config)")
    
    plant.SetPositions(context, panda2, configs['r2_handover'])
    
    # R2 Lift (sopra handover, con orientamento place)
    # Prova prima con orientamento place
    lift_pos = [place_pos[0], place_pos[1], place_pos[2] + approach_height + 0.1]
    q = solve_ik(plant, context, panda2, lift_pos, place_rpy_r2)
    if q is None:
        # Fallback: usa handover
        q = configs['r2_handover']
    configs['r2_lift'] = np.array(q)
    print(f"    {'✅' if q is not None else '⚠️'} R2 Lift")
    
    if q is not None:
        plant.SetPositions(context, panda2, q)
    
    # R2 Approach Place
    approach_place_pos = [place_pos[0], place_pos[1], place_pos[2] + approach_height]
    q = solve_ik(plant, context, panda2, approach_place_pos, place_rpy_r2)
    configs['r2_approach_place'] = np.array(q) if q is not None else configs['r2_lift']
    print(f"    {'✅' if q is not None else '❌'} R2 Approach Place: {approach_place_pos}")
    
    if q is not None:
        plant.SetPositions(context, panda2, q)
    
    # R2 Place
    place_pos_grasp = [place_pos[0], place_pos[1], place_pos[2] + grasp_z_offset]
    q = solve_ik(plant, context, panda2, place_pos_grasp, place_rpy_r2)
    configs['r2_place'] = np.array(q) if q is not None else configs['r2_approach_place']
    print(f"    {'✅' if q is not None else '❌'} R2 Place: {place_pos_grasp}")
    
    print("[IK] Completato!\n")
    
    return configs


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("STATE MACHINE - INFO")
    print("=" * 50)
    
    print("\nStati disponibili:")
    for s in State:
        print(f"  {s.value:2d}: {s.name}")
    
    print(f"\nTiming:")
    print(f"  PHASE_DURATION = {PHASE_DURATION}s")
    print(f"  GRIPPER_WAIT = {GRIPPER_WAIT}s")
    
    print(f"\nDurata stimata task:")
    n_phases = 10  # Fasi di movimento
    n_gripper = 3  # Azioni gripper
    total = n_phases * PHASE_DURATION + n_gripper * GRIPPER_WAIT
    print(f"  ~{total:.1f}s")
    
    print("\n" + "=" * 50)
    print("Uso:")
    print("  from state_machine import HandoverStateMachine, State")
    print("  from gripper_manager import DualGripperManager")
    print("")
    print("  gripper = DualGripperManager(plant, panda1, panda2)")
    print("  sm = HandoverStateMachine(gripper, ik_configs)")
    print("  sm.update(t, context, cube_body)")
    print("  q1, q2 = sm.get_targets()")
    print("=" * 50)