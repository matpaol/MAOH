"""
gripper_manager.py
==================
Gestione del gripper Panda e attach/detach oggetti.

Il Panda ha 9 DOF:
- Joints 0-6: braccio (7 DOF)
- Joints 7-8: dita gripper (panda_finger_joint1, panda_finger_joint2)

Uso:
    from gripper_manager import GripperController, DualGripperManager
    
    # Singolo gripper
    gripper = GripperController(plant, panda1_model)
    gripper.open(plant_context)
    gripper.close(plant_context)
    gripper.attach(plant_context, cube_body)
    gripper.update(plant_context)  # chiama ogni step per teleportare il cubo
    gripper.detach(plant_context)
    
    # Dual gripper (per handover)
    dual = DualGripperManager(plant, panda1, panda2)
    dual.gripper1.attach(context, cube)
    dual.handover(context)  # trasferisce da R1 a R2
    dual.gripper2.detach(context)
"""

import numpy as np
from pydrake.math import RigidTransform


# ============================================================
# CONFIGURAZIONE
# ============================================================

GRIPPER_OPEN = 0.04       # Dita aperte (4cm)
GRIPPER_CLOSED = 0.008    # Dita chiuse (8mm)

FINGER1_INDEX = 7
FINGER2_INDEX = 8

# Offset cubo rispetto al frame panda_hand quando afferrato
GRASP_OFFSET = RigidTransform([0, 0, -0.04])


# ============================================================
# GRIPPER CONTROLLER
# ============================================================

class GripperController:
    """
    Controlla il gripper di un robot Panda.
    
    Gestisce:
    - Apertura/chiusura dita
    - Attach/detach oggetti (simulato via teleport)
    """
    
    def __init__(self, plant, robot_model, gripper_frame_name="panda_hand"):
        """
        Args:
            plant: MultibodyPlant
            robot_model: ModelInstanceIndex del robot
            gripper_frame_name: Nome del frame del gripper
        """
        self.plant = plant
        self.robot = robot_model
        self.gripper_frame = plant.GetFrameByName(gripper_frame_name, robot_model)
        
        # Stato
        self.is_open = True
        self.attached_body = None
        self.grasp_offset = GRASP_OFFSET
        
    def open(self, context):
        """Apre il gripper."""
        self._set_fingers(context, GRIPPER_OPEN)
        self.is_open = True
        
    def close(self, context):
        """Chiude il gripper."""
        self._set_fingers(context, GRIPPER_CLOSED)
        self.is_open = False
        
    def _set_fingers(self, context, position):
        """Imposta la posizione delle dita."""
        q = np.array(self.plant.GetPositions(context, self.robot))
        q[FINGER1_INDEX] = position
        q[FINGER2_INDEX] = position
        self.plant.SetPositions(context, self.robot, q)
        
    def get_finger_position(self, context):
        """Ritorna la posizione corrente delle dita."""
        q = self.plant.GetPositions(context, self.robot)
        return (q[FINGER1_INDEX] + q[FINGER2_INDEX]) / 2
    
    def attach(self, context, body):
        """
        Attacca un oggetto al gripper.
        
        Args:
            context: plant context
            body: Body dell'oggetto da attaccare
        """
        self.attached_body = body
        self.close(context)
        self.update(context)
        
    def detach(self, context):
        """Stacca l'oggetto dal gripper."""
        self.attached_body = None
        self.open(context)
        
    def update(self, context):
        """
        Aggiorna la posizione dell'oggetto attaccato.
        CHIAMARE OGNI STEP DI SIMULAZIONE!
        """
        if self.attached_body is None:
            return
            
        # Posa gripper nel mondo
        X_WG = self.gripper_frame.CalcPoseInWorld(context)
        
        # Posa oggetto = gripper + offset
        X_WO = X_WG.multiply(self.grasp_offset)
        
        # Teleporta oggetto
        self.plant.SetFreeBodyPose(context, self.attached_body, X_WO)
        
    def is_holding(self):
        """Ritorna True se tiene un oggetto."""
        return self.attached_body is not None
    
    def get_gripper_pose(self, context):
        """Ritorna la posa del gripper nel mondo."""
        return self.gripper_frame.CalcPoseInWorld(context)


# ============================================================
# DUAL GRIPPER MANAGER
# ============================================================

class DualGripperManager:
    """
    Gestisce i gripper di entrambi i robot per il task di handover.
    """
    
    def __init__(self, plant, panda1_model, panda2_model):
        """
        Args:
            plant: MultibodyPlant
            panda1_model: ModelInstanceIndex di panda_1
            panda2_model: ModelInstanceIndex di panda_2
        """
        self.plant = plant
        self.gripper1 = GripperController(plant, panda1_model)
        self.gripper2 = GripperController(plant, panda2_model)
        
    def update(self, context):
        """Aggiorna posizione oggetti. CHIAMARE OGNI STEP!"""
        self.gripper1.update(context)
        self.gripper2.update(context)
        
    def handover(self, context):
        """
        Trasferisce l'oggetto da Robot 1 a Robot 2.
        
        Returns:
            True se il trasferimento è riuscito
        """
        if not self.gripper1.is_holding():
            print("❌ Handover: R1 non tiene niente!")
            return False
            
        if self.gripper2.is_holding():
            print("❌ Handover: R2 tiene già qualcosa!")
            return False
            
        # Trasferisci
        obj = self.gripper1.attached_body
        self.gripper1.attached_body = None
        self.gripper1.open(context)
        self.gripper2.attach(context, obj)
        
        print("✅ Handover completato!")
        return True
        
    def open_all(self, context):
        """Apre entrambi i gripper."""
        self.gripper1.open(context)
        self.gripper2.open(context)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("TEST GRIPPER MANAGER")
    print("=" * 50)
    
    from pydrake.systems.framework import DiagramBuilder
    from config01 import SIM_TIME_STEP, PANDA1_NAME, PANDA2_NAME, Q_HOME
    from visualize import create_meshcat, create_scene, add_visualization
    
    # Setup
    meshcat = create_meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = create_scene(builder, SIM_TIME_STEP)
    add_visualization(builder, meshcat)
    diagram = builder.Build()
    
    context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    
    # Robots
    panda1 = plant.GetModelInstanceByName(PANDA1_NAME)
    panda2 = plant.GetModelInstanceByName(PANDA2_NAME)
    cube_body = plant.GetBodyByName("link", plant.GetModelInstanceByName("cube"))
    
    plant.SetPositions(plant_context, panda1, Q_HOME)
    plant.SetPositions(plant_context, panda2, Q_HOME)
    
    print("\n🌐 http://localhost:7001")
    
    # Test
    dual = DualGripperManager(plant, panda1, panda2)
    
    input("\n[1] Premi ENTER per aprire gripper...")
    dual.open_all(plant_context)
    diagram.ForcedPublish(context)
    print(f"    Gripper1 aperto: {dual.gripper1.is_open}")
    
    input("\n[2] Premi ENTER per chiudere gripper R1...")
    dual.gripper1.close(plant_context)
    diagram.ForcedPublish(context)
    print(f"    Gripper1 aperto: {dual.gripper1.is_open}")
    
    input("\n[3] Premi ENTER per attach cubo a R1...")
    dual.gripper1.attach(plant_context, cube_body)
    diagram.ForcedPublish(context)
    print(f"    R1 tiene cubo: {dual.gripper1.is_holding()}")
    
    input("\n[4] Premi ENTER per handover a R2...")
    dual.handover(plant_context)
    diagram.ForcedPublish(context)
    print(f"    R1 tiene cubo: {dual.gripper1.is_holding()}")
    print(f"    R2 tiene cubo: {dual.gripper2.is_holding()}")
    
    input("\n[5] Premi ENTER per detach da R2...")
    dual.gripper2.detach(plant_context)
    diagram.ForcedPublish(context)
    
    print("\n✅ Test completato!")