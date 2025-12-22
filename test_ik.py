"""
test_ik.py
==========
Testa l'IK facendo muovere i robot verso le posizioni di handover.

PREREQUISITO: Eseguire prima workspace_analysis.py per generare handover_config.py

Usa:
- Controller da controller.py
- plot_q_dq_joint_response da controller.py
- create_meshcat, create_scene, add_visualization da visualize.py
"""

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource, LogVectorOutput
from pydrake.systems.analysis import Simulator

from config01 import Q_HOME, SIM_TIME_STEP, SIM_DURATION
from visualize import create_meshcat, create_scene, add_visualization
from controller import Controller, plot_q_dq_joint_response

# Import configurazione handover
try:
    from handover_config import HANDOVER_Q1, HANDOVER_Q2
    print("✅ handover_config.py caricato")
except ImportError:
    print("❌ handover_config.py non trovato!")
    print("   Esegui prima: python workspace_analysis.py")
    exit(1)


def main():
    print("=" * 50)
    print("TEST IK - Movimento verso Handover")
    print("=" * 50)
    
    # Setup
    print("\n[1/3] Setup scena...")
    meshcat = create_meshcat()
    
    builder = DiagramBuilder()
    plant, scene_graph = create_scene(builder, SIM_TIME_STEP)
    add_visualization(builder, meshcat)
    
    # Robots
    panda1 = plant.GetModelInstanceByName("panda_1")
    panda2 = plant.GetModelInstanceByName("panda_2")
    plant.SetDefaultPositions(panda1, Q_HOME)
    plant.SetDefaultPositions(panda2, Q_HOME)
    
    # Controllers (dal tuo controller.py)
    print("\n[2/3] Setup controllers...")
    controller1 = builder.AddNamedSystem("Controller_Panda1", Controller(plant, panda1))
    controller2 = builder.AddNamedSystem("Controller_Panda2", Controller(plant, panda2))
    
    # Target = posizioni handover
    des_pos_1 = builder.AddNamedSystem("Target_Panda1", ConstantVectorSource(HANDOVER_Q1))
    des_pos_2 = builder.AddNamedSystem("Target_Panda2", ConstantVectorSource(HANDOVER_Q2))
    
    # Connessioni Panda1
    builder.Connect(plant.GetOutputPort("panda_1_state"), controller1.GetInputPort("Current_state"))
    builder.Connect(controller1.GetOutputPort("tau_u"), plant.GetInputPort("panda_1_actuation"))
    builder.Connect(des_pos_1.get_output_port(), controller1.GetInputPort("Desired_state"))
    
    # Connessioni Panda2
    builder.Connect(plant.GetOutputPort("panda_2_state"), controller2.GetInputPort("Current_state"))
    builder.Connect(controller2.GetOutputPort("tau_u"), plant.GetInputPort("panda_2_actuation"))
    builder.Connect(des_pos_2.get_output_port(), controller2.GetInputPort("Desired_state"))
    
    # Loggers
    logger_1 = LogVectorOutput(plant.GetOutputPort("panda_1_state"), builder)
    logger_2 = LogVectorOutput(plant.GetOutputPort("panda_2_state"), builder)
    
    diagram = builder.Build()
    
    # Simulate
    print("\n[3/3] Simulazione...")
    print("🌐 http://localhost:7001")
    
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    
    meshcat.StartRecording()
    simulator.AdvanceTo(SIM_DURATION)
    meshcat.PublishRecording()
    
    print("\n✅ Simulazione completata!")
    
    # Plot (dalla tua funzione)
    print("\nGenerazione grafici...")
    sim_context = simulator.get_context()
    plot_q_dq_joint_response(logger_1, sim_context, HANDOVER_Q1, "Panda 1")
    plot_q_dq_joint_response(logger_2, sim_context, HANDOVER_Q2, "Panda 2")
    
    print("\n" + "=" * 50)
    print("✅ TEST COMPLETATO!")
    print("=" * 50)


if __name__ == "__main__":
    main()