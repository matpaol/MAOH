"""
Controller Module
-----------------
PD + Gravity compensation controller for Franka Panda robots.
"""

import numpy as np
import matplotlib.pyplot as plt

from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import ConstantVectorSource, LogVectorOutput
from pydrake.systems.analysis import Simulator

from config01 import SIM_TIME_STEP, SIM_DURATION, Q_HOME
from visualize import create_meshcat, create_scene, add_visualization


######################################################################################################
#                                    PLOTTING FUNCTION
######################################################################################################

def plot_q_dq_joint_response(logger_state, sim_cntx, des_pos, robot_name="Robot", num_joints=9):
    """Plot joint positions and velocities."""
    log = logger_state.FindLog(sim_cntx)
    time = log.sample_times()
    data = log.data()

    q = data[:num_joints, :]
    dq = data[num_joints:, :]
    q_r = des_pos

    # Joint positions
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    for i in range(7):
        axes[i].plot(time, q[i, :], label='q_n')
        axes[i].plot(time, q_r[i] * np.ones_like(time), 'r--', label='q_r')
        axes[i].set_ylabel(f'Joint {i+1} [rad]')
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{robot_name} - Joint Positions')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Joint velocities
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    for i in range(7):
        axes[i].plot(time, dq[i, :], label=f'Joint {i+1} velocity')
        axes[i].set_ylabel(f'Joint {i+1} [rad/s]')
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{robot_name} - Joint Velocities')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


######################################################################################################
#                                    PD+G CONTROLLER
######################################################################################################

class Controller(LeafSystem):
    """PD + Gravity compensation controller."""
    
    def __init__(self, plant, robot): 
        super().__init__()

        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=18)
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=9)
        self.robot = robot

        self.Kp_ = 60 * np.array([7.0, 6.0, 5.0, 4.0, 4.0, 3.0, 3.0, 1, 1])
        self.Kd_ = 20 * np.array([4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2, 2])

        

        self.plant = plant
        self.plant_context_ad = plant.CreateDefaultContext()
        
        state_index = self.DeclareDiscreteState(9)
        self.DeclareStateOutputPort("tau_u", state_index)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/1000,
            offset_sec=0.0,
            update=self.compute_tau_u)

    def compute_tau_u(self, context, discrete_state):
        num_positions = self.plant.num_positions(self.robot)
        self.q_d = self._desired_state_port.Eval(context)
        self.q   = self._current_state_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context_ad, self.robot, self.q)
        tau_g_all = self.plant.CalcGravityGeneralizedForces(self.plant_context_ad)
        gravity = -self.plant.GetVelocitiesFromArray(self.robot, tau_g_all)
        tau = (
            self.Kp_ * (self.q_d - self.q[:num_positions])
            - self.Kd_ * self.q[num_positions:]
            + gravity)
        discrete_state.get_mutable_vector().SetFromVector(tau)

######################################################################################################
#                                    TEST FUNCTION
######################################################################################################

def test_controller(sim_time=SIM_DURATION, time_step=SIM_TIME_STEP):
    """Test controller on both robots."""
    meshcat = create_meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = create_scene(builder, time_step)
    add_visualization(builder, meshcat)

    # Setup robots
    panda1 = plant.GetModelInstanceByName("panda_1")
    panda2 = plant.GetModelInstanceByName("panda_2")
    plant.SetDefaultPositions(panda1, Q_HOME)
    plant.SetDefaultPositions(panda2, Q_HOME)

    # Controllers
    controller1 = builder.AddNamedSystem("Controller_Panda1", Controller(plant, panda1))
    controller2 = builder.AddNamedSystem("Controller_Panda2", Controller(plant, panda2))

    # Targets
    q_target_1 = [0.5, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785, 0.0, 0.0]
    q_target_2 = [-0.5, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785, 0.0, 0.0]
    des_pos_1 = builder.AddNamedSystem("Target_Panda1", ConstantVectorSource(q_target_1))
    des_pos_2 = builder.AddNamedSystem("Target_Panda2", ConstantVectorSource(q_target_2))

    # Connections
    builder.Connect(plant.GetOutputPort("panda_1_state"), controller1.GetInputPort("Current_state"))
    builder.Connect(controller1.GetOutputPort("tau_u"), plant.GetInputPort("panda_1_actuation"))
    builder.Connect(des_pos_1.get_output_port(), controller1.GetInputPort("Desired_state"))

    builder.Connect(plant.GetOutputPort("panda_2_state"), controller2.GetInputPort("Current_state"))
    builder.Connect(controller2.GetOutputPort("tau_u"), plant.GetInputPort("panda_2_actuation"))
    builder.Connect(des_pos_2.get_output_port(), controller2.GetInputPort("Desired_state"))

    # Loggers
    logger_1 = LogVectorOutput(plant.GetOutputPort("panda_1_state"), builder)
    logger_2 = LogVectorOutput(plant.GetOutputPort("panda_2_state"), builder)
    
    diagram = builder.Build()

    # Simulate
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    print("🚀 Test controller... http://localhost:7001")
    
    meshcat.StartRecording()
    simulator.AdvanceTo(sim_time)
    meshcat.PublishRecording()

    plot_q_dq_joint_response(logger_1, simulator.get_context(), q_target_1, "Panda 1")
    plot_q_dq_joint_response(logger_2, simulator.get_context(), q_target_2, "Panda 2")

    print("✅ Test completato!")


if __name__ == "__main__":
    test_controller()


