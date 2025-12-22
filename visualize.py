"""
Visualize Module
----------------
Functions to load and visualize the scene.
"""
import os
import pydot

from pydrake.geometry import StartMeshcat
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer
from pydrake.systems.analysis import Simulator

from config01 import (
    SDF_FILE_PATH,
    FIGURES_PATH,
    VISUALIZE_ONLY,
    SIM_TIME_STEP,
    SIM_DURATION
)


######################################################################################################
#                                    HELPER FUNCTIONS (per import)
######################################################################################################

def create_meshcat():
    """Create and clean meshcat instance."""
    meshcat = StartMeshcat()
    meshcat.Delete()
    meshcat.DeleteAddedControls()
    return meshcat


def create_scene(builder, time_step=SIM_TIME_STEP):
    """Create MultibodyPlant + SceneGraph with SDF loaded."""
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    Parser(plant).AddModelsFromUrl("file://" + SDF_FILE_PATH)
    plant.Finalize()
    return plant, scene_graph


def add_visualization(builder, meshcat):
    """Add default visualization to the diagram."""
    AddDefaultVisualization(builder, meshcat)


######################################################################################################
#                                    STANDALONE FUNCTIONS
######################################################################################################

def run_visualizer():
    """Minimal visualization using ModelVisualizer."""
    meshcat = create_meshcat()
    visualizer = ModelVisualizer(meshcat=meshcat)
    visualizer.parser().AddModelsFromUrl("file://" + SDF_FILE_PATH)
    visualizer.Run()


def run_visualization():
    """Run simulation without controllers."""
    meshcat = create_meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = create_scene(builder, SIM_TIME_STEP)
    add_visualization(builder, meshcat)
    
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    simulator.set_publish_every_time_step(True)
    
    meshcat.StartRecording()
    simulator.AdvanceTo(SIM_DURATION)
    meshcat.PublishRecording()
    
    # Save block diagram
    svg_data = diagram.GetGraphvizString(max_depth=2)
    graph = pydot.graph_from_dot_data(svg_data)[0]
    output_path = os.path.join(FIGURES_PATH, "block_diagram_sim.png")
    os.makedirs(FIGURES_PATH, exist_ok=True)
    graph.write_png(output_path)
    print(f"Block diagram saved as {output_path}")


if __name__ == "__main__":
    if VISUALIZE_ONLY:
        run_visualizer()
    else:
        run_visualization()