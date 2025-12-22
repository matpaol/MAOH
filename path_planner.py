"""
path_planner.py
===============
Path Planning con OMPL (RRT-Connect) per il progetto Multi-Agent Handover.

Basato su tutorial_04_path_planner.py, adattato per:
- Due robot Panda
- Collision checking multi-robot
- Integrazione con moduli esistenti (visualize.py, config01.py)

Funzioni principali:
- plan_path(): Pianifica path per un singolo robot
- plan_path_with_other_robot(): Pianifica considerando l'altro robot fermo
- CollisionChecker: Classe per verificare collisioni

Uso:
    from path_planner import plan_path, create_planning_scene
    
    scene = create_planning_scene()
    path = plan_path(scene, 'panda_1', q_start, q_goal)
"""

import numpy as np
from ompl import base as ob
from ompl import geometric as og

from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser

from config01 import (
    SDF_FILE_PATH,
    SIM_TIME_STEP,
    PANDA1_NAME,
    PANDA2_NAME,
    Q_HOME,
)


# ============================================================
# CONFIGURAZIONE PATH PLANNER
# ============================================================

# Parametri di default per il planner
DEFAULT_TIMEOUT = 3.0           # secondi
DEFAULT_NUM_POINTS = 30         # waypoints interpolati
DEFAULT_MIN_DISTANCE = 0.02     # distanza minima collision (metri)
DEFAULT_GOAL_TOLERANCE = 1e-2   # tolleranza goal (radianti)


# ============================================================
# SETUP SCENA PER PLANNING
# ============================================================

def create_planning_scene(time_step=SIM_TIME_STEP):
    """
    Crea una scena dedicata al path planning (separata dalla simulazione).
    
    Il path planner ha bisogno di un plant/context separato per poter
    testare configurazioni senza influenzare la simulazione.
    
    Returns:
        dict: {
            'plant': MultibodyPlant,
            'context': Context,
            'panda1': ModelInstance,
            'panda2': ModelInstance,
        }
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    
    # Carica la scena completa dall'SDF
    Parser(plant).AddModelsFromUrl("file://" + SDF_FILE_PATH)
    plant.Finalize()
    
    # Ottieni riferimenti ai robot
    panda1 = plant.GetModelInstanceByName(PANDA1_NAME)
    panda2 = plant.GetModelInstanceByName(PANDA2_NAME)
    
    # Crea il diagram (necessario per geometry queries)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    
    # Imposta posizioni iniziali
    plant.SetPositions(plant_context, panda1, Q_HOME)
    plant.SetPositions(plant_context, panda2, Q_HOME)
    
    return {
        'plant': plant,
        'context': plant_context,
        'diagram': diagram,
        'diagram_context': diagram_context,
        'panda1': panda1,
        'panda2': panda2,
    }


# ============================================================
# COLLISION CHECKER
# ============================================================

class CollisionChecker:
    """
    Verifica collisioni per una configurazione robot.
    
    Basato su tutorial_04_path_planner.py, esteso per due robot.
    
    Attributes:
        plant: MultibodyPlant
        context: Context del plant
        robot_model: ModelInstance del robot da controllare
        min_distance: Distanza minima accettabile
        ignore_models: Lista di model names da ignorare nelle collisioni
    """
    
    def __init__(self, plant, context, robot_model, min_distance=DEFAULT_MIN_DISTANCE,
                 ignore_models=None):
        """
        Args:
            plant: MultibodyPlant della scena
            context: Context del plant
            robot_model: ModelInstance del robot che si sta pianificando
            min_distance: Distanza minima per considerare collision-free
            ignore_models: Lista di nomi modelli da ignorare (es. cubo se lo teniamo)
        """
        self.plant = plant
        self.context = context
        self.robot_model = robot_model
        self.robot_name = plant.GetModelInstanceName(robot_model)
        self.min_distance = min_distance
        self.ignore_models = ignore_models or []
        
        # Numero di posizioni del robot
        self.num_positions = plant.num_positions(robot_model)
    
    def check_configuration(self, q):
        """
        Verifica se la configurazione q è collision-free.
        
        Args:
            q: Array di joint positions per il robot
            
        Returns:
            bool: True se collision-free, False altrimenti
        """
        # Imposta la configurazione del robot
        self.plant.SetPositions(self.context, self.robot_model, q)
        
        # Ottieni query object per collision detection
        query_object = self.plant.get_geometry_query_input_port().Eval(self.context)
        inspector = query_object.inspector()
        
        # Calcola distanze tra tutte le coppie di geometrie
        distances = query_object.ComputeSignedDistancePairwiseClosestPoints()
        
        # Trova la distanza minima rilevante
        min_dist = float("inf")
        
        for pair in distances:
            # Ottieni i body coinvolti
            body_A, body_B = self._get_bodies_from_pair(pair, inspector)
            
            # Ottieni i nomi dei modelli
            model_A = self._get_model_name(body_A)
            model_B = self._get_model_name(body_B)
            
            # Salta se entrambi appartengono allo stesso modello (self-collision)
            if model_A == model_B:
                continue
            
            # Salta se uno dei modelli è nella lista da ignorare
            if model_A in self.ignore_models or model_B in self.ignore_models:
                continue
            
            # Salta se nessuno dei due è il nostro robot
            if model_A != self.robot_name and model_B != self.robot_name:
                continue
            
            # Considera questa coppia
            min_dist = min(min_dist, pair.distance)
        
        # Se nessuna coppia rilevante trovata, è valido
        if min_dist == float("inf"):
            return True
        
        return min_dist >= self.min_distance
    
    def _get_bodies_from_pair(self, pair, inspector):
        """Estrae i Body da una coppia di distanze."""
        frame_id_A = inspector.GetFrameId(pair.id_A)
        frame_id_B = inspector.GetFrameId(pair.id_B)
        body_A = self.plant.GetBodyFromFrameId(frame_id_A)
        body_B = self.plant.GetBodyFromFrameId(frame_id_B)
        return body_A, body_B
    
    def _get_model_name(self, body):
        """Ottiene il nome del modello a cui appartiene il body."""
        model_instance = body.model_instance()
        return self.plant.GetModelInstanceName(model_instance)


# ============================================================
# OMPL VALIDITY CHECKER WRAPPER
# ============================================================

class JointSpaceValidityChecker(ob.StateValidityChecker):
    """
    Wrapper OMPL per il CollisionChecker.
    
    OMPL richiede una classe che eredita da StateValidityChecker.
    Questa classe fa da ponte tra OMPL e il nostro CollisionChecker.
    """
    
    def __init__(self, si, collision_checker, num_dof):
        """
        Args:
            si: SpaceInformation di OMPL
            collision_checker: Istanza di CollisionChecker
            num_dof: Numero di gradi di libertà
        """
        super().__init__(si)
        self.collision_checker = collision_checker
        self.num_dof = num_dof
    
    def isValid(self, state):
        """
        Metodo richiesto da OMPL per verificare validità.
        
        Args:
            state: Stato OMPL
            
        Returns:
            bool: True se valido
        """
        # Converti stato OMPL in numpy array
        q = np.array([state[i] for i in range(self.num_dof)])
        return self.collision_checker.check_configuration(q)


# ============================================================
# PATH PLANNING FUNCTIONS
# ============================================================

def plan_path(scene, robot_name, q_start, q_goal, 
              timeout=DEFAULT_TIMEOUT, 
              num_points=DEFAULT_NUM_POINTS,
              min_distance=DEFAULT_MIN_DISTANCE,
              ignore_models=None):
    """
    Pianifica un path collision-free usando RRT-Connect.
    
    Basato su tutorial_04_path_planner.py.
    
    Args:
        scene: Dict ritornato da create_planning_scene()
        robot_name: 'panda_1' o 'panda_2'
        q_start: Configurazione iniziale (9,)
        q_goal: Configurazione finale (9,)
        timeout: Tempo massimo di planning (secondi)
        num_points: Numero di waypoints nel path interpolato
        min_distance: Distanza minima per collision checking
        ignore_models: Lista di modelli da ignorare nelle collisioni
        
    Returns:
        np.ndarray: Path [N x 9] oppure None se fallisce
    """
    plant = scene['plant']
    context = scene['context']
    
    # Ottieni il model instance del robot
    robot_model = plant.GetModelInstanceByName(robot_name)
    num_dof = plant.num_positions(robot_model)
    
    # Converti input in numpy arrays
    q_start = np.array(q_start)
    q_goal = np.array(q_goal)
    
    # Crea collision checker
    collision_checker = CollisionChecker(
        plant, context, robot_model, 
        min_distance=min_distance,
        ignore_models=ignore_models
    )
    
    # Verifica che start e goal siano validi
    if not collision_checker.check_configuration(q_start):
        print(f"⚠️  Configurazione start non valida per {robot_name}")
        return None
    
    # Reset e verifica goal
    plant.SetPositions(context, robot_model, q_start)
    if not collision_checker.check_configuration(q_goal):
        print(f"⚠️  Configurazione goal non valida per {robot_name}")
        return None
    
    # Reset a start per il planning
    plant.SetPositions(context, robot_model, q_start)
    
    # === Setup OMPL ===
    
    # Spazio degli stati (joint space)
    space = ob.RealVectorStateSpace(num_dof)
    
    # Joint limits
    bounds = ob.RealVectorBounds(num_dof)
    lower_limits = plant.GetPositionLowerLimits()
    upper_limits = plant.GetPositionUpperLimits()
    
    # Estrai solo i limiti del robot corrente
    # (il plant ha tutti i joint di tutti i modelli)
    robot_lower = plant.GetPositionsFromArray(robot_model, lower_limits)
    robot_upper = plant.GetPositionsFromArray(robot_model, upper_limits)
    
    for i in range(num_dof):
        bounds.setLow(i, robot_lower[i])
        bounds.setHigh(i, robot_upper[i])
    space.setBounds(bounds)
    
    # Space information con validity checker
    si = ob.SpaceInformation(space)
    validity_checker = JointSpaceValidityChecker(si, collision_checker, num_dof)
    si.setStateValidityChecker(validity_checker)
    si.setup()
    
    # Stati start e goal
    start_state = ob.State(space)
    goal_state = ob.State(space)
    for i in range(num_dof):
        start_state[i] = q_start[i]
        goal_state[i] = q_goal[i]
    
    # Problem definition
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state, DEFAULT_GOAL_TOLERANCE)
    
    # Planner: RRT-Connect (bidirezionale, efficiente)
    planner = og.RRTConnect(si)
    planner.setProblemDefinition(pdef)
    planner.setup()
    
    # === Risolvi ===
    solved = planner.solve(timeout)
    
    if solved:
        # Ottieni il path
        path = pdef.getSolutionPath()
        
        # Semplifica (rimuovi waypoints ridondanti)
        simplifier = og.PathSimplifier(si)
        simplifier.simplifyMax(path)
        
        # Interpola per ottenere path smooth
        path.interpolate(num_points)
        
        # Converti in numpy array
        waypoints = np.array([
            [state[i] for i in range(num_dof)]
            for state in path.getStates()
        ])
        
        print(f"✅ Path trovato per {robot_name}: {len(waypoints)} waypoints")
        return waypoints
    
    else:
        print(f"❌ RRT-Connect fallito per {robot_name}")
        return None


def plan_path_with_other_robot(scene, robot_name, q_start, q_goal,
                                other_robot_name, other_robot_q,
                                timeout=DEFAULT_TIMEOUT,
                                num_points=DEFAULT_NUM_POINTS,
                                ignore_models=None):
    """
    Pianifica path per un robot mentre l'altro è in una posizione fissa.
    
    Importante per l'handover: quando un robot si muove, l'altro
    potrebbe essere fermo in posizione e va considerato come ostacolo.
    
    Args:
        scene: Dict da create_planning_scene()
        robot_name: Robot che si muove ('panda_1' o 'panda_2')
        q_start: Configurazione iniziale
        q_goal: Configurazione finale
        other_robot_name: Robot fermo ('panda_1' o 'panda_2')
        other_robot_q: Configurazione del robot fermo
        timeout: Tempo massimo planning
        num_points: Waypoints nel path
        ignore_models: Modelli da ignorare
        
    Returns:
        np.ndarray: Path [N x 9] oppure None
    """
    plant = scene['plant']
    context = scene['context']
    
    # Imposta la posizione dell'altro robot
    other_robot = plant.GetModelInstanceByName(other_robot_name)
    plant.SetPositions(context, other_robot, other_robot_q)
    
    # Pianifica normalmente (l'altro robot è ora un "ostacolo")
    return plan_path(
        scene, robot_name, q_start, q_goal,
        timeout=timeout,
        num_points=num_points,
        ignore_models=ignore_models
    )


def plan_synchronized_paths(scene, 
                            q1_start, q1_goal,
                            q2_start, q2_goal,
                            timeout=DEFAULT_TIMEOUT,
                            num_points=DEFAULT_NUM_POINTS):
    """
    Pianifica path per entrambi i robot simultaneamente.
    
    Strategia: pianifica prima robot1, poi robot2 considerando robot1.
    Non è ottimale ma è semplice e funziona per handover.
    
    Args:
        scene: Dict da create_planning_scene()
        q1_start, q1_goal: Start/goal per panda_1
        q2_start, q2_goal: Start/goal per panda_2
        
    Returns:
        tuple: (path1, path2) oppure (None, None) se fallisce
    """
    # Prima pianifica robot 1 (robot 2 fermo a start)
    path1 = plan_path_with_other_robot(
        scene, PANDA1_NAME, q1_start, q1_goal,
        PANDA2_NAME, q2_start,
        timeout=timeout, num_points=num_points
    )
    
    if path1 is None:
        return None, None
    
    # Poi pianifica robot 2 (robot 1 fermo a goal)
    path2 = plan_path_with_other_robot(
        scene, PANDA2_NAME, q2_start, q2_goal,
        PANDA1_NAME, q1_goal,
        timeout=timeout, num_points=num_points
    )
    
    if path2 is None:
        return None, None
    
    return path1, path2


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def interpolate_path(waypoints, num_points):
    """
    Interpola linearmente tra waypoints per ottenere path più denso.
    
    Args:
        waypoints: Array [M x dof]
        num_points: Numero di punti desiderati
        
    Returns:
        np.ndarray: Array [num_points x dof]
    """
    if len(waypoints) < 2:
        return waypoints
    
    # Parametro normalizzato [0, 1]
    t_original = np.linspace(0, 1, len(waypoints))
    t_new = np.linspace(0, 1, num_points)
    
    # Interpola ogni joint
    interpolated = np.zeros((num_points, waypoints.shape[1]))
    for j in range(waypoints.shape[1]):
        interpolated[:, j] = np.interp(t_new, t_original, waypoints[:, j])
    
    return interpolated


def path_length(waypoints):
    """
    Calcola la lunghezza totale del path in joint space.
    
    Args:
        waypoints: Array [N x dof]
        
    Returns:
        float: Lunghezza (somma delle distanze tra waypoints)
    """
    if len(waypoints) < 2:
        return 0.0
    
    total = 0.0
    for i in range(1, len(waypoints)):
        total += np.linalg.norm(waypoints[i] - waypoints[i-1])
    return total


def validate_path(scene, robot_name, waypoints, min_distance=DEFAULT_MIN_DISTANCE):
    """
    Verifica che tutti i waypoints siano collision-free.
    
    Args:
        scene: Dict da create_planning_scene()
        robot_name: Nome del robot
        waypoints: Array [N x dof]
        min_distance: Distanza minima
        
    Returns:
        bool: True se tutto il path è valido
    """
    plant = scene['plant']
    context = scene['context']
    robot_model = plant.GetModelInstanceByName(robot_name)
    
    checker = CollisionChecker(plant, context, robot_model, min_distance)
    
    for i, q in enumerate(waypoints):
        if not checker.check_configuration(q):
            print(f"⚠️  Waypoint {i} non valido: {q}")
            return False
    
    return True


# ============================================================
# TEST
# ============================================================

def test_path_planner():
    """Test base del path planner."""
    print("=" * 50)
    print("TEST PATH PLANNER")
    print("=" * 50)
    
    # Setup
    print("\n[1/4] Creazione scena...")
    scene = create_planning_scene()
    print("✅ Scena creata")
    
    # Test single robot planning
    print("\n[2/4] Test planning singolo robot...")
    q_start = Q_HOME.copy()
    q_goal = [0.5, -0.5, 0.3, -2.0, 0.2, 1.0, 0.5, 0.04, 0.04]
    
    path = plan_path(scene, PANDA1_NAME, q_start, q_goal, timeout=5.0)
    
    if path is not None:
        print(f"   Waypoints: {len(path)}")
        print(f"   Lunghezza: {path_length(path):.3f}")
        print(f"   Start: {path[0][:3]}...")
        print(f"   Goal:  {path[-1][:3]}...")
    
    # Test with other robot
    print("\n[3/4] Test planning con altro robot...")
    path2 = plan_path_with_other_robot(
        scene, PANDA2_NAME, Q_HOME, q_goal,
        PANDA1_NAME, q_goal  # panda1 è al goal
    )
    
    if path2 is not None:
        print(f"   Waypoints: {len(path2)}")
        print(f"   Lunghezza: {path_length(path2):.3f}")
    
    # Validate
    print("\n[4/4] Validazione paths...")
    if path is not None:
        valid = validate_path(scene, PANDA1_NAME, path)
        print(f"   Path 1: {'✅ Valido' if valid else '❌ Non valido'}")
    
    if path2 is not None:
        valid2 = validate_path(scene, PANDA2_NAME, path2)
        print(f"   Path 2: {'✅ Valido' if valid2 else '❌ Non valido'}")
    
    print("\n" + "=" * 50)
    print("✅ TEST COMPLETATO")
    print("=" * 50)


if __name__ == "__main__":
    test_path_planner()