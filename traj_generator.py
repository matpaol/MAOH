"""
trajectory_system.py
====================
Trajectory Generator con profilo trapezoidale per il progetto Multi-Agent Handover.

Basato su tutorial_04_traj.py, esteso per:
- Waypoints multipli (output del path planner)
- Sincronizzazione giunti (tutti arrivano insieme)
- Supporto per cambio target runtime

Classi principali:
- TrapezoidalProfile: Funzioni pure per calcolo profili
- SingleSegmentTrajectory: LeafSystem start→goal (come tutorial)
- MultiSegmentTrajectory: LeafSystem che segue waypoints
- WaypointTrajectorySystem: LeafSystem con input waypoints dinamici

Uso:
    from trajectory_system import MultiSegmentTrajectory, compute_trajectory_duration
    
    # Con waypoints dal path planner
    traj = MultiSegmentTrajectory(waypoints, v_max=0.5, a_max=2.0)
"""

import numpy as np
from pydrake.systems.framework import LeafSystem, BasicVector

from config01 import (
    TRAJ_V_MAX,
    TRAJ_A_MAX,
)


# ============================================================
# PROFILO TRAPEZOIDALE - FUNZIONI PURE
# ============================================================

class TrapezoidalProfile:
    """
    Calcola e valuta profili trapezoidali di velocità.
    
    Il profilo trapezoidale ha tre fasi:
    1. Accelerazione costante (a_max)
    2. Velocità costante (v_max)  
    3. Decelerazione costante (-a_max)
    
    Se la distanza è corta, il profilo diventa triangolare
    (salta la fase a velocità costante).
    """
    
    @staticmethod
    def compute_single_joint_profile(q0, qf, v_max, a_max):
        """
        Calcola il profilo per un singolo giunto.
        
        Args:
            q0: Posizione iniziale
            qf: Posizione finale
            v_max: Velocità massima
            a_max: Accelerazione massima
            
        Returns:
            dict: Parametri del profilo
        """
        dq = qf - q0
        s = np.sign(dq) if dq != 0 else 1.0
        dq_abs = abs(dq)
        
        # Tempo per raggiungere v_max
        t_acc = v_max / a_max if a_max > 0 else 0.0
        
        # Distanza percorsa durante accelerazione + decelerazione
        d_acc = a_max * t_acc**2  # = v_max^2 / a_max
        
        if dq_abs < d_acc:
            # Profilo TRIANGOLARE (non raggiunge v_max)
            t_acc = np.sqrt(dq_abs / max(a_max, 1e-9))
            t_flat = 0.0
            v_peak = a_max * t_acc
        else:
            # Profilo TRAPEZOIDALE
            v_peak = v_max
            t_flat = (dq_abs - d_acc) / max(v_max, 1e-9)
        
        T_total = 2 * t_acc + t_flat
        
        return {
            "q0": q0,
            "qf": qf,
            "dq": dq,
            "s": s,
            "t_acc": t_acc,
            "t_flat": t_flat,
            "T": T_total,
            "a": a_max,
            "v_peak": v_peak,
        }
    
    @staticmethod
    def compute_multi_joint_profiles(q_start, q_goal, v_max, a_max, synchronize=True):
        """
        Calcola profili per tutti i giunti.
        
        Args:
            q_start: Array posizioni iniziali (n,)
            q_goal: Array posizioni finali (n,)
            v_max: Velocità max (scalare o array)
            a_max: Accelerazione max (scalare o array)
            synchronize: Se True, allunga i profili per sincronizzare
            
        Returns:
            list[dict]: Lista di profili per ogni giunto
            float: Durata totale
        """
        n = len(q_start)
        v_max = np.broadcast_to(v_max, (n,))
        a_max = np.broadcast_to(a_max, (n,))
        
        profiles = []
        max_duration = 0.0
        
        # Calcola profili individuali
        for i in range(n):
            profile = TrapezoidalProfile.compute_single_joint_profile(
                q_start[i], q_goal[i], v_max[i], a_max[i]
            )
            profiles.append(profile)
            max_duration = max(max_duration, profile["T"])
        
        # Sincronizza: scala i profili per avere la stessa durata
        if synchronize and max_duration > 0:
            for i, p in enumerate(profiles):
                if p["T"] < max_duration and abs(p["dq"]) > 1e-9:
                    # Riscala velocità e accelerazione per allungare il profilo
                    scale = p["T"] / max_duration
                    profiles[i] = TrapezoidalProfile._rescale_profile(p, max_duration)
        
        return profiles, max_duration
    
    @staticmethod
    def _rescale_profile(profile, new_duration):
        """Riscala un profilo per durare new_duration."""
        if profile["T"] < 1e-9 or new_duration < 1e-9:
            return profile
        
        scale = profile["T"] / new_duration
        
        return {
            "q0": profile["q0"],
            "qf": profile["qf"],
            "dq": profile["dq"],
            "s": profile["s"],
            "t_acc": profile["t_acc"] / scale,
            "t_flat": profile["t_flat"] / scale,
            "T": new_duration,
            "a": profile["a"] * scale**2,
            "v_peak": profile["v_peak"] * scale,
        }
    
    @staticmethod
    def eval_profile(t, profile):
        """
        Valuta posizione e velocità al tempo t.
        
        Args:
            t: Tempo corrente
            profile: Dict con parametri del profilo
            
        Returns:
            tuple: (q, qd) posizione e velocità
        """
        q0 = profile["q0"]
        s = profile["s"]
        a = profile["a"]
        t_acc = profile["t_acc"]
        t_flat = profile["t_flat"]
        T = profile["T"]
        v_peak = profile["v_peak"]
        dq = profile["dq"]
        
        if t <= 0:
            # Prima dell'inizio
            return q0, 0.0
        
        elif t < t_acc:
            # Fase 1: Accelerazione
            q = q0 + s * 0.5 * a * t**2
            qd = s * a * t
            return q, qd
        
        elif t < t_acc + t_flat:
            # Fase 2: Velocità costante
            q = q0 + s * (0.5 * a * t_acc**2 + v_peak * (t - t_acc))
            qd = s * v_peak
            return q, qd
        
        elif t < T:
            # Fase 3: Decelerazione
            td = t - (t_acc + t_flat)
            q = q0 + s * (0.5 * a * t_acc**2 + v_peak * t_flat + 
                         v_peak * td - 0.5 * a * td**2)
            qd = s * (v_peak - a * td)
            return q, qd
        
        else:
            # Dopo la fine: mantieni posizione finale
            return q0 + dq, 0.0


# ============================================================
# SINGLE SEGMENT TRAJECTORY (come tutorial)
# ============================================================

class SingleSegmentTrajectory(LeafSystem):
    """
    Trajectory Generator per movimento start → goal.
    
    Identico a JointSpaceTrajectorySystem del tutorial.
    Output: [q_ref (9), qd_ref (9)] = 18 valori
    """
    
    def __init__(self, q_start, q_goal, v_max=TRAJ_V_MAX, a_max=TRAJ_A_MAX):
        """
        Args:
            q_start: Configurazione iniziale (9,)
            q_goal: Configurazione finale (9,)
            v_max: Velocità massima (rad/s)
            a_max: Accelerazione massima (rad/s²)
        """
        super().__init__()
        
        self.q_start = np.array(q_start)
        self.q_goal = np.array(q_goal)
        self.n = len(q_start)
        
        # Calcola profili
        self.profiles, self.duration = TrapezoidalProfile.compute_multi_joint_profiles(
            self.q_start, self.q_goal, v_max, a_max, synchronize=True
        )
        
        # Output port: [q_ref, qd_ref]
        self.DeclareVectorOutputPort(
            "trajectory_state", 
            BasicVector(2 * self.n), 
            self._output_reference
        )
    
    def _output_reference(self, context, output):
        """Calcola [q_ref, qd_ref] al tempo corrente."""
        t = context.get_time()
        
        q_ref = np.zeros(self.n)
        qd_ref = np.zeros(self.n)
        
        for i, profile in enumerate(self.profiles):
            q_ref[i], qd_ref[i] = TrapezoidalProfile.eval_profile(t, profile)
        
        output.SetFromVector(np.hstack([q_ref, qd_ref]))
    
    def get_duration(self):
        """Ritorna la durata totale della traiettoria."""
        return self.duration


# ============================================================
# MULTI-SEGMENT TRAJECTORY (per waypoints dal path planner)
# ============================================================

class MultiSegmentTrajectory(LeafSystem):
    """
    Trajectory Generator che segue una sequenza di waypoints.
    
    Utile per seguire il path generato da path_planner.py.
    Concatena segmenti trapezoidali tra waypoints consecutivi.
    
    Output: [q_ref (9), qd_ref (9)] = 18 valori
    """
    
    def __init__(self, waypoints, v_max=TRAJ_V_MAX, a_max=TRAJ_A_MAX):
        """
        Args:
            waypoints: Array [N x 9] di configurazioni
            v_max: Velocità massima (rad/s)
            a_max: Accelerazione massima (rad/s²)
        """
        super().__init__()
        
        self.waypoints = np.array(waypoints)
        self.n_joints = self.waypoints.shape[1]
        self.n_waypoints = len(self.waypoints)
        self.v_max = v_max
        self.a_max = a_max
        
        # Calcola tutti i segmenti
        self._compute_all_segments()
        
        # Output port
        self.DeclareVectorOutputPort(
            "trajectory_state",
            BasicVector(2 * self.n_joints),
            self._output_reference
        )
    
    def _compute_all_segments(self):
        """Precalcola profili per ogni segmento tra waypoints."""
        self.segments = []
        self.segment_start_times = [0.0]
        
        total_time = 0.0
        
        for i in range(self.n_waypoints - 1):
            q_start = self.waypoints[i]
            q_goal = self.waypoints[i + 1]
            
            profiles, duration = TrapezoidalProfile.compute_multi_joint_profiles(
                q_start, q_goal, self.v_max, self.a_max, synchronize=True
            )
            
            self.segments.append({
                "profiles": profiles,
                "duration": duration,
                "q_start": q_start,
                "q_goal": q_goal,
            })
            
            total_time += duration
            self.segment_start_times.append(total_time)
        
        self.total_duration = total_time
    
    def _output_reference(self, context, output):
        """Calcola [q_ref, qd_ref] al tempo corrente."""
        t = context.get_time()
        
        # Trova il segmento corrente
        segment_idx = 0
        for i in range(len(self.segments)):
            if t >= self.segment_start_times[i]:
                segment_idx = i
            else:
                break
        
        # Tempo locale nel segmento
        t_local = t - self.segment_start_times[segment_idx]
        
        # Se siamo oltre l'ultimo segmento, mantieni posizione finale
        if segment_idx >= len(self.segments):
            q_ref = self.waypoints[-1]
            qd_ref = np.zeros(self.n_joints)
        else:
            segment = self.segments[segment_idx]
            q_ref = np.zeros(self.n_joints)
            qd_ref = np.zeros(self.n_joints)
            
            for j, profile in enumerate(segment["profiles"]):
                q_ref[j], qd_ref[j] = TrapezoidalProfile.eval_profile(t_local, profile)
        
        output.SetFromVector(np.hstack([q_ref, qd_ref]))
    
    def get_duration(self):
        """Ritorna la durata totale della traiettoria."""
        return self.total_duration
    
    def get_segment_times(self):
        """Ritorna i tempi di inizio di ogni segmento."""
        return self.segment_start_times.copy()


# ============================================================
# DYNAMIC TRAJECTORY (per cambio target runtime)
# ============================================================

class DynamicTrajectory(LeafSystem):
    """
    Trajectory Generator con target dinamico.
    
    Può ricevere un nuovo target in input e ricalcola la traiettoria
    dalla posizione corrente. Utile per la State Machine.
    
    Input Ports:
        - current_state (18,): [q, qd] attuale
        - target_config (9,): q_target desiderato
        
    Output Ports:
        - trajectory_state (18,): [q_ref, qd_ref]
        - trajectory_done (1,): 1.0 se arrivato, 0.0 altrimenti
    """
    
    def __init__(self, num_joints=9, v_max=TRAJ_V_MAX, a_max=TRAJ_A_MAX, 
                 target_tolerance=0.01):
        """
        Args:
            num_joints: Numero di giunti (default 9 per Panda)
            v_max: Velocità massima
            a_max: Accelerazione massima
            target_tolerance: Tolleranza per considerare target raggiunto
        """
        super().__init__()
        
        self.n = num_joints
        self.v_max = v_max
        self.a_max = a_max
        self.target_tolerance = target_tolerance
        
        # Stato interno
        self.current_target = None
        self.profiles = None
        self.trajectory_start_time = 0.0
        self.duration = 0.0
        
        # Input ports
        self._current_state_port = self.DeclareVectorInputPort(
            "current_state", self.n * 2
        )
        self._target_port = self.DeclareVectorInputPort(
            "target_config", self.n
        )
        
        # Output ports
        self.DeclareVectorOutputPort(
            "trajectory_state",
            BasicVector(2 * self.n),
            self._output_reference
        )
        self.DeclareVectorOutputPort(
            "trajectory_done",
            BasicVector(1),
            self._output_done
        )
        
        # Update periodico per controllare cambio target
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.01,  # 100 Hz
            offset_sec=0.0,
            update=self._check_target_change
        )
    
    def _check_target_change(self, context, discrete_state):
        """Controlla se il target è cambiato e ricalcola traiettoria."""
        new_target = self._target_port.Eval(context)
        current_state = self._current_state_port.Eval(context)
        current_q = current_state[:self.n]
        
        # Controlla se target è cambiato
        target_changed = (
            self.current_target is None or
            np.linalg.norm(new_target - self.current_target) > self.target_tolerance
        )
        
        if target_changed:
            # Ricalcola traiettoria dalla posizione corrente
            self.current_target = new_target.copy()
            self.trajectory_start_time = context.get_time()
            
            self.profiles, self.duration = TrapezoidalProfile.compute_multi_joint_profiles(
                current_q, new_target, self.v_max, self.a_max, synchronize=True
            )
    
    def _output_reference(self, context, output):
        """Calcola [q_ref, qd_ref]."""
        if self.profiles is None:
            # Nessuna traiettoria: mantieni posizione corrente
            current_state = self._current_state_port.Eval(context)
            q_ref = current_state[:self.n]
            qd_ref = np.zeros(self.n)
        else:
            t = context.get_time() - self.trajectory_start_time
            q_ref = np.zeros(self.n)
            qd_ref = np.zeros(self.n)
            
            for i, profile in enumerate(self.profiles):
                q_ref[i], qd_ref[i] = TrapezoidalProfile.eval_profile(t, profile)
        
        output.SetFromVector(np.hstack([q_ref, qd_ref]))
    
    def _output_done(self, context, output):
        """Output 1.0 se traiettoria completata, 0.0 altrimenti."""
        if self.profiles is None:
            output.SetFromVector([0.0])
            return
        
        t = context.get_time() - self.trajectory_start_time
        done = 1.0 if t >= self.duration else 0.0
        output.SetFromVector([done])


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def compute_trajectory_duration(q_start, q_goal, v_max=TRAJ_V_MAX, a_max=TRAJ_A_MAX):
    """
    Calcola la durata di una traiettoria senza creare il LeafSystem.
    
    Utile per stimare tempi nella State Machine.
    """
    _, duration = TrapezoidalProfile.compute_multi_joint_profiles(
        np.array(q_start), np.array(q_goal), v_max, a_max, synchronize=True
    )
    return duration


def compute_waypoints_duration(waypoints, v_max=TRAJ_V_MAX, a_max=TRAJ_A_MAX):
    """
    Calcola la durata totale per seguire una lista di waypoints.
    """
    total = 0.0
    for i in range(len(waypoints) - 1):
        total += compute_trajectory_duration(waypoints[i], waypoints[i+1], v_max, a_max)
    return total


def interpolate_waypoints_time(waypoints, dt, v_max=TRAJ_V_MAX, a_max=TRAJ_A_MAX):
    """
    Campiona waypoints a intervalli dt regolari.
    
    Args:
        waypoints: Array [N x dof]
        dt: Intervallo di campionamento (secondi)
        v_max, a_max: Limiti di velocità/accelerazione
        
    Returns:
        np.ndarray: Array [M x dof] campionato uniformemente
        np.ndarray: Array [M] dei tempi corrispondenti
    """
    # Crea sistema temporaneo
    traj = MultiSegmentTrajectory(waypoints, v_max, a_max)
    duration = traj.get_duration()
    
    # Campiona
    times = np.arange(0, duration + dt, dt)
    n_joints = waypoints.shape[1]
    
    sampled_q = np.zeros((len(times), n_joints))
    sampled_qd = np.zeros((len(times), n_joints))
    
    for k, t in enumerate(times):
        # Trova segmento
        seg_idx = 0
        for i in range(len(traj.segments)):
            if t >= traj.segment_start_times[i]:
                seg_idx = i
        
        t_local = t - traj.segment_start_times[seg_idx]
        
        if seg_idx < len(traj.segments):
            for j, profile in enumerate(traj.segments[seg_idx]["profiles"]):
                sampled_q[k, j], sampled_qd[k, j] = TrapezoidalProfile.eval_profile(t_local, profile)
        else:
            sampled_q[k] = waypoints[-1]
            sampled_qd[k] = 0.0
    
    return sampled_q, sampled_qd, times


# ============================================================
# TEST
# ============================================================

def test_trajectory_system():
    """Test delle classi di trajectory generation."""
    import matplotlib.pyplot as plt
    
    print("=" * 50)
    print("TEST TRAJECTORY SYSTEM")
    print("=" * 50)
    
    # Test 1: Single Segment
    print("\n[1/3] Test SingleSegmentTrajectory...")
    q_start = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]
    q_goal = [0.5, -0.5, 0.3, -2.0, 0.2, 1.0, 0.5, 0.04, 0.04]
    
    traj = SingleSegmentTrajectory(q_start, q_goal, v_max=0.5, a_max=2.0)
    print(f"   Durata: {traj.get_duration():.2f} s")
    
    # Test 2: Multi Segment (simula output path planner)
    print("\n[2/3] Test MultiSegmentTrajectory...")
    waypoints = np.array([
        [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0],
        [0.2, -0.6, 0.1, -2.2, 0.1, 1.4, 0.6, 0.02, 0.02],
        [0.4, -0.4, 0.2, -2.0, 0.2, 1.2, 0.4, 0.03, 0.03],
        [0.5, -0.5, 0.3, -2.0, 0.2, 1.0, 0.5, 0.04, 0.04],
    ])
    
    multi_traj = MultiSegmentTrajectory(waypoints, v_max=0.5, a_max=2.0)
    print(f"   Waypoints: {len(waypoints)}")
    print(f"   Durata totale: {multi_traj.get_duration():.2f} s")
    print(f"   Tempi segmenti: {[f'{t:.2f}' for t in multi_traj.get_segment_times()]}")
    
    # Test 3: Campionamento temporale
    print("\n[3/3] Test interpolazione temporale...")
    sampled_q, sampled_qd, times = interpolate_waypoints_time(waypoints, dt=0.05)
    print(f"   Campioni: {len(times)}")
    print(f"   Tempo totale: {times[-1]:.2f} s")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Posizioni
    for j in range(7):  # Solo primi 7 giunti
        axes[0].plot(times, sampled_q[:, j], label=f'Joint {j+1}')
    axes[0].set_ylabel('Position [rad]')
    axes[0].set_title('Trajectory - Joint Positions')
    axes[0].legend(loc='upper right', ncol=4)
    axes[0].grid(True)
    
    # Velocità
    for j in range(7):
        axes[1].plot(times, sampled_qd[:, j], label=f'Joint {j+1}')
    axes[1].set_ylabel('Velocity [rad/s]')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_title('Trajectory - Joint Velocities')
    axes[1].legend(loc='upper right', ncol=4)
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/trajectory_test.png', dpi=100)
    print(f"\n   Plot salvato: /tmp/trajectory_test.png")
    plt.close()
    
    print("\n" + "=" * 50)
    print("✅ TEST COMPLETATO")
    print("=" * 50)


if __name__ == "__main__":
    test_trajectory_system()