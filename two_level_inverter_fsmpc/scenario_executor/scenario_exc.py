# === Simulation Parameters ===
i_a_traj = []
i_ref_traj = []
s_traj = []
cost_func_val = []
t_sim = []

# === Run the simulation ===
def sim_executor(load, inverter, mpc, currentReference, s0, t_0 = 0, i_a_0 = 0, sampling_rate = 1e-4, sim_time = 0.1):
    """ Runs the simulation for the given parameters.
    Parameters:
    - load: Instance of Load class.
    - inverter: Instance of Inverter class.
    - mpc: Instance of MPCSSolver class.
    - currentReference: Instance of CurrentReference class.
    - s0: Initial guess for switching signal sequence.
    - t_0: Initial time.
    - i_a_0: Initial load current.
    - sampling_rate: Sampling rate in seconds.
    - sim_time: Total simulation time in seconds.
    Returns:
    - s_traj: Trajectory of switching signals.
    - i_a_traj: Trajectory of load current.
    - i_ref_traj: Trajectory of reference current.
    - t_sim: Time vector for the simulation.
    - cost_func_val: Cost function values over time.
    """
    current_time = t_0
    while current_time < sim_time:
        try:
            s_sig, cost_func_val_np = mpc.solveMPC(inverter,load,currentReference,current_time,i_a_0,s0)
        except Exception:
            # Hard fallback: repeat last action
            s_sig = [getattr(mpc, '_last_action', 1)]*mpc.cont_horizon
            cost_func_val_np = 1e9
        i_a_next = load.calculateLoadDynamics(i_a_0,inverter.generateOutputVoltage([s_sig[0]]),current_time,sampling_rate)[-1]
        i_ref_next = currentReference.generateRefTrajectory(current_time)
        i_a_traj.append(i_a_next)
        i_ref_traj.append(i_ref_next[0])
        t_sim.append(current_time)
        cost_func_val.append(float(cost_func_val_np))
        s_traj.append(s_sig[0])
        for i, s in enumerate(s_sig):
            if s <= 0.5:
                s0[i] = False
            else:
                s0[i] = True
        i_a_0 = i_a_next
        current_time += sampling_rate

    return s_traj, i_a_traj, i_ref_traj, t_sim, cost_func_val