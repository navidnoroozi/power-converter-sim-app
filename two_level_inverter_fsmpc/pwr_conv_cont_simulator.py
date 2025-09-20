from itertools import islice, cycle
import math
import os
import uuid
from two_level_inverter_fsmpc.power_current_conv.power_current_handler import RequiredPowerCurrentHandler
from two_level_inverter_fsmpc.mpc_contr.mpc_contr_calc import MPCSSolver
from two_level_inverter_fsmpc.load.load_dyn_cal import Load
from two_level_inverter_fsmpc.inverter.inverter_behave import Inverter
from two_level_inverter_fsmpc.current_reference.current_ref_gen import CurrentReference
from two_level_inverter_fsmpc.scenario_executor.scenario_exc import sim_executor
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Agg")

# --- New imports for strict lifecycle ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Ensure static/plots exists
PLOT_FOLDER = os.path.join("static", "plots")
os.makedirs(PLOT_FOLDER, exist_ok=True)


def simPowerConvControlSyst(
    P_req=3e3,
    Q_req=0.0,
    V_rms_req=230.0,
    cont_horizon=5,
    t_0=0.0,
    i_a_0=5.0,
    sampling_rate=25e-6,
    sim_time=50e-4,
    R=10.0,
    L=10e-3,
    V_backemf=60.0,
    f_ref=50.0,
):
    """Run MPC simulation and generate plots.
    Parameters:
        P_req (float): Required active power (W)
        Q_req (float): Required reactive power (VAR)
        V_rms_req (float): Required RMS voltage (V)
        cont_horizon (int): Control horizon (number of time steps)
        t_0 (float): Initial time (s)
        i_a_0 (float): Initial load current (A)
        sampling_rate (float): Sampling time (s)
        sim_time (float): Total simulation time (s)
        R (float): Load resistance (Ohm)
        L (float): Load inductance (H)
        V_backemf (float): Back EMF voltage (V)
        f_ref (float): Reference frequency (Hz)
    Returns:
        filename1 (str): Filename of the first plot (switching signal and load current)
        filename2 (str): Filename of the second plot (cost function value)    
    """

    V_dc = V_rms_req * math.sqrt(2) * 2

    inverter = Inverter(V_dc)
    load = Load(R, L, V_backemf, f_backemf=f_ref)
    powerCurrentHandler = RequiredPowerCurrentHandler(P_req, Q_req, V_rms_req)
    I_ref_peak, phi_ref = powerCurrentHandler.calculateCurrentMagnitudeAndPhase()
    currentReference = CurrentReference(sampling_rate, cont_horizon, I_ref_peak, phi_ref, f_ref)
    mpc = MPCSSolver(cont_horizon=cont_horizon, timelimit=0.001, mipgap=0.05, threads=1, use_gurobi=False)

    s0 = list(islice(cycle([True, False]), cont_horizon))

    s_traj, i_a_traj, i_ref_traj, t_sim, cost_func_val = sim_executor(
        load, inverter, mpc, currentReference, s0, t_0, i_a_0, sampling_rate, sim_time
    )

    filename1 = f"simulation_{uuid.uuid4().hex}.png"
    plot_path1 = os.path.join(PLOT_FOLDER, filename1)
    filename2 = f"simulation_{uuid.uuid4().hex}.png"
    plot_path2 = os.path.join(PLOT_FOLDER, filename2)

    # === Plot 1: Switching signal + Load current ===
    fig1 = Figure(figsize=(8, 6))
    axs = fig1.subplots(2, 1)

    axs[0].step(t_sim, s_traj, label="Switching Signal s")
    axs[0].set_title("Switching Signal")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("s")
    axs[0].grid()
    axs[0].legend()
    axs[0].set_xlim([t_0, sim_time])

    axs[1].plot(t_sim, i_a_traj, label="Load Current i_a")
    axs[1].plot(t_sim, i_ref_traj, "--", label="Reference Current i_a_ref")
    axs[1].set_title("Load Current")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("i_a (A)")
    axs[1].grid()
    axs[1].legend()
    axs[1].set_xlim([t_0, sim_time])

    fig1.tight_layout()
    canvas1 = FigureCanvas(fig1)
    canvas1.print_png(open(plot_path1, "wb"))
    fig1.clf()

    # === Plot 2: Cost function ===
    fig2 = Figure(figsize=(8, 4))
    ax = fig2.subplots()

    ax.plot(t_sim, cost_func_val, label="Cost Function Value")
    ax.set_title("Cost Function Value Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cost Function Value")
    ax.grid()
    ax.legend()
    ax.set_xlim([t_0, sim_time])

    fig2.tight_layout()
    canvas2 = FigureCanvas(fig2)
    canvas2.print_png(open(plot_path2, "wb"))
    fig2.clf()

    # Explicit cleanup
    del fig1, fig2, axs, ax, canvas1, canvas2

    return filename1, filename2
