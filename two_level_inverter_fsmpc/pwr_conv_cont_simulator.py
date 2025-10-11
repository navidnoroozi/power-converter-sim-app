from itertools import islice, cycle
import math
import os
import json
from two_level_inverter_fsmpc.power_current_conv.power_current_handler import RequiredPowerCurrentHandler
from two_level_inverter_fsmpc.mpc_contr.mpc_contr_calc import MPCSSolver
from two_level_inverter_fsmpc.load.load_dyn_cal import Load
from two_level_inverter_fsmpc.inverter.inverter_behave import Inverter
from two_level_inverter_fsmpc.current_reference.current_ref_gen import CurrentReference
from two_level_inverter_fsmpc.scenario_executor.scenario_exc import sim_executor
import plotly.graph_objects as go
from plotly.subplots import make_subplots



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
        fig1 (plotly.graph_objs._figure.Figure): Plotly figure object for switching signal and load current
        fig2 (plotly.graph_objs._figure.Figure): Plotly figure object for cost function value
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

    # === Plot 1: Switching signal + Load current ===
    fig1 = make_subplots(rows=2, cols=1,
                         subplot_titles=("Switching Signal", "Load Current"),
                         shared_xaxes=True,vertical_spacing=0.12)

    # Switching signal
    fig1.add_trace(go.Scatter(x=t_sim, y=s_traj, mode="lines",
                              name="Switching Signal s",
                              line=dict(shape="hv")), row=1, col=1)
    
    fig1.update_yaxes(title_text="s", row=1, col=1)
    fig1.update_xaxes(title_text="Time (s)", row=1, col=1)
    
    # Load current
    fig1.add_trace(go.Scatter(x=t_sim, y=i_a_traj, mode="lines",
                              name="Load Current i_a"), row=2, col=1)
    fig1.add_trace(go.Scatter(x=t_sim, y=i_ref_traj, mode="lines",
                              line=dict(dash="dash"),
                              name="Reference Current i_a_ref"), row=2, col=1)
    fig1.update_yaxes(title_text="i_a (A)", row=2, col=1)
    fig1.update_xaxes(title_text="Time (s)", row=2, col=1)
    # Layout tweaks
    fig1.update_layout(
        height=600,
        width=600,
        template="simple_white",
        showlegend=True,
        margin=dict(l=60, r=30, t=60, b=60),  # tighter margins
    legend=dict(
        x=0.02,      # left edge (0 = far left, 1 = far right)
        y=0.98,      # top edge (0 = bottom, 1 = top)
        bgcolor="rgba(255,255,255,0.5)"  # semi-transparent
    ))
    # === Plot 2: Cost function ===
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t_sim, y=cost_func_val, mode="lines",
                              name="Cost Function Value"))
    # Layout tweaks
    fig2.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Cost Function Value",
        height=400,
        width=600,
        template="simple_white",
        showlegend=True,
        margin=dict(l=60, r=30, t=60, b=60),  # tighter margins
    legend=dict(
        x=0.02,      # left edge (0 = far left, 1 = far right)
        y=0.98,      # top edge (0 = bottom, 1 = top)
        bgcolor="rgba(255,255,255,0.5)"  # semi-transparent
    ))

    return fig1, fig2
