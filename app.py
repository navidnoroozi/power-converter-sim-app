from flask import Flask, render_template, request, url_for
from two_level_inverter_fsmpc.pwr_conv_cont_simulator import simPowerConvControlSyst
import os
import glob
import uuid  # Remove this line if you no longer use uuid for filenames

app = Flask(__name__)

# Ensure static/plots exists
PLOT_FOLDER = os.path.join(app.static_folder, "plots")
os.makedirs(PLOT_FOLDER, exist_ok=True)


def cleanup_old_plots(keep_last: int = 0):
    """
    Remove old plot files from the static/plots directory,
    keeping only the most recent `keep_last` files.
    """
    files = sorted(
        glob.glob(os.path.join(PLOT_FOLDER, "*.png")),
        key=os.path.getmtime,
        reverse=True,
    )
    for f in files[keep_last:]:
        try:
            os.remove(f)
        except OSError:
            pass

@app.after_request
def add_header(response):
    """
    Disable caching for all responses (HTML + static files).
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    ## Clean up old plots
    cleanup_old_plots(keep_last=0)  # Keep only the most recent 0 plots
    
    if request.method == "POST":

        ## Get form data:
        ## MPC parameters
        # cont_horizon: Control horizon
        # TO BE ADDED IN THE FUTURE: weighting factors for cost function
        control_horizon = int(request.form.get("control_horizon"))  # Control horizon

        ## Simulation parameters
        # t_0: Initial time in seconds
        # sim_time: Total simulation time in seconds
        # sampling_rate: Sampling rate in seconds (also the time step for simulation)
        sampling_time = float(request.form.get("sampling_time"))  # Sampling time in seconds
        sim_time = float(request.form.get("sim_time"))  # Total simulation time in seconds
        t_0 = 0.0  # Fixed to 0.0 seconds for now

        ## Power requirements
        # P_req: Active power in W
        # Q_req: Reactive power in VAR
        # V_rms_req: RMS voltage in V
        # f_ref: Reference frequency in Hz
        P_req = float(request.form.get("P_req"))  # Active power in W
        Q_req = float(request.form.get("Q_req"))  # Reactive power in VAR
        V_rms_req = float(request.form.get("V_rms_req"))  # RMS voltage in V
        f_ref = 50.0  # Fixed to 50 Hz for now

        ## RL + Motor Load parameters
        # R: Resistance in Ohm
        # L: Inductance in H
        # V_backemf: Back EMF voltage in V
        # i_a_0: Initial load current
        R = float(request.form.get("R"))  # Resistance in Ohm
        L = float(request.form.get("L"))  # Inductance in H
        V_backemf = float(request.form.get("V_backemf"))  # Back EMF voltage in V
        i_a_0 = 5.0  # Initial current fixed to 5.0 A

        ## Run the simulation
        filename1,filename2 = simPowerConvControlSyst(
            P_req,
            Q_req,
            V_rms_req,
            control_horizon,
            t_0,
            i_a_0,
            sampling_time,
            sim_time,
            R,
            L,
            V_backemf,
            f_ref
        )

        return render_template("index.html",
            filename1=url_for("static", filename=f"plots/{filename1}"),
            filename2=url_for("static", filename=f"plots/{filename2}"),
            uuid=uuid.uuid4().hex
        )
    return render_template("index.html", filename1=None, filename2=None,
            uuid=uuid.uuid4().hex)

if __name__ == "__main__":
    app.run(debug=True)
