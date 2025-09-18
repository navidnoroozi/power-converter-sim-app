import numpy as np
import math

class CurrentReference():
    def __init__(self,sampling_rate,cont_horizon,I_ref_peak=5.0,phi_ref=0.0,f_ref=50.0):
        """
        Initializes the Current Reference generator.

        Parameters:
        - sampling_rate: Time step for simulation.
        - cont_horizon: Control horizon.
        - I_ref_peak: Peak value of the reference current.
        - phi_ref: Phase angle of the reference current in radians.
        - f_ref: Frequency of the reference current in Hz.
        """
        self.sampling_rate = sampling_rate
        self.cont_horizon = cont_horizon
        self.I_ref_peak = I_ref_peak
        self.phi_ref = phi_ref
        self.f_ref = f_ref

    def generateRefTrajectory(self,t_0):
        """
        Generates the reference current trajectory over the control horizon.

        Parameters:
        - t_0: Initial time.

        Returns:
        - Reference current trajectory as a list.
        """
        i_a_ref_traj = []
        for i in range(self.cont_horizon):
            t = t_0 + i * self.sampling_rate
            i_a_ref = self.I_ref_peak * math.cos(2 * math.pi * self.f_ref * t - self.phi_ref)
            i_a_ref_traj.append(i_a_ref)
        return np.array(i_a_ref_traj)