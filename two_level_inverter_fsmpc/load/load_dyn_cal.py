import math

class Load():
    def __init__(self,R=10.0,L=0.005,V_backemf=0.0,f_backemf=50.0):
        """
        Initializes the Load with given resistance and inductance.

        Parameters:
        - R: Resistance in ohms.
        - L: Inductance in henrys.
        - V_backemf: Back electromotive force.
        - f_backemf: Frequency of back EMF.
        """
        self.R = R
        self.L = L
        self.V_backemf = V_backemf
        self.f_backemf = f_backemf

    def calculateLoadDynamics(self,i_a_0,v_an_tarj,t_0,sampling_rate=1e-4):
        """
        Calculates the load current trajectory based on the applied voltage.

        Parameters:
        - i_a_0: Initial load current.
        - v_an_tarj: Applied voltage trajectory.
        - t_0: Initial time.
        - sampling_rate: Time step for simulation.

        Returns:
        - Load current trajectory as a list of float.
        """
        i_a_traj = []
        t = t_0
        i_a = i_a_0
        for v_an in v_an_tarj:
            v_backemf = self.V_backemf * math.sin(2 * math.pi * self.f_backemf * t)
            di_dt = (v_an - v_backemf - self.R * i_a) / self.L
            i_a_next = i_a + di_dt * sampling_rate
            i_a_traj.append(i_a_next)
            i_a = i_a_next
            t += sampling_rate
        return i_a_traj