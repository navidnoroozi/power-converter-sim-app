class Inverter():
    def __init__(self,V_dc=400.0):
        """
        Initializes the Inverter with a given DC voltage.
        """
        self.V_dc = V_dc

    def generateOutputVoltage(self,s_seq):
        """
        Generates the output voltage trajectory based on the switching signal sequence.
        Parameters:
        - s_seq: Sequence of switching signals (list or array of binary values).
        Returns:
        - v_aN_trajectory: List of output voltages corresponding to the switching signals.
        """
        v_aN_trajectory = []
        for s in s_seq:
            v_aN = self.V_dc * (2*s - 1) / 2.0
            v_aN_trajectory.append(v_aN)
        
        return v_aN_trajectory