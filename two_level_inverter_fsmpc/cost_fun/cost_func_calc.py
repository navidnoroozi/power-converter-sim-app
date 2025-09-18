import cvxpy as cp
import numpy as np

## Cost function calculation for the two-level inverter MPC

class CostFunction:
    def __init__(self, stage_func):
        """
        Initializes the Cost Function.

        Parameters:
        - stage_func: Function \ell(.,.) describng the stage cost.
        """
        self.stage_func = stage_func
        
    def calculateCostFunc(self,i_a_0,t_0,cont_horizon,s_seq,pwm_or_inverter,load,currentReference,mpc_method,with_pwm=False):
        """
        Recursively sums the stage costs over the control horizon.

        Parameters:
        - i_a_0: initial load current phase_a
        - t_0: initial time
        - cont_horizon: control horizon
        - s_seq: sequence of control inputs from MPC
        - pwm: instance of PWM class
        - load: instance of Load class
        - currentReference: instance of CurrentReference class

        Returns:
        - The total cost over the control horizon.
        """
        cost_func = 0.0
        if with_pwm:
            pwm = pwm_or_inverter
            v_an_tarj, _, _ = pwm.generateGatingSignals(s_seq,t_0)
        else:
            inverter = pwm_or_inverter
            v_an_tarj = inverter.generateOutputVoltage(s_seq,mpc_method)
        i_a_traj = load.calculateLoadDynamics(i_a_0,v_an_tarj,t_0)
        i_a_ref_traj = currentReference.generateRefTrajectory(t_0)
        if mpc_method == 'PWM':
            for i in range(cont_horizon):
                cost_func += self.stage_func(i_a_traj[i],i_a_ref_traj[i])
        else:
            i_a_traj = np.array(i_a_traj[:-1])
            i_a_ref_traj = np.array(i_a_ref_traj)
            cost_func = cp.sum_squares(i_a_ref_traj - i_a_traj)  
              
        return cost_func