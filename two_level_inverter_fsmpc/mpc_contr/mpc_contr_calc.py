import cvxpy as cp

class MPCSSolver:
    def __init__(self,cont_horizon=5):
        """
        Initializes the MPC Solver.

        Parameters:
        - cont_horizon: Control horizon.
        """
        self.cont_horizon = cont_horizon

    def solveMPC(self,inverter,load,currentReference,current_time,i_a_0,s0):
        """
        Solves the MPC optimization problem to find the optimal switching signal sequence.

        Parameters:
        - inverter: Instance of Inverter class.
        - load: Instance of Load class.
        - currentReference: Instance of CurrentReference class.
        - current_time: Current time.
        - i_a_0: Initial load current.
        - s0: Initial guess for switching signal sequence.

        Returns:
        - Optimal switching signal sequence as a numpy array.
        - Value of the cost function at the optimum.
        """
        s_seq = cp.Variable(self.cont_horizon, boolean=True)  # Binary decision variables
        # if s_seq.value is None:
        s_seq.value = s0
        
        t_0 = current_time
        v_an_tarj = inverter.generateOutputVoltage(s_seq)
        i_a_traj = load.calculateLoadDynamics(i_a_0,v_an_tarj,t_0)
        i_a_ref_traj = cp.hstack(currentReference.generateRefTrajectory(t_0))
        i_a_traj = cp.hstack(i_a_traj)
        i_delta = i_a_traj - i_a_ref_traj
        s_delta = s_seq - s0
        cost_func = cp.sum_squares(i_delta) + 0.1*cp.sum_squares(s_delta)
        
        prob = cp.Problem(cp.Minimize(cost_func)) # , constraints
        
        prob.solve(solver=cp.ECOS_BB)  # small problems only
        
        return s_seq, prob.value