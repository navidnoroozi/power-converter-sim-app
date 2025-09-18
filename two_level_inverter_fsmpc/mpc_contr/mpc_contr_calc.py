from gurobipy import Model, GRB, QuadExpr
import math

class MPCSSolver:
    def __init__(self, cont_horizon=5, lambda_switch=0.1, timelimit=None, mipgap=None):
        """
        Gurobi-based MPC solver for a two-level inverter with RL load.
        - cont_horizon: prediction horizon N
        - lambda_switch: weight on switch movement (s - s_prev)^2
        - timelimit: optional time limit in seconds
        - mipgap: optional relative MIP gap
        """
        self.cont_horizon = cont_horizon
        self.lambda_switch = lambda_switch
        self.timelimit = timelimit
        self.mipgap = mipgap

    def solveMPC(self, inverter, load, currentReference, current_time, i_a_0, s0):
        """
        Solves the MIQP with binary switching s[k] ∈ {0,1} and linear dynamics:
            i[k+1] = (1 - Ts*R/L) * i[k] + (Ts/L * Vdc) * s[k] + Ts/L * (-Vdc/2 - e[k])
        Objective:
            sum_{k=1..N} (i[k] - i_ref[k-1])^2 + lambda * sum_{k=0..N-1} (s[k] - s0[k])^2
        Returns (s_seq, obj_value) where s_seq is a list of 0/1.
        """
        N = self.cont_horizon
        Ts = currentReference.sampling_rate
        R = load.R
        L = load.L
        Vdc = inverter.V_dc

        # Precompute references and back-EMF
        i_ref = list(currentReference.generateRefTrajectory(current_time))  # length N
        e = [load.V_backemf * math.sin(2 * math.pi * load.f_backemf * (current_time + k*Ts)) for k in range(N)]

        a = 1.0 - Ts * R / L
        b = Ts * Vdc / L
        c = [Ts / L * (-Vdc/2.0 - e_k) for e_k in e]

        m = Model("fsmpc")
        m.Params.OutputFlag = 0
        if self.timelimit is not None:
            m.Params.TimeLimit = float(self.timelimit)
        if self.mipgap is not None:
            m.Params.MIPGap = float(self.mipgap)

        # Variables
        s = m.addVars(N, vtype=GRB.BINARY, name="s")
        i = m.addVars(N+1, lb=-GRB.INFINITY, name="i")

        # Initial condition
        m.addConstr(i[0] == i_a_0, name="i0")

        # Dynamics
        for k in range(N):
            m.addConstr(i[k+1] == a * i[k] + b * s[k] + c[k], name=f"dyn_{k}")

        # Objective: tracking + switching regularization
        obj = QuadExpr()
        for k in range(1, N+1):
            obj.add((i[k] - i_ref[k-1]) * (i[k] - i_ref[k-1]))
        for k in range(N):
            diff = s0[k] - 1 if isinstance(s0[k], bool) and s0[k] else s0[k]
            # s0 is list of bools; (s - s0)^2 simplifies but we keep quadratic form:
            obj.add(self.lambda_switch * (s[k] - (1 if s0[k] else 0)) * (s[k] - (1 if s0[k] else 0)))

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        # Extract results
        s_seq = [int(round(s[k].X)) for k in range(N)]
        return s_seq, float(m.ObjVal)
