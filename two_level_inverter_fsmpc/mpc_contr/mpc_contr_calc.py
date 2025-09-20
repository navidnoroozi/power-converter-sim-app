
import math
from typing import List, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except Exception:
    _HAS_GUROBI = False

class MPCSSolver:
    """
    Robust FS-MPC solver wrapper for a single-phase two-level converter.
    Primary: MILP with Gurobi (if available).
    Fallback 1: Finite-set brute-force enumeration over 2^N sequences (N = horizon).
    Fallback 2: Safe-mode PI step and/or repeat-last-action.
    """
    def __init__(self, cont_horizon=1, lambda_switch=0.1, timelimit=0.001, mipgap=0.05, threads=1,
                 safe_Kp=0.2, safe_Ki=50.0, duty_saturation=1.0):
        self.cont_horizon = int(cont_horizon)
        self.lambda_switch = float(lambda_switch)
        self.timelimit = float(timelimit) if timelimit is not None else None
        self.mipgap = float(mipgap) if mipgap is not None else None
        self.threads = int(threads) if threads is not None else None
        # Safe-mode PI on current error
        self.safe_Kp = float(safe_Kp)
        self.safe_Ki = float(safe_Ki)
        self.duty_saturation = float(duty_saturation)
        self._int_err = 0.0
        self._last_action = 1  # default +1 state

    # --- Convenience: predict one-step current with Euler ---
    @staticmethod
    def _i_next(i, v_inv, v_backemf, R, L, Ts):
        di_dt = (v_inv - v_backemf - R*i) / L
        return i + di_dt * Ts

    def _make_refs(self, currentReference, t0) -> List[float]:
        return list(currentReference.generateRefTrajectory(t0))

    def _make_vbemf_seq(self, f_bemf, V_bemf, t0, Ts, N):
        seq = []
        for k in range(N):
            t = t0 + k*Ts
            seq.append(V_bemf * math.sin(2*math.pi*f_bemf*t))
        return seq

    def _cost(self, i_seq, iref_seq, s_seq, s0):
        J = 0.0
        for k in range(len(iref_seq)):
            e = i_seq[k] - iref_seq[k]
            J += e*e
            # penalize switch movement relative to "previous plan" s0
            sk0 = 1 if (k < len(s0) and s0[k]) else 0
            J += self.lambda_switch * ( (s_seq[k] - sk0) ** 2 )
        return J

    def _bruteforce(self, inverter, load, currentReference, t0, i0, s0) -> Tuple[List[int], float]:
        """Exhaustive enumeration of 2^N sequences (N small)."""
        N = self.cont_horizon
        Ts = currentReference.sampling_rate
        Vdc = inverter.V_dc
        R, L = load.R, load.L
        # reference and back-EMF
        iref = self._make_refs(currentReference, t0)
        vbemf = self._make_vbemf_seq(load.f_backemf, load.V_backemf, t0, Ts, N)

        best_J = float('inf')
        best_seq = [self._last_action] * N

        # Generate all sequences by counting 0..(2^N-1)
        total = 1 << N
        for mask in range(total):
            s_seq = [ (mask >> k) & 1 for k in range(N) ]  # LSB = step 0
            # Predict current over horizon
            i = i0
            i_seq = []
            for k in range(N):
                v_inv = Vdc * (2*s_seq[k]-1) / 2.0
                i = self._i_next(i, v_inv, vbemf[k], R, L, Ts)
                i_seq.append(i)
            J = self._cost(i_seq, iref, s_seq, s0)
            if J < best_J:
                best_J = J
                best_seq = s_seq

        return best_seq, best_J

    def _safe_mode(self, i0, iref0) -> int:
        """PI on current error -> duty -> choose nearest switch state."""
        e = iref0 - i0
        Ts_eff = 1e-4  # tiny stabilizing step for integral (won't matter much)
        self._int_err += e * Ts_eff
        duty = self.safe_Kp*e + self.safe_Ki*self._int_err
        # saturate
        duty = max(-self.duty_saturation, min(self.duty_saturation, duty))
        s = 1 if duty >= 0 else 0
        return s

    def solveMPC(self, inverter, load, currentReference, current_time, i_a_0, s0):
        N = self.cont_horizon
        Ts = currentReference.sampling_rate
        Vdc = inverter.V_dc
        R, L = load.R, load.L

        # quick single-step fallback if horizon is weird
        if N <= 0:
            return [self._last_action], 0.0

        # references and back-emf sequences
        iref = self._make_refs(currentReference, current_time)
        vbemf = self._make_vbemf_seq(load.f_backemf, load.V_backemf, current_time, Ts, N)

        # Try Gurobi (if available)
        if _HAS_GUROBI:
            try:
                m = gp.Model("fsmpc")
                m.Params.OutputFlag = 0
                if self.timelimit is not None:
                    m.Params.TimeLimit = self.timelimit
                if self.mipgap is not None:
                    m.Params.MIPGap = self.mipgap
                if self.threads is not None:
                    m.Params.Threads = self.threads
                # Numerical hygiene
                m.Params.NumericFocus = 3
                m.Params.ScaleFlag = 2

                # Variables
                s = m.addVars(N, vtype=GRB.BINARY, name="s")
                i = m.addVars(N+1, lb=-GRB.INFINITY, name="i")

                # Initial condition
                m.addConstr(i[0] == float(i_a_0))

                # Dynamics
                for k in range(N):
                    v_inv_k = (Vdc/2.0) * (2*s[k]-1)
                    rhs = i[k] + Ts * ((v_inv_k - vbemf[k] - R*i[k]) / L)
                    m.addConstr(i[k+1] == rhs, name=f"dynamics_{k}")

                # Objective
                obj = gp.QuadExpr(0.0)
                for k in range(N):
                    obj += (i[k+1] - float(iref[k]))*(i[k+1] - float(iref[k]))
                    sk0 = 1 if (k < len(s0) and s0[k]) else 0
                    obj += self.lambda_switch * ( (s[k] - sk0) * (s[k] - sk0) )

                m.setObjective(obj, GRB.MINIMIZE)
                m.optimize()

                if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
                    # If time limit, Gurobi still gives best incumbent if any
                    if m.SolCount > 0:
                        s_seq = [int(round(s[k].X)) for k in range(N)]
                        self._last_action = s_seq[0]
                        return s_seq, float(m.ObjVal if m.SolCount>0 else 0.0)

            except Exception as _e:
                # Will fall back below
                pass

        # Fallback 1: brute-force enumeration
        try:
            seq, J = self._bruteforce(inverter, load, currentReference, current_time, i_a_0, s0)
            self._last_action = seq[0]
            return seq, float(J)
        except Exception:
            # Fallback 2: safe-mode single step, repeat across horizon
            s = self._safe_mode(i_a_0, iref[0])
            self._last_action = s
            return [s]*N, 1e9  # big cost to indicate fallback used
