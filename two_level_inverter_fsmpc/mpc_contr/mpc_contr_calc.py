
import math
from typing import List, Tuple

# Gurobi is optional; FS-MPC runs brute-force first by default.
try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except Exception:
    _HAS_GUROBI = False

class MPCSSolver:
    """
    FS-MPC solver for single-phase two-level converter with robust fallbacks.
    Primary: finite-set brute force (2^N sequences).
    Optional: Gurobi MILP if use_gurobi=True.
    Safe fallback: PI-based single-step if all else fails.
    """
    def __init__(self, cont_horizon=1, lambda_switch=0.1, timelimit=0.001, mipgap=0.05, threads=1,
                 use_gurobi=False, safe_Kp=0.2, safe_Ki=50.0, duty_saturation=1.0):
        self.cont_horizon = int(cont_horizon)
        self.lambda_switch = float(lambda_switch)
        self.timelimit = float(timelimit) if timelimit is not None else None
        self.mipgap = float(mipgap) if mipgap is not None else None
        self.threads = int(threads) if threads is not None else None
        self.use_gurobi = bool(use_gurobi)

        # Safe-mode PI on current error
        self.safe_Kp = float(safe_Kp)
        self.safe_Ki = float(safe_Ki)
        self.duty_saturation = float(duty_saturation)
        self._int_err = 0.0
        self._last_action = 1  # default +1 state

    # --- Utilities ---
    @staticmethod
    def _i_next(i, v_inv, v_backemf, R, L, Ts):
        di_dt = (v_inv - v_backemf - R*i) / L
        return i + di_dt * Ts

    def _make_refs(self, currentReference, t0) -> List[float]:
        arr = currentReference.generateRefTrajectory(t0)
        return [float(x) for x in arr]

    def _make_vbemf_seq(self, f_bemf, V_bemf, t0, Ts, N):
        return [V_bemf * math.sin(2*math.pi*f_bemf*(t0 + k*Ts)) for k in range(N)]

    def _cost(self, i_seq, iref_seq, s_seq, s0):
        J = 0.0
        for k in range(len(iref_seq)):
            e = i_seq[k] - iref_seq[k]
            J += e*e
            sk0 = 1 if (k < len(s0) and s0[k]) else 0
            J += self.lambda_switch * ((s_seq[k] - sk0) ** 2)
        return J

    # --- Primary solver: brute-force ---
    def _bruteforce(self, inverter, load, currentReference, t0, i0, s0) -> Tuple[List[int], float]:
        N = self.cont_horizon
        Ts = currentReference.sampling_rate
        Vdc = inverter.V_dc
        R, L = load.R, load.L

        iref = self._make_refs(currentReference, t0)
        vbemf = self._make_vbemf_seq(load.f_backemf, load.V_backemf, t0, Ts, N)

        best_J = float('inf')
        best_seq = [self._last_action] * N
        total = 1 << N  # 2^N

        for mask in range(total):
            s_seq = [(mask >> k) & 1 for k in range(N)]  # LSB is step 0
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

    # --- Optional solver: Gurobi MILP/QP (skipped unless use_gurobi=True) ---
    def _gurobi_solve(self, inverter, load, currentReference, t0, i0, s0):
        if not (_HAS_GUROBI and self.use_gurobi):
            raise RuntimeError("Gurobi path disabled or unavailable")
        N = self.cont_horizon
        Ts = currentReference.sampling_rate
        Vdc = inverter.V_dc
        R, L = load.R, load.L
        iref = self._make_refs(currentReference, t0)
        vbemf = self._make_vbemf_seq(load.f_backemf, load.V_backemf, t0, Ts, N)

        m = gp.Model("fsmpc")
        m.Params.OutputFlag = 0
        if self.timelimit is not None: m.Params.TimeLimit = self.timelimit
        if self.mipgap is not None: m.Params.MIPGap = self.mipgap
        if self.threads is not None: m.Params.Threads = self.threads
        m.Params.NumericFocus = 3
        m.Params.ScaleFlag = 2

        s = m.addVars(N, vtype=GRB.BINARY, name="s")
        i = m.addVars(N+1, lb=-GRB.INFINITY, name="i")
        m.addConstr(i[0] == float(i0))

        for k in range(N):
            v_inv_k = (Vdc/2.0) * (2*s[k]-1)
            rhs = i[k] + Ts * ((v_inv_k - vbemf[k] - R*i[k]) / L)
            m.addConstr(i[k+1] == rhs, name=f"dynamics_{k}")

        obj = gp.QuadExpr(0.0)
        for k in range(N):
            diff = i[k+1] - float(iref[k])
            obj.add(diff*diff)
            sk0 = 1 if (k < len(s0) and s0[k]) else 0
            obj.add(self.lambda_switch * (s[k]-sk0)*(s[k]-sk0))

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
            s_seq = [int(round(s[k].X)) for k in range(N)]
            return s_seq, float(m.ObjVal)
        raise RuntimeError(f"Gurobi failed with status {m.Status}")

    # --- Safe mode PI ---
    def _safe_mode(self, i0, iref0) -> int:
        e = iref0 - i0
        Ts_eff = 1e-4
        self._int_err += e * Ts_eff
        duty = self.safe_Kp*e + self.safe_Ki*self._int_err
        duty = max(-self.duty_saturation, min(self.duty_saturation, duty))
        return 1 if duty >= 0 else 0

    # --- Public API ---
    def solveMPC(self, inverter, load, currentReference, current_time, i_a_0, s0):
        N = self.cont_horizon
        if N <= 0:
            return [self._last_action], 0.0

        # 1) Primary: brute force
        try:
            seq, J = self._bruteforce(inverter, load, currentReference, current_time, i_a_0, s0)
            self._last_action = seq[0]
            return seq, float(J)
        except Exception:
            pass

        # 2) Optional: Gurobi (only if explicitly enabled)
        try:
            seq, J = self._gurobi_solve(inverter, load, currentReference, current_time, i_a_0, s0)
            self._last_action = seq[0]
            return seq, float(J)
        except Exception:
            pass

        # 3) Safe mode
        # Create a constant sequence from one safe decision
        iref = self._make_refs(currentReference, current_time)
        s = self._safe_mode(i_a_0, iref[0])
        self._last_action = s
        return [s]*N, 1e9
