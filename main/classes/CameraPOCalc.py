import numpy as np
from math import sin, cos, tan, atan2

class CameraPOCalc:
    def __init__(self, W=1920, H=1080,
                 alpha_h=np.radians(60),
                 alpha_v=np.radians(60),
                 theta_p_bounds=(np.radians(0), np.radians(359)),
                 theta_r_bounds=(np.radians(0), np.radians(359))):
        """
        카메라 사양 및 제약 조건 초기화
        W, H: 해상도 (pixels)
        alpha_h, alpha_v: 수평/수직 FOV (radians)
        theta_p_bounds: pitch 허용 범위
        theta_r_bounds: roll 허용 범위
        """
        self.W, self.H = W, H
        self.alpha_h, self.alpha_v = alpha_h, alpha_v
        self.theta_p_bounds = theta_p_bounds
        self.theta_r_bounds = theta_r_bounds
        self.K = self._intrinsics_from_fov(W, H, alpha_h, alpha_v)

    # ---------------- 내부 유틸 ----------------
    def _R_from_ypr(self, yaw, pitch, roll):
        cy, sy = cos(yaw), sin(yaw)
        cp, sp = cos(pitch), sin(pitch)
        cr, sr = cos(roll), sin(roll)
        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]], dtype=float)
        Ry = np.array([[ cp, 0, sp],
                       [  0, 1,  0],
                       [-sp, 0, cp]], dtype=float)
        Rx = np.array([[1,   0,    0],
                       [0,  cr, -sr],
                       [0,  sr,  cr]], dtype=float)
        return Rz @ Ry @ Rx

    def _intrinsics_from_fov(self, W, H, alpha_h, alpha_v):
        fx = (W/2.0) / np.tan(alpha_h/2.0)
        fy = (H/2.0) / np.tan(alpha_v/2.0)
        cx = W/2.0; cy = H/2.0
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0,  0,  1]], dtype=float)

    def _project_points(self, Ps, C, yaw, pitch, roll):
        R = self._R_from_ypr(yaw, pitch, roll)
        Pc = (R @ (Ps - C).T).T
        x, y, z = Pc[:,0], Pc[:,1], Pc[:,2]
        u = self.K[0,0] * (x/z) + self.K[0,2]
        v = self.K[1,1] * (y/z) + self.K[1,2]
        return u, v, z

    def _smin(self, values, tau=2.0):
        vals = np.asarray(values)
        return -tau * np.log(np.sum(np.exp(-vals / tau)))

    def _softplus(self, t, beta=10.0):
        return np.log1p(np.exp(beta*t)) / beta

    def _objective(self, x, Ps,
                   tau=2.0, beta=40.0,
                   lambda_z=10.0, lambda_theta=10.0):
        C = np.array([x[0], x[1], x[2]], dtype=float)
        yaw, pitch, roll = x[3], x[4], x[5]
        try:
            u, v, z = self._project_points(Ps, C, yaw, pitch, roll)
        except Exception:
            return 1e6
        margins = np.vstack([u, self.W - u, v, self.H - v])
        per_corner_min = np.min(margins, axis=0)
        soft_margin = self._smin(per_corner_min, tau=tau)
        depth_pen = np.sum(self._softplus(-z, beta=beta))
        pmin, pmax = self.theta_p_bounds
        rmin, rmax = self.theta_r_bounds
        pen_pitch = self._softplus(pitch-pmax, beta=beta) + self._softplus(pmin-pitch, beta=beta)
        pen_roll  = self._softplus(roll-rmax, beta=beta)  + self._softplus(rmin-roll, beta=beta)
        J = -soft_margin + lambda_z*depth_pen + lambda_theta*(pen_pitch+pen_roll)
        if C[2] <= 0: J += 1e3 + 100*abs(C[2])
        return float(J)

    def _optimize(self, Ps, n_restarts=8, iters=800):
        rect_center = np.mean(Ps, axis=0)
        a_x = np.max(Ps[:,0]) - np.min(Ps[:,0])
        a_y = np.max(Ps[:,1]) - np.min(Ps[:,1])
        h0 = max(a_x/2/np.tan(self.alpha_h/2),
                 a_y/2/np.tan(self.alpha_v/2), 1.0)
        rng = np.random.default_rng(seed=1234)
        best_x, best_J = None, 1e9
        for _ in range(n_restarts):
            x0 = np.array([rect_center[0]+rng.normal(scale=a_x),
                           rect_center[1]+rng.normal(scale=a_y),
                           h0+ rng.normal(scale=0.5*h0),
                           rng.normal(scale=0.5),
                           -1.0+ rng.normal(scale=0.2),
                           rng.normal(scale=0.1)])
            cur_x, cur_J = x0.copy(), self._objective(x0, Ps)
            step = np.array([0.2*a_x, 0.2*a_y, 0.3*h0, 0.5,0.3,0.1])
            for _ in range(iters):
                cand = cur_x + rng.normal(scale=step)
                if cand[2] <= 0: cand[2] = max(0.05, cur_x[2]*0.5)
                cand_J = self._objective(cand, Ps)
                if cand_J < cur_J:
                    cur_x, cur_J = cand, cand_J
                    step *= 0.995
            if cur_J < best_J:
                best_x, best_J = cur_x.copy(), cur_J
        return best_x, best_J

    # ---------------- 공개 API ----------------
    def solve(self, a=None, b=None, O=None, phi=None, P_list=None,
              n_restarts=12, iters=1200):
        if P_list is None:
            if a is None or b is None or O is None or phi is None:
                raise ValueError("Either P_list or (a,b,O,phi) must be provided.")
            X0,Y0,_ = O
            Rrect = np.array([[np.cos(phi), -np.sin(phi), 0],
                              [np.sin(phi),  np.cos(phi), 0],
                              [0,0,1]])
            p_local = np.array([[0,0,0],[a,0,0],[a,b,0],[0,b,0]])
            P_list = (Rrect @ p_local.T).T + np.array(O)
        Ps = np.asarray(P_list,float)
        best_x, best_J = self._optimize(Ps, n_restarts, iters)
        if best_x is None:
            return {"success": False, "best_J": best_J}
        C = best_x[:3].tolist()
        yaw,pitch,roll = best_x[3], best_x[4], best_x[5]
        u,v,z = self._project_points(Ps, np.array(C), yaw, pitch, roll)
        return {"success": True, "C": C,
                "yaw": float(yaw), "pitch": float(pitch), "roll": float(roll),
                "diagnostics": {"u": u.tolist(),"v": v.tolist(),"z": z.tolist(),
                                "soft_J": best_J}}