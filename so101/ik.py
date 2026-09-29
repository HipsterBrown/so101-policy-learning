"""FK/IK on the Viam SO-101 kinematic chain (so101.json), shared with the real arm."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

VIAM_JSON = Path(__file__).resolve().parents[1] / "assets/viam/so101.json"
# Grasp point between the jaw tips, in the Viam "tool" frame (metres).
# From ../so-101/internal/geometry/gripper.go GripperTCPPose. Calibration knob.
# Note: this is where the jaws meet when CLOSED, i.e. on the fixed jaw's contact face.
GRIPPER_TCP = np.array([0.006835, 0.0, 0.0999])
# To grasp a block, put its centre one half-width off the fixed-jaw face, toward the moving jaw (tool -x):
# shift the commanded TCP +x in the tool frame by 0.006835 - (0.0079 - 0.015). Calibration knob.
GRASP_X_OFFSET = 0.0139
LINKS = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "tool"]


@dataclass
class Chain:
    fixed: list  # 6 fixed 4x4 transforms: base, then one after each joint (last is tool)
    limits: np.ndarray  # (5, 2) radians


def _link_tf(link):
    T = np.eye(4)
    t = link.get("translation", {})
    T[:3, 3] = [t.get(k, 0.0) / 1000.0 for k in "xyz"]
    e = link["orientation"]["value"]
    # Viam euler_angles == URDF rpy: R = Rz(yaw) Ry(pitch) Rx(roll)
    T[:3, :3] = R.from_euler("ZYX", [e["yaw"], e["pitch"], e["roll"]]).as_matrix()
    return T


def load_chain(path=VIAM_JSON) -> Chain:
    cfg = json.loads(Path(path).read_text())
    links = {l["id"]: l for l in cfg["links"]}
    lim = np.radians([[j["min"], j["max"]] for j in cfg["joints"]])
    return Chain([_link_tf(links[n]) for n in LINKS], lim)


def fk(chain: Chain, q, tcp=GRIPPER_TCP) -> np.ndarray:
    """Pose (4x4) of the TCP in the arm base frame. tcp=None gives the bare tool frame."""
    T = chain.fixed[0].copy()
    for i in range(5):
        Rz = np.eye(4)
        Rz[:3, :3] = R.from_rotvec([0, 0, q[i]]).as_matrix()
        T = T @ Rz @ chain.fixed[i + 1]
    if tcp is not None:
        T = T.copy()
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ tcp
    return T


def _pose_err(T, target):
    pos_mm = np.linalg.norm(T[:3, 3] - target[:3, 3]) * 1000
    rot_deg = np.degrees(np.linalg.norm(R.from_matrix(target[:3, :3].T @ T[:3, :3]).as_rotvec()))
    return pos_mm, rot_deg


def ik(chain: Chain, target, q_init, rot_weight=1.0, attempts=10, seed=0):
    """Bounded least squares on position (mm) + weighted orientation (deg).

    Returns (q, ok). ok: pos < 1 mm, and rot < 1 deg when rot_weight >= 1.
    With 0 < rot_weight < 1 orientation is a soft preference; callers check their own tilt.
    """
    lo, hi = chain.limits[:, 0], chain.limits[:, 1]
    rng = np.random.default_rng(seed)

    def resid(q):
        T = fk(chain, q)
        r = [(T[:3, 3] - target[:3, 3]) * 1000]
        if rot_weight > 0:
            rv = R.from_matrix(target[:3, :3].T @ T[:3, :3]).as_rotvec()
            r.append(rot_weight * np.degrees(rv))
        return np.concatenate(r)

    best = None
    for a in range(attempts):
        q0 = np.clip(q_init, lo, hi) if a == 0 else rng.uniform(lo, hi)
        q = least_squares(resid, q0, bounds=(lo, hi)).x
        pos, rot = _pose_err(fk(chain, q), target)
        ok = pos < 1.0 and (rot_weight < 1 or rot < 1.0)
        if ok:
            return q, True
        score = pos + rot_weight * rot
        if best is None or score < best[1]:
            best = (q, score)
    return best[0], False


def top_down(p, yaw):
    """Target pose at point p with the approach axis (tool +Z) pointing down, jaw yaw about world Z."""
    T = np.eye(4)
    T[:3, :3] = R.from_euler("ZYX", [yaw, np.pi, 0]).as_matrix()
    T[:3, 3] = p
    return T


def tilt_deg(T):
    """Angle between the approach axis and straight down."""
    return np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))


def grasp_pose(p, yaw, x_offset=GRASP_X_OFFSET):
    """Top-down TCP target that centres a block at p between the open jaws."""
    T = top_down(p, yaw)
    T[:3, 3] += T[:3, :3] @ np.array([x_offset, 0.0, 0.0])
    return T
