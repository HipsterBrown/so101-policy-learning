"""Print box <collision> XML for the SO-101 jaws, sized from the vendored STLs.

Each jaw is split at a cut plane into a bracket box and a finger box, both in the link frame.
Cut planes were picked from mesh slices (see the Phase 0 plan).
"""
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ASSETS = Path(__file__).resolve().parents[1] / "assets/so101/assets"


def stl_vertices(name, xyz, rpy):
    d = (ASSETS / name).read_bytes()
    n = struct.unpack("<I", d[80:84])[0]
    tri = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    v = tri["v"].reshape(-1, 3).astype(float)
    return v @ R.from_euler("xyz", rpy).as_matrix().T + np.array(xyz)  # URDF rpy = fixed-axis XYZ


def box_xml(v, label):
    lo, hi = v.min(0), v.max(0)
    c, s = (lo + hi) / 2, hi - lo
    return (f'    <!-- {label} -->\n    <collision>\n'
            f'      <origin xyz="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" rpy="0 0 0"/>\n'
            f'      <geometry><box size="{s[0]:.5f} {s[1]:.5f} {s[2]:.5f}"/></geometry>\n    </collision>')


# Origins copied from the <collision> elements in so101_new_calib.urdf.
fixed = stl_vertices("wrist_roll_follower_so101_v1.stl", (0, -0.000218214, 0.000949706), (-3.14159, 0, 0))
moving = stl_vertices("moving_jaw_so101_v1.stl", (0, 0, 0.0189), (0, 0, 0))

FIXED_CUT_Z = -0.045   # gripper_link: finger is below this
MOVING_CUT_Y = -0.03   # moving jaw link: finger is beyond this (-y)

print("<!-- gripper_link: replace the wrist_roll_follower_so101_v1 <collision> with -->")
print(box_xml(fixed[fixed[:, 2] >= FIXED_CUT_Z], "fixed jaw bracket"))
print(box_xml(fixed[fixed[:, 2] < FIXED_CUT_Z], "fixed jaw finger"))
print("<!-- moving_jaw_so101_v1_link: replace its <collision> with -->")
print(box_xml(moving[moving[:, 1] >= MOVING_CUT_Y], "moving jaw hinge"))
print(box_xml(moving[moving[:, 1] < MOVING_CUT_Y], "moving jaw finger"))
