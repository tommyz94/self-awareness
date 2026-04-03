from __future__ import annotations


# Neutral arm-rest pose for MuJoCo humanoids tuned for "arms along the body".
# Values are chosen so upper-arm + forearm direction points mostly downward in
# this specific humanoid kinematic chain (avoids both forward boxing guard and
# backward folded posture).
MUJOCO_ARM_REST_POSE = {
    "shoulder1_right": 0.89,
    "shoulder2_right": -1.10,
    "elbow_right": -0.56,
    "shoulder1_left": 0.89,
    "shoulder2_left": -1.10,
    "elbow_left": -0.56,
}
