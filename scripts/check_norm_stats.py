import json

d = json.load(open("assets/vtla_tactile_posttrain/task1_tuberack_train/norm_stats.json"))
state = d["norm_stats"]["state"]["mean"][10:19]
action = d["norm_stats"]["actions"]["mean"][10:19]
print("state[10:19] mean:", state)
print("action[10:19] mean:", action)
xyz_small = all(abs(v) < 50 for v in action[0:3])
rot_near_identity = abs(action[3] - 1) < 0.3 and abs(action[7] - 1) < 0.3
print("action xyz looks like a small delta (not absolute ~370mm):", xyz_small)
print("action rot6d looks near-identity (correct-delta signature):", rot_near_identity)
