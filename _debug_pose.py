#!/usr/bin/env python3
"""Debug the BVH-export rotations on a single still image (yoga tree pose)."""
from __future__ import annotations
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fitness_app.pose import _ensure_pose_landmarker_model, landmarks_to_array, draw_pose_from_array
from fitness_app.mocap.mediapipe_joints import (
    landmarks_to_skeleton_joints, normalized_to_xyz, skeleton_joint_names_ordered,
)
from fitness_app.mocap.bvh_export import build_animation_from_joints_list, _root_basis_from_joints

OUT = "_debug"
os.makedirs(OUT, exist_ok=True)
IMG = "datasets/TestPose/pose_model.png"

# ---------------------------------------------------------------- 1. MediaPipe (IMAGE mode)
opts = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=_ensure_pose_landmarker_model()),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
)
lm = vision.PoseLandmarker.create_from_options(opts)
img = cv2.imread(IMG)
rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
assert res.pose_landmarks, "no pose detected"
norm = landmarks_to_array(res.pose_landmarks[0])            # (33,4) normalized image space
world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])  # (33,3) metric, Y-down

# annotated 2D overlay
ann = img.copy()
draw_pose_from_array(ann, norm)
cv2.imwrite(f"{OUT}/annotated.png", ann)

# ---------------------------------------------------------------- 2. diagnostics
xyz = normalized_to_xyz(norm)  # current pipeline: Y-up, X right, z weak
def avg_y(idx): return float(np.mean([xyz[i][1] for i in idx]))
print("=== RAW (normalized->xyz, Y-up) heights ===")
print(f"  nose.y     = {xyz[0][1]:+.3f}")
print(f"  shoulder.y = {avg_y([11,12]):+.3f}")
print(f"  hip.y      = {avg_y([23,24]):+.3f}")
print(f"  knee.y     = {avg_y([25,26]):+.3f}")
print(f"  ankle.y    = {avg_y([27,28]):+.3f}   (upright => nose>shoulder>hip>ankle)")

joints = landmarks_to_skeleton_joints(norm)
torso = joints["Spine1"] - joints["Hips"]
tn = torso / (np.linalg.norm(torso) + 1e-9)
print("\n=== torso (Spine1-Hips) ===")
print(f"  vector = ({torso[0]:+.3f},{torso[1]:+.3f},{torso[2]:+.3f})  angle_from_+Y = {np.degrees(np.arccos(np.clip(tn[1],-1,1))):.1f} deg")
B = _root_basis_from_joints(joints)
print("\n=== root basis columns [right, up, forward] ===")
for k, name in enumerate(["right", "up   ", "fwd  "]):
    v = B[:, k]
    print(f"  {name} = ({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})")
print(f"  det(basis) = {np.linalg.det(B):+.3f}  (should be +1 for a proper right-handed frame)")

print("\n=== raw z (depth) spread - weak hint, expect small/noisy ===")
for nm, i in [("nose",0),("L_sh",11),("R_sh",12),("L_hip",23),("R_hip",24),("L_ank",27),("R_ank",28)]:
    print(f"  {nm:5s} z={norm[i,2]:+.3f}")

# --- root basis from WORLD landmarks (the candidate fix): metric 3D, Y-down -> Y-up ---
wlYup = world * np.array([1.0, -1.0, 1.0])
wj = {"Hips": 0.5*(wlYup[23]+wlYup[24]), "LeftHip": wlYup[23], "RightHip": wlYup[24],
      "Spine1": 0.5*(wlYup[11]+wlYup[12])}
Bw = _root_basis_from_joints(wj)
print("\n=== root basis from WORLD landmarks [right, up, forward] (candidate fix) ===")
for k, name in enumerate(["right", "up   ", "fwd  "]):
    v = Bw[:, k]
    print(f"  {name} = ({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})")
print("  (good = right~+/-X with small Z, fwd~+/-Z; small Z in 'right' means correct yaw/facing)")

# ---------------------------------------------------------------- 3. export + FK (BOTH paths)
# minimal parser + FK matching web/src/bvh/poseSkeleton.ts exactly
def Rz(d): r=np.radians(d);c,s=np.cos(r),np.sin(r);m=np.eye(4);m[0,0]=c;m[0,1]=-s;m[1,0]=s;m[1,1]=c;return m
def Ry(d): r=np.radians(d);c,s=np.cos(r),np.sin(r);m=np.eye(4);m[0,0]=c;m[0,2]=s;m[2,0]=-s;m[2,2]=c;return m
def Rx(d): r=np.radians(d);c,s=np.cos(r),np.sin(r);m=np.eye(4);m[1,1]=c;m[1,2]=-s;m[2,1]=s;m[2,2]=c;return m
def Tr(v): m=np.eye(4);m[:3,3]=v;return m

def parse_bvh(text):
    toks = text.replace("{"," { ").replace("}"," } ").split()
    i=0; flat=[]; stack=[]
    while i < len(toks):
        t=toks[i]
        if t in ("ROOT","JOINT"):
            name=toks[i+1]; parent = stack[-1] if stack else -1
            flat.append({"name":name,"parent":parent,"offset":[0,0,0],"channels":[]}); i+=2; continue
        if t=="End":
            depth=0; i+=1
            while i<len(toks):
                if toks[i]=="{": depth+=1
                elif toks[i]=="}":
                    depth-=1
                    if depth==0: break
                i+=1
            i+=1; continue
        if t=="{": stack.append(len(flat)-1); i+=1; continue
        if t=="}": stack.pop(); i+=1; continue
        if t=="OFFSET": flat[-1]["offset"]=[float(toks[i+1]),float(toks[i+2]),float(toks[i+3])]; i+=4; continue
        if t=="CHANNELS": n=int(toks[i+1]); flat[-1]["channels"]=toks[i+2:i+2+n]; i+=2+n; continue
        if t=="MOTION": break
        i+=1
    mi=toks.index("Time:"); row=list(map(float,toks[mi+1:mi+1+sum(len(j['channels']) for j in flat)]))
    return flat, row

def fk(flat,row):
    wm=[None]*len(flat); cur=0
    for k,j in enumerate(flat):
        ch=j["channels"]; off=j["offset"]; vals=row[cur:cur+len(ch)]; cur+=len(ch)
        d={c:v for c,v in zip(ch,vals)}
        rot=Rz(d.get("Zrotation",0))@Ry(d.get("Yrotation",0))@Rx(d.get("Xrotation",0))
        local=(Tr([d["Xposition"],d["Yposition"],d["Zposition"]])@rot) if "Xposition" in d else (rot@Tr(off))
        p=j["parent"]; wm[k]= local if p<0 else wm[p]@local
    return wm

def export_fk(J, tag):
    hierarchy, motion = build_animation_from_joints_list([J], fps=25.0)
    bvh = hierarchy.rstrip()+"\nMOTION\nFrames: 1\nFrame Time: 0.04\n"+" ".join(f"{v:.6f}" for v in motion[0])+"\n"
    open(f"{OUT}/pose_{tag}.bvh","w").write(bvh)
    flat,row = parse_bvh(bvh); wm=fk(flat,row)
    return np.array([m[:3,3] for m in wm]), [j["name"] for j in flat], [j["parent"] for j in flat]

world4 = np.hstack([world, norm[:, 3:4]])                 # (33,4): world xyz + visibility
J_img = landmarks_to_skeleton_joints(norm, space="image")  # legacy (weak-z) path
J_wld = landmarks_to_skeleton_joints(world4, space="world")# NEW fix path
pos_i, names, parent = export_fk(J_img, "image")
pos_w, _, _ = export_fk(J_wld, "world")

# ---------------------------------------------------------------- 4. figure
def plot2d(ax, P, links, a, b, title, color, arrow=None):
    for u, v in links:
        ax.plot([P[u][a],P[v][a]],[P[u][b],P[v][b]],"-",color=color,lw=2.2)
    ax.plot(P[:,a], P[:,b], "k.", ms=4)
    if arrow is not None:
        o=arrow["o"]
        for vec,col in [(arrow["right"],"g"),(arrow["fwd"],"darkorange")]:
            ax.annotate("", xy=(o[a]+vec[a]*0.4,o[b]+vec[b]*0.4), xytext=(o[a],o[b]),
                        arrowprops=dict(arrowstyle="->",color=col,lw=2))
    ax.set_title(title); ax.set_aspect("equal","box"); ax.grid(alpha=0.3)
    ax.set_xlabel("XYZ"[a]); ax.set_ylabel("XYZ"[b])

links = [(i,p) for i,p in enumerate(parent) if p>=0]
hip = names.index("Hips")
arr_i = {"o":pos_i[hip], "right":B[:,0], "fwd":B[:,2]}
arr_w = {"o":pos_w[hip], "right":Bw[:,0], "fwd":Bw[:,2]}

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
axes[0,0].imshow(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)); axes[0,0].set_title("INPUT (upright tree pose)"); axes[0,0].axis("off")
plot2d(axes[0,1], pos_i, links, 0,1, "BEFORE (image z) - FRONT X-Y", "C3", arrow=arr_i)
plot2d(axes[0,2], pos_i, links, 0,2, "BEFORE (image z) - TOP X-Z (green=right orange=fwd)", "C3", arrow=arr_i)
axes[1,0].axis("off")
axes[1,0].text(0.02,0.5, "BEFORE = normalized image landmarks (weak z)\nAFTER  = MediaPipe world landmarks (metric 3D)\n\n"
                         "FRONT view: both should look upright.\nTOP view (bird's-eye): the green 'right' arrow\n"
                         "should point along X with little Z.\n\nIf AFTER's arrow is axis-aligned and BEFORE's\n"
                         "is spun ~45 deg, the world-landmark fix works.", fontsize=12, va="center")
plot2d(axes[1,1], pos_w, links, 0,1, "AFTER (world) - FRONT X-Y", "C0", arrow=arr_w)
plot2d(axes[1,2], pos_w, links, 0,2, "AFTER (world) - TOP X-Z (green=right orange=fwd)", "C0", arrow=arr_w)
plt.tight_layout(); plt.savefig(f"{OUT}/compare.png", dpi=95)
print(f"\nwrote {OUT}/compare.png, {OUT}/pose_image.bvh, {OUT}/pose_world.bvh")
