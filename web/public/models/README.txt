Drop skinned rigs here (.glb, .gltf, .fbx) and register each file in manifest.json — otherwise nothing appears in Character.

Filenames with spaces are fine; use %-encoding in JSON URL (e.g. Lola%20B%20Styperek.fbx).

Viewer: load a BVH first — the procedural rig and character mesh only mount after motion is parsed.

Suggested:
- Rig with T/A-pose compatible with Mixamo-style bone names when prefix is stripped.
- Match exporter BVH names: Hips, Spine, Spine1, Neck, Head, LeftShoulder, LeftArm, LeftForeArm,
  LeftHand, RightShoulder, RightArm, RightForeArm, RightHand,
  LeftUpLeg, LeftLeg, LeftFoot, LeftToeEnd, RightUpLeg, RightLeg, RightFoot, RightToeEnd.

If a URL 404s, check the browser console — the skeleton view still works.
