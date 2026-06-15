# BVH procedural viewer

Small web UI to inspect `motion.bvh` files emitted by [`../process_mocap.py`](../process_mocap.py) (Kalman smoothing + retarget pipeline).

## Setup

From this directory:

```bash
npm install
```

## Dev server

```bash
npm run dev
```

Then open the printed localhost URL and choose **BVH file** (`keypoints/<id>/motion.bvh`).

Drag-and-drop support is omitted; use the file picker input.

Use **orbit** (drag), **zoom** (scroll / pinch), and **Play/Pause**, **Scrub**, and **Speed** in the toolbar.

## Build

```bash
npm run build
npm run preview
```

## Tests

```bash
npm test
```

BVH parsing is covered by a synthetic two-bone snippet that follows the exporter's channel naming.

## Skinned characters (GLB)

1. Drop a **skinned** GLB under `public/models/` (Mixamo T-pose rigs work well).
2. List it in `public/models/manifest.json` with `id`, `label`, `url` (for example `"/models/you.glb"`).
3. Use `bonePrefix` if every joint is prefixed (for example `mixamorig:`).
4. Tune `sceneScale` so the mesh matches BVH units; use `scale` for hip translation scaling in retarget.
5. In the UI, pick **Character** or leave **Skeleton only**; use **Skeleton overlay** to draw the procedural cylinders on top.

Retargeting copies each source bone's world-space rotation onto the matching target
bone's bind pose (names lined up after `bonePrefix` strip, e.g. `mixamorig:Hips` →
`Hips`), plus the hip's world position. Only rotations are copied, so the character keeps
its own bind bone-lengths and stays correctly sized regardless of source/target units.
