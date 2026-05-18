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

BVH parsing is covered by a synthetic two-bone snippet that follows the exporter’s channel naming.
