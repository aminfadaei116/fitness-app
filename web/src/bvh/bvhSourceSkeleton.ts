import {
  Bone,
  BoxGeometry,
  Float32BufferAttribute,
  Matrix4,
  MeshBasicMaterial,
  SkinnedMesh,
  Skeleton,
  Uint16BufferAttribute,
} from "three";

import type { BvhDocument, BvhJoint } from "./types.ts";

/** Total vertical extent (Y) of the BVH skeleton in its zero-rotation rest pose. */
export function bvhRestPoseHeight(doc: BvhDocument): number {
  let minY = Infinity;
  let maxY = -Infinity;
  const visit = (j: BvhJoint, parentY: number): void => {
    const y = parentY + j.offset[1];
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    for (const c of j.children) visit(c, y);
  };
  visit(doc.root, 0);
  const span = maxY - minY;
  return Number.isFinite(span) && span > 1e-6 ? span : 1;
}

const _invPw = new Matrix4();
const _local = new Matrix4();

/**
 * Invisible SkinnedMesh carrying a Bone tree that mirrors ``doc`` flat joint order.
 * Used as the *source* skeleton for ``SkeletonUtils.retarget`` onto a GLTF character.
 *
 * ``unitScale`` multiplies every offset so the rest pose lives in a chosen unit
 * (typically meters via ``TARGET_HEIGHT_M / bvhRestPoseHeight``). This keeps the
 * source and the retarget target in the same world-scale frame, avoiding
 * residual bone scaling after ``SkeletonUtils.retarget``.
 */
export function createBvhSourceSkinnedMesh(doc: BvhDocument, unitScale = 1): SkinnedMesh {
  const joints = doc.flatJoints;
  const parents = doc.parentIndex;
  const bones: Bone[] = joints.map(() => new Bone());

  for (let i = 0; i < joints.length; i++) {
    const j = joints[i]!;
    const b = bones[i]!;
    b.name = j.name;
    b.position.set(j.offset[0] * unitScale, j.offset[1] * unitScale, j.offset[2] * unitScale);

    const pid = parents[i];
    if (pid >= 0) {
      bones[pid]!.add(b);
    }
  }

  const rootIdx = Array.from(parents).findIndex((p) => p < 0);
  if (rootIdx < 0) {
    throw new Error("BVH joint tree has no root (-1 parent)");
  }

  const geometry = new BoxGeometry(0.01, 0.01, 0.01);
  const vCount = geometry.attributes.position.count;

  const idx = new Uint16Array(vCount * 4);
  const w = new Float32Array(vCount * 4);
  for (let v = 0; v < vCount; v++) {
    w[v * 4 + 0] = 1;
    w[v * 4 + 1] = 0;
    w[v * 4 + 2] = 0;
    w[v * 4 + 3] = 0;
  }
  geometry.setAttribute("skinIndex", new Uint16BufferAttribute(idx, 4));
  geometry.setAttribute("skinWeight", new Float32BufferAttribute(w, 4));

  const mat = new MeshBasicMaterial({
    depthWrite: false,
    transparent: true,
    opacity: 0,
    visible: false,
  });

  const mesh = new SkinnedMesh(geometry, mat);
  mesh.name = "BVH_SourceDriver";
  mesh.frustumCulled = false;
  mesh.add(bones[rootIdx]!);

  const skeleton = new Skeleton(bones);
  mesh.bind(skeleton);

  return mesh;
}

const _scaledWorld = /* @__PURE__ */ new Matrix4();

/**
 * Pose the source skeleton from per-joint world matrices, multiplying every
 * world-space translation by ``unitScale`` so the resulting bone hierarchy lives
 * in the same metric as the retarget target. Rotations are preserved.
 */
export function syncBvhSourcePose(
  skinned: SkinnedMesh,
  doc: BvhDocument,
  worldMats: readonly Matrix4[],
  unitScale = 1,
): void {
  const bones = skinned.skeleton.bones;
  const parents = doc.parentIndex;
  if (bones.length !== worldMats.length || bones.length !== doc.flatJoints.length) {
    throw new Error(
      `BVH bone / matrix mismatch: skeleton ${bones.length}, keys ${worldMats.length}`,
    );
  }

  for (let i = 0; i < bones.length; i++) {
    const b = bones[i]!;
    const wid = parents[i]!;
    const world = worldMats[i];
    if (!world) continue;

    _scaledWorld.copy(world);
    if (unitScale !== 1) {
      _scaledWorld.elements[12] *= unitScale;
      _scaledWorld.elements[13] *= unitScale;
      _scaledWorld.elements[14] *= unitScale;
    }

    if (wid < 0) {
      _scaledWorld.decompose(b.position, b.quaternion, b.scale);
    } else {
      const parentWorld = bones[wid]!.matrixWorld;
      _invPw.copy(parentWorld).invert();
      _local.multiplyMatrices(_invPw, _scaledWorld);
      _local.decompose(b.position, b.quaternion, b.scale);
    }
    b.updateMatrixWorld(false);
  }

  const rootIdx = Array.from(parents).findIndex((p) => p < 0);
  if (rootIdx >= 0) {
    bones[rootIdx]!.updateMatrixWorld(true);
  }
  skinned.skeleton.update();
}
