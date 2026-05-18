import { Matrix4 } from "three";

import type { BvhJoint } from "./types.ts";

const _rz = /* @__PURE__ */ new Matrix4();
const _ry = /* @__PURE__ */ new Matrix4();
const _rx = /* @__PURE__ */ new Matrix4();

/** Rz(degZ) Ry(degY) Rx(degX) — pairing with exporter Z-Y-X stacking. */
function eulerZYXdeg(degZ: number, degY: number, degX: number): Matrix4 {
  const r = /* @__PURE__ */ new Matrix4();
  _rz.makeRotationZ((degZ * Math.PI) / 180);
  _ry.makeRotationY((degY * Math.PI) / 180);
  _rx.makeRotationX((degX * Math.PI) / 180);
  r.multiplyMatrices(_rz, _ry);
  r.multiply(_rx);
  return r;
}

const _offs = /* @__PURE__ */ new Matrix4();
const _tw = /* @__PURE__ */ new Matrix4();
const _out = /* @__PURE__ */ new Matrix4();

function composeLocal(joint: BvhJoint, values: Float32Array): Matrix4 {
  let px = 0;
  let py = 0;
  let pz = 0;
  let rz = 0;
  let ry = 0;
  let rx = 0;

  for (let i = 0; i < joint.channels.length; i++) {
    const nm = joint.channels[i]!;
    const v = values[i]!;
    switch (nm) {
      case "Xposition":
        px = v;
        break;
      case "Yposition":
        py = v;
        break;
      case "Zposition":
        pz = v;
        break;
      case "Zrotation":
        rz = v;
        break;
      case "Yrotation":
        ry = v;
        break;
      case "Xrotation":
        rx = v;
        break;
      default:
        throw new Error(`Unknown channel: ${String(nm)}`);
    }
  }

  _offs.makeTranslation(joint.offset[0], joint.offset[1], joint.offset[2]);
  const rot = eulerZYXdeg(rz, ry, rx);
  _out.multiplyMatrices(rot, _offs);

  if (!joint.isRoot) {
    return _out.clone();
  }

  _tw.makeTranslation(px, py, pz);
  const rootLocal = /* @__PURE__ */ new Matrix4();
  rootLocal.multiplyMatrices(_tw, rot);
  rootLocal.multiply(_offs);
  return rootLocal;
}

export function poseFrame(
  flatJoints: ReadonlyArray<BvhJoint>,
  parents: Int16Array | Int32Array | ArrayLike<number>,
  motionRow: Float32Array,
): Matrix4[] {
  const wm: Matrix4[] = [];
  let cursor = 0;
  for (let i = 0; i < flatJoints.length; i++) {
    const j = flatJoints[i]!;
    const slice = motionRow.subarray(cursor, cursor + j.channels.length);
    cursor += j.channels.length;
    const local = composeLocal(j, slice);
    wm[i] = new Matrix4();

    const p = parents[i]!;
    if (p < 0) {
      wm[i].copy(local);
    } else {
      wm[i].multiplyMatrices(wm[p]!, local);
    }
  }
  if (cursor !== motionRow.length) {
    throw new Error(`Channel cursor ${cursor} vs row ${motionRow.length}`);
  }
  return wm;
}
