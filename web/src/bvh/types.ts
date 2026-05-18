export type Vec3 = readonly [number, number, number];

/** Channel names matching Biovision BVH (this repo emits root + body joints). */
export type BvhChannel =
  | "Xposition"
  | "Yposition"
  | "Zposition"
  | "Xrotation"
  | "Yrotation"
  | "Zrotation";

export interface BvhJoint {
  readonly name: string;
  readonly isRoot: boolean;
  readonly offset: Vec3;
  readonly channels: readonly BvhChannel[];
  readonly children: readonly BvhJoint[];
}

export interface BvhDocument {
  readonly root: BvhJoint;
  readonly frameCount: number;
  readonly frameTime: number;
  /** Seconds per MOTION sample. */
  readonly channelsPerFrame: number;
  /** Pre-order DFS of joints that carry CHANNELS rows (excluding End Sites). */
  flatJoints: ReadonlyArray<BvhJoint>;
  /** Parallel to ``flatJoints``; `-1` for root. Parent always precedes children. */
  parentIndex: Int16Array;
  /** Packed ``frameCount * channelsPerFrame`` floats row-major. */
  readonly motionData: Float32Array;
}
