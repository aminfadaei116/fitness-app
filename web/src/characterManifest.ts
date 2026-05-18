/** One character entry shipped with the app (`public/models/manifest.json`). */

export type CharacterManifestEntry = {
  readonly id: string;
  readonly label: string;
  /** URL relative to site root, e.g. `/models/player.glb`, `.fbx`, or `.obj`. */
  readonly url: string;
  /** Strip from mesh bone names before matching BVH bones, e.g. `mixamorig:` */
  readonly bonePrefix?: string;
  /** Hip bone semantic name after prefix strip (`SkeletonUtils.retarget`, or OBJ hip-follow). Usually `Hips`. */
  readonly hip?: string;
  /** Extra scale on hip translation (SkeletonUtils.retarget). */
  readonly scale?: number;
  /** Uniform scale on the cloned scene root (fits model to BVH units). */
  readonly sceneScale?: number;
};
