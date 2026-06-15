import { Suspense, useLayoutEffect, useMemo, useRef, type RefObject } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import type { Matrix4, Object3D, SkinnedMesh } from "three";
import { Box3, Mesh, Quaternion, Vector3 } from "three";
import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";

import {
  bvhRestPoseHeight,
  createBvhSourceSkinnedMesh,
  syncBvhSourcePose,
} from "../bvh/bvhSourceSkeleton.ts";
import type { BvhDocument } from "../bvh/types.ts";
import type { CharacterManifestEntry } from "../characterManifest.ts";
import { characterModelKind } from "../modelFormat.ts";
import { BvhRig } from "./BvhRig.tsx";

const _hipPos = /* @__PURE__ */ new Vector3();
const _hipQuat = /* @__PURE__ */ new Quaternion();
const _hipScale = /* @__PURE__ */ new Vector3();

// Scratch objects reused by the per-frame retarget (no per-frame allocation).
const _srcWorldQ = /* @__PURE__ */ new Quaternion();
const _desiredWorldQ = /* @__PURE__ */ new Quaternion();
const _parentWorldInvQ = /* @__PURE__ */ new Quaternion();
const _hipWorldPos = /* @__PURE__ */ new Vector3();

/**
 * One source→target bone correspondence plus the target bone's bind-pose world
 * rotation, captured once while the character is still in its bind pose.
 */
type RetargetPair = {
  readonly targetBone: Object3D;
  readonly sourceBone: Object3D;
  readonly targetBindWorldQ: Quaternion;
};

type RetargetBinding = {
  readonly target: SkinnedMesh;
  readonly source: SkinnedMesh;
  readonly prefix: string;
  readonly pairs: RetargetPair[];
  readonly hip: RetargetPair | null;
};

/**
 * Capture the bone correspondence and the target's bind-pose world rotations.
 * Must run while ``target`` is in its bind pose (i.e. before any retarget write).
 * The BVH source skeleton has an identity-rotation rest pose, so its bind world
 * rotation is the identity for every bone and need not be stored.
 */
function buildRetargetBinding(
  target: SkinnedMesh,
  source: SkinnedMesh,
  prefix: string,
  hipName: string,
): RetargetBinding {
  // Reset to bind pose before capturing, in case the mesh was already being posed
  // by a previous binding (e.g. the source BVH changed while the character stayed).
  target.skeleton.pose();
  target.updateMatrixWorld(true);
  const sourceByName = new Map<string, Object3D>();
  for (const b of source.skeleton.bones) sourceByName.set(b.name, b);

  const pairs: RetargetPair[] = [];
  let hip: RetargetPair | null = null;
  for (const targetBone of target.skeleton.bones) {
    const name = stripBoneName(targetBone.name, prefix);
    const sourceBone = sourceByName.get(name);
    if (!sourceBone) continue;
    const pair: RetargetPair = {
      targetBone,
      sourceBone,
      targetBindWorldQ: targetBone.getWorldQuaternion(new Quaternion()),
    };
    pairs.push(pair);
    if (name === hipName) hip = pair;
  }
  return { target, source, prefix, pairs, hip };
}

/**
 * Drive the target skeleton from the (already-posed) source skeleton by applying
 * each source bone's world-space rotation-from-bind onto the target bone's bind.
 * Source bind rotation is identity, so target world rotation = sourceWorld * targetBind.
 * Bones are processed root-first (skeleton.bones order) so each parent's world
 * matrix is current before its children read it. Only rotations are copied, which
 * keeps the target's own bind bone lengths — so the character stays correctly sized
 * regardless of the source/target unit mismatch. The hip additionally follows the
 * source hip's world position so the whole body translates.
 */
function applyRetarget(binding: RetargetBinding): void {
  for (const { targetBone, sourceBone, targetBindWorldQ } of binding.pairs) {
    sourceBone.getWorldQuaternion(_srcWorldQ);
    _desiredWorldQ.copy(_srcWorldQ).multiply(targetBindWorldQ);
    const parent = targetBone.parent;
    if (parent) {
      parent.getWorldQuaternion(_parentWorldInvQ).invert();
      targetBone.quaternion.copy(_parentWorldInvQ).multiply(_desiredWorldQ);
    } else {
      targetBone.quaternion.copy(_desiredWorldQ);
    }
    targetBone.updateWorldMatrix(false, false);
  }

  if (binding.hip) {
    const { targetBone, sourceBone } = binding.hip;
    sourceBone.getWorldPosition(_hipWorldPos);
    if (targetBone.parent) targetBone.parent.worldToLocal(_hipWorldPos);
    targetBone.position.copy(_hipWorldPos);
  }

  binding.target.updateMatrixWorld(true);
}

/** Target on-screen character height in meters (average adult). */
const TARGET_HEIGHT_M = 1.75;

function firstSkinned(root: Object3D): SkinnedMesh | undefined {
  let found: SkinnedMesh | undefined;
  root.traverse((o: Object3D) => {
    if (found) return;
    if ("isSkinnedMesh" in o && o.type === "SkinnedMesh") {
      found = o as SkinnedMesh;
    }
  });
  return found;
}

/**
 * Normalise a bone name so source/target match regardless of how the importer
 * handled the namespace (FBXLoader strips ``:`` via PropertyBinding.sanitizeNodeName,
 * GLTF usually keeps it). Tries the manifest prefix as written, then the same
 * prefix with a trailing colon removed, then drops anything before the last colon.
 */
function stripBoneName(name: string, prefix: string): string {
  let n = name;
  if (prefix) {
    if (n.startsWith(prefix)) {
      n = n.slice(prefix.length);
    } else {
      const sanitized = prefix.replace(/:/g, "");
      if (sanitized && sanitized !== prefix && n.startsWith(sanitized)) {
        n = n.slice(sanitized.length);
      }
    }
  }
  const c = n.lastIndexOf(":");
  if (c >= 0) {
    n = n.slice(c + 1);
  }
  return n;
}

export type RigSceneProps = {
  readonly doc: BvhDocument;
  readonly matrices: ReadonlyArray<Matrix4> | null;
  readonly character: CharacterManifestEntry | null;
  readonly overlaySkeleton: boolean;
  /** Flat-joint index to emphasise in the procedural overlay (debug correspondence). */
  readonly highlightIndex?: number | null;
};

function hipFlatIndex(
  joints: ReadonlyArray<{ readonly name: string }>,
  hipName: string,
): number {
  const n = hipName.trim();
  const i = joints.findIndex((j) => j.name === n);
  if (i >= 0) return i;
  const low = n.toLowerCase();
  return joints.findIndex((j) => j.name.toLowerCase() === low);
}

/** Invisible BVH driver + optional character (glTF / FBX skinned rig, OBJ hip-follow) + optional procedural overlay. */
export function RigAndCharacterScene({
  doc,
  matrices,
  character,
  overlaySkeleton,
  highlightIndex = null,
}: RigSceneProps) {
  /**
   * BVH joint offsets are template ratios, not metric. Normalising the rest pose to
   * ``TARGET_HEIGHT_M`` lets us drive the source skeleton, render the procedural overlay,
   * and target a sized character all in the same world frame (meters). The retarget copies
   * rotations only (plus the hip's world position), so the character keeps its own bind
   * bone lengths and metric scale regardless of the source units.
   */
  const bvhMeterScale = useMemo(() => TARGET_HEIGHT_M / bvhRestPoseHeight(doc), [doc]);
  const sourceMesh = useMemo(
    () => createBvhSourceSkinnedMesh(doc, bvhMeterScale),
    [doc, bvhMeterScale],
  );
  const sourceRef = useRef<SkinnedMesh | null>(null);
  const targetRef = useRef<SkinnedMesh | null>(null);
  /** When the asset has no skin (e.g. OBJ), pose this root from hip world matrix instead of retarget. */
  const rigidFollowRef = useRef<Object3D | null>(null);
  /** Cached bone correspondence + target bind rotations; rebuilt when source/target/prefix change. */
  const retargetBindingRef = useRef<RetargetBinding | null>(null);

  useLayoutEffect(() => {
    return () => {
      sourceMesh.geometry.dispose();
      if (!Array.isArray(sourceMesh.material)) {
        sourceMesh.material.dispose();
      }
    };
  }, [sourceMesh]);

  const retargetOptsRef = useRef({
    hip: character?.hip ?? "Hips",
    scale: character?.scale ?? 1,
    prefix: character?.bonePrefix ?? "",
  });

  useLayoutEffect(() => {
    retargetOptsRef.current = {
      hip: character?.hip ?? "Hips",
      scale: character?.scale ?? 1,
      prefix: character?.bonePrefix ?? "",
    };
  }, [character]);

  useFrame(() => {
    if (!matrices?.length) return;
    const src = sourceRef.current;
    if (!src) return;
    syncBvhSourcePose(src, doc, matrices, bvhMeterScale);

    if (!character) return;

    const tgt = targetRef.current;
    const rigid = rigidFollowRef.current;
    const { hip, prefix } = retargetOptsRef.current;

    if (tgt) {
      let binding = retargetBindingRef.current;
      if (
        !binding ||
        binding.target !== tgt ||
        binding.source !== src ||
        binding.prefix !== prefix
      ) {
        binding = buildRetargetBinding(tgt, src, prefix, hip);
        retargetBindingRef.current = binding;
      }
      applyRetarget(binding);
      return;
    }

    if (rigid) {
      const hi = hipFlatIndex(doc.flatJoints, hip);
      if (hi >= 0) {
        matrices[hi]!.decompose(_hipPos, _hipQuat, _hipScale);
        rigid.position.copy(_hipPos);
        rigid.quaternion.copy(_hipQuat);
        rigid.updateMatrixWorld(true);
      }
    }
  });

  const showProcedural = Boolean(matrices && (!character || overlaySkeleton) && doc.parentIndex);

  return (
    <>
      <primitive object={sourceMesh} ref={sourceRef} />
      {character ? (
        <Suspense fallback={null}>
          <CharacterAsset
            character={character}
            targetLocalHeight={TARGET_HEIGHT_M}
            targetRef={targetRef}
            rigidFollowRef={rigidFollowRef}
          />
        </Suspense>
      ) : null}
      {showProcedural && matrices ? (
        <BvhRig
          matrices={matrices}
          parentIndex={doc.parentIndex}
          unitScale={bvhMeterScale}
          highlightIndex={highlightIndex}
        />
      ) : null}
    </>
  );
}

type CharacterLoadProps = {
  character: CharacterManifestEntry;
  targetLocalHeight: number;
  targetRef: RefObject<SkinnedMesh | null>;
  rigidFollowRef: RefObject<Object3D | null>;
};

function CharacterAsset(props: CharacterLoadProps) {
  const kind = characterModelKind(props.character.url);
  if (kind === "unknown") {
    console.warn(`Unsupported character URL (use .glb, .gltf, .fbx, or .obj): ${props.character.url}`);
    return null;
  }
  if (kind === "glb" || kind === "gltf") return <CharacterGltf {...props} />;
  if (kind === "fbx") return <CharacterFbx {...props} />;
  return <CharacterObj {...props} />;
}

/**
 * Pick a uniform scale for the loaded asset so it matches ``targetHeight`` in
 * local units. Honors a manifest override; otherwise measures the rest-pose
 * vertical extent (skeleton bounds preferred, falling back to mesh bounds) and
 * normalises. A 200-cm Mixamo FBX and a 1.7-unit GLB both end up the same size.
 */
function computeCharacterScale(
  root: Object3D,
  character: CharacterManifestEntry,
  targetHeight: number,
): number {
  if (character.sceneScale !== undefined && character.sceneScale > 0) {
    return character.sceneScale;
  }
  root.updateMatrixWorld(true);
  const skin = firstSkinned(root);
  let minY = Infinity;
  let maxY = -Infinity;
  if (skin) {
    skin.skeleton.bones[0]?.updateMatrixWorld(true);
    for (const b of skin.skeleton.bones) {
      const y = b.matrixWorld.elements[13]!;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minY) || maxY - minY < 1e-6) {
    const box = new Box3().setFromObject(root);
    minY = box.min.y;
    maxY = box.max.y;
  }
  const span = maxY - minY;
  if (!Number.isFinite(span) || span < 1e-6) return 1;
  return targetHeight / span;
}

function attachCharacterRoot(
  root: Object3D,
  character: CharacterManifestEntry,
  targetHeightLocal: number,
  targetRef: RefObject<SkinnedMesh | null>,
  rigidFollowRef: RefObject<Object3D | null>,
) {
  root.scale.set(1, 1, 1);
  root.updateMatrixWorld(true);
  const scale = computeCharacterScale(root, character, targetHeightLocal);
  root.scale.setScalar(scale);
  root.traverse((o: Object3D) => {
    if (o instanceof Mesh) {
      o.frustumCulled = false;
    }
  });
  const skin = firstSkinned(root);
  if (skin) {
    skin.frustumCulled = false;
    targetRef.current = skin;
    rigidFollowRef.current = null;
  } else {
    targetRef.current = null;
    rigidFollowRef.current = root;
  }
}

function useBindCharacterScene(
  root: Object3D,
  character: CharacterManifestEntry,
  targetLocalHeight: number,
  targetRef: RefObject<SkinnedMesh | null>,
  rigidFollowRef: RefObject<Object3D | null>,
) {
  useLayoutEffect(() => {
    attachCharacterRoot(root, character, targetLocalHeight, targetRef, rigidFollowRef);
    return () => {
      targetRef.current = null;
      rigidFollowRef.current = null;
    };
  }, [character, root, targetLocalHeight, rigidFollowRef, targetRef]);
}

function SkinnedCloneScene({
  character,
  sourceScene,
  targetLocalHeight,
  targetRef,
  rigidFollowRef,
}: CharacterLoadProps & { sourceScene: Object3D }) {
  const root = useMemo(() => SkeletonUtils.clone(sourceScene), [character.url, sourceScene]);
  useBindCharacterScene(root, character, targetLocalHeight, targetRef, rigidFollowRef);
  return <primitive object={root} />;
}

function CharacterGltf(props: CharacterLoadProps) {
  const gltf = useGLTF(props.character.url);
  return <SkinnedCloneScene {...props} sourceScene={gltf.scene} />;
}

function CharacterFbx(props: CharacterLoadProps) {
  const fbx = useLoader(FBXLoader, props.character.url);
  return <SkinnedCloneScene {...props} sourceScene={fbx} />;
}

function CharacterObj(props: CharacterLoadProps) {
  const objRoot = useLoader(OBJLoader, props.character.url);
  return <SkinnedCloneScene {...props} sourceScene={objRoot} />;
}
