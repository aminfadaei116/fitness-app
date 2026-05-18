import { Suspense, useLayoutEffect, useMemo, useRef, type RefObject } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import type { Bone, Matrix4, Object3D, SkinnedMesh } from "three";
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
export function RigAndCharacterScene({ doc, matrices, character, overlaySkeleton }: RigSceneProps) {
  /**
   * BVH joint offsets are template ratios, not metric. Normalising the rest pose to
   * ``TARGET_HEIGHT_M`` lets us drive the source skeleton, render the procedural overlay,
   * and target a sized character all in the same world frame (meters) — which avoids the
   * residual scale that ``SkeletonUtils.retarget`` bakes into bones when source and target
   * live in mismatched parent-scale chains.
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
    const { hip, scale, prefix } = retargetOptsRef.current;

    if (tgt) {
      SkeletonUtils.retarget(tgt, src, {
        preserveBoneMatrix: true,
        hip,
        scale,
        getBoneName(bone: Bone): string {
          return stripBoneName(bone.name, prefix);
        },
      });
      tgt.updateMatrixWorld(true);
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
        <BvhRig matrices={matrices} parentIndex={doc.parentIndex} unitScale={bvhMeterScale} />
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
