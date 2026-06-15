import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import type { Matrix4 } from "three";

import type { CharacterManifestEntry } from "../characterManifest.ts";
import type { BvhDocument } from "../bvh/types.ts";
import { RigAndCharacterScene } from "./RigAndCharacter.tsx";

type Props = {
  readonly doc: BvhDocument | null;
  readonly matrices: ReadonlyArray<Matrix4> | null;
  readonly character: CharacterManifestEntry | null;
  readonly overlaySkeleton: boolean;
  /** Flat-joint index to emphasise in the procedural overlay (debug correspondence). */
  readonly highlightIndex?: number | null;
};

export function ViewerCanvas({
  doc,
  matrices,
  character,
  overlaySkeleton,
  highlightIndex = null,
}: Props) {
  const showRig =
    doc !== null && matrices !== null && matrices.length > 0 && doc.flatJoints.length > 0;

  return (
    <Canvas shadows camera={{ position: [2.4, 1.85, 3.6], fov: 45, near: 0.08, far: 200 }}>
      <color attach="background" args={["#0a0a14"]} />
      <ambientLight intensity={0.42} />
      <directionalLight
        castShadow
        intensity={1.05}
        position={[4.5, 9, 6]}
        shadow-mapSize={[1024, 1024]}
      />
      <Grid
        infiniteGrid
        fadeDistance={45}
        sectionSize={1}
        cellSize={0.25}
        sectionColor={"#475569"}
        cellColor={"#1e293b"}
      />
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} target={[0, 1, 0]} />
      {showRig && doc && matrices ? (
        <RigAndCharacterScene
          doc={doc}
          matrices={matrices}
          character={character}
          overlaySkeleton={overlaySkeleton}
          highlightIndex={highlightIndex}
        />
      ) : null}
    </Canvas>
  );
}
