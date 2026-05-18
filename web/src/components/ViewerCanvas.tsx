import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import type { Matrix4 } from "three";

import { BvhRig } from "./BvhRig.tsx";

type Props = {
  matrices: ReadonlyArray<Matrix4> | null;
  parentIndex: Int16Array | null;
};

export function ViewerCanvas({ matrices, parentIndex }: Props) {
  const rig =
    matrices && parentIndex && matrices.length > 0 ? (
      <BvhRig matrices={matrices} parentIndex={parentIndex} />
    ) : null;

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
      {rig}
    </Canvas>
  );
}
