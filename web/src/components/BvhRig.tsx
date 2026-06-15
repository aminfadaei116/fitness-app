import { useMemo, type ReactNode } from "react";
import { Quaternion, Matrix4, Vector3 } from "three";

type Props = {
  matrices: ReadonlyArray<Matrix4>;
  parentIndex: Int16Array;
  /** Multiplier on world translations so the procedural overlay renders in meters. */
  unitScale?: number;
  /** Flat-joint index to emphasise (debug correspondence); rendered larger and highlighted. */
  highlightIndex?: number | null;
};

export function BvhRig({ matrices, parentIndex, unitScale = 1, highlightIndex = null }: Props) {
  const boneColor = "#38bdf8";
  const jointColor = "#94f7c6";
  const highlightColor = "#facc15";

  const meshes = useMemo(() => {
    const up = new Vector3(0, 1, 0);
    const q = new Quaternion();
    const vA = new Vector3();
    const vB = new Vector3();
    const axis = new Vector3();
    const mid = new Vector3();
    const nodes: ReactNode[] = [];

    const radius = 0.026;
    for (let i = 0; i < matrices.length; i++) {
      const pIx = parentIndex[i]!;
      if (pIx < 0) continue;
      const wp = matrices[pIx];
      const wc = matrices[i];
      if (!wp || !wc) continue;

      const ea = wp.elements;
      const eb = wc.elements;
      vA.set(ea[12] * unitScale, ea[13] * unitScale, ea[14] * unitScale);
      vB.set(eb[12] * unitScale, eb[13] * unitScale, eb[14] * unitScale);
      axis.copy(vB).sub(vA);
      const len = axis.length();
      if (len < 1e-5) continue;
      axis.normalize();
      q.setFromUnitVectors(up, axis);
      mid.copy(vA).add(vB).multiplyScalar(0.5);

      nodes.push(
        <mesh
          key={`b-${String(i)}`}
          position={[mid.x, mid.y, mid.z]}
          quaternion={q.clone()}
          castShadow
          receiveShadow
        >
          <cylinderGeometry args={[radius, radius, len, 8]} />
          <meshStandardMaterial color={boneColor} roughness={0.55} metalness={0.15} />
        </mesh>,
      );
    }

    for (let i = 0; i < matrices.length; i++) {
      const em = matrices[i]!.elements;
      const x = em[12]! * unitScale;
      const y = em[13]! * unitScale;
      const z = em[14]! * unitScale;
      const hi = i === highlightIndex;
      nodes.push(
        <mesh key={`j-${String(i)}`} position={[x, y, z]} castShadow receiveShadow>
          <sphereGeometry args={[hi ? 0.08 : 0.045, 14, 14]} />
          <meshStandardMaterial
            color={hi ? highlightColor : jointColor}
            emissive={hi ? highlightColor : "#000000"}
            emissiveIntensity={hi ? 0.6 : 0}
            roughness={0.35}
            metalness={0.2}
          />
        </mesh>,
      );
    }

    return nodes;
  }, [matrices, parentIndex, unitScale, highlightIndex]);

  return <group>{meshes}</group>;
}
