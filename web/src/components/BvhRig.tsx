import { useMemo, type ReactNode } from "react";
import { Quaternion, Matrix4, Vector3 } from "three";

type Props = {
  matrices: ReadonlyArray<Matrix4>;
  parentIndex: Int16Array;
};

export function BvhRig({ matrices, parentIndex }: Props) {
  const boneColor = "#38bdf8";
  const jointColor = "#94f7c6";

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
      vA.set(ea[12], ea[13], ea[14]);
      vB.set(eb[12], eb[13], eb[14]);
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
      const x = em[12]!;
      const y = em[13]!;
      const z = em[14]!;
      nodes.push(
        <mesh key={`j-${String(i)}`} position={[x, y, z]} castShadow receiveShadow>
          <sphereGeometry args={[0.045, 10, 10]} />
          <meshStandardMaterial color={jointColor} roughness={0.35} metalness={0.2} />
        </mesh>,
      );
    }

    return nodes;
  }, [matrices, parentIndex]);

  return <group>{meshes}</group>;
}
