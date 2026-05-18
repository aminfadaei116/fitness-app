/** Supported character mesh URLs (`public/models/…`). */

export type CharacterAssetKind = "glb" | "gltf" | "fbx" | "obj" | "unknown";

export function characterModelKind(url: string): CharacterAssetKind {
  const pathname = url.split(/[?#]/)[0] ?? "";
  const lower = pathname.toLowerCase();
  if (lower.endsWith(".glb")) return "glb";
  if (lower.endsWith(".gltf")) return "gltf";
  if (lower.endsWith(".fbx")) return "fbx";
  if (lower.endsWith(".obj")) return "obj";
  return "unknown";
}
