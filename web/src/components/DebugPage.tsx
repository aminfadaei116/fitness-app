import { useEffect, useMemo, useState } from "react";

import type { BvhDocument } from "../bvh/types.ts";
import { parseBvh } from "../bvh/parseBvh.ts";
import { poseFrame } from "../bvh/poseSkeleton.ts";
import type { CharacterManifestEntry } from "../characterManifest.ts";
import { ViewerCanvas } from "./ViewerCanvas.tsx";

/** Bundle written by ``scripts/build_pose_debug.py`` into ``web/public/debug/``. */
type DebugBundle = {
  image: string;
  width: number;
  height: number;
  points: [number, number, number][]; // normalized x, y, visibility (×33)
  connections: [number, number][];
  names: Record<string, string>;
  landmarkToBvh: Record<string, string>;
};

function colorFor(bundle: DebugBundle, i: number): string {
  const nm = bundle.names[String(i)] ?? "";
  if (nm.startsWith("L-")) return "#60a5fa";
  if (nm.startsWith("R-")) return "#fb923c";
  if (bundle.landmarkToBvh[String(i)]) return "#34d399";
  return "#64748b";
}

export function DebugPage() {
  const [bundle, setBundle] = useState<DebugBundle | null>(null);
  const [doc, setDoc] = useState<BvhDocument | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const [characters, setCharacters] = useState<CharacterManifestEntry[]>([]);
  const [characterId, setCharacterId] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const b = (await (await fetch("/debug/landmarks.json")).json()) as DebugBundle;
        const bvhText = await (await fetch("/debug/pose.bvh")).text();
        if (cancelled) return;
        setBundle(b);
        setDoc(parseBvh(bvhText));
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch("/models/manifest.json")
      .then((r) => (r.ok ? r.json() : []))
      .then((rows: unknown) => {
        if (cancelled || !Array.isArray(rows)) return;
        const out = rows.filter(
          (r): r is CharacterManifestEntry =>
            !!r && typeof r === "object" && typeof (r as CharacterManifestEntry).url === "string",
        );
        setCharacters(out);
        if (out[0]) setCharacterId(out[0].id);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const matrices = useMemo(() => {
    if (!doc || doc.motionData.length === 0) return null;
    const row = doc.motionData.subarray(0, doc.channelsPerFrame);
    return poseFrame(doc.flatJoints, doc.parentIndex, row);
  }, [doc]);

  const jointIndex = useMemo(() => {
    const m = new Map<string, number>();
    doc?.flatJoints.forEach((j, i) => m.set(j.name, i));
    return m;
  }, [doc]);

  const selectedCharacter = useMemo(
    () => characters.find((c) => c.id === characterId) ?? null,
    [characters, characterId],
  );

  const mappedJoint = hovered != null && bundle ? bundle.landmarkToBvh[String(hovered)] : undefined;
  const highlightIndex =
    mappedJoint != null ? (jointIndex.get(mappedJoint) ?? null) : null;

  return (
    <div className="app">
      <header className="toolbar">
        <strong>Pose debug — image ↔ 3D correspondence</strong>
        <a className="debugLink" href="#" onClick={() => (window.location.hash = "")}>
          ← Back to viewer
        </a>
        <label>
          Character
          <select
            className="toolbarSelect"
            value={characterId}
            onChange={(e) => setCharacterId(e.target.value)}
          >
            <option value="">Skeleton only</option>
            {characters.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <span style={{ fontSize: ".82rem", opacity: 0.8 }}>
          {hovered != null
            ? `${bundle?.names[String(hovered)] ?? `landmark ${hovered}`} (idx ${hovered})` +
              (mappedJoint ? `  →  BVH ${mappedJoint}` : "  →  (no joint)")
            : "Hover a landmark to highlight its 3D joint"}
        </span>
        {error ? <span className="errorText">{error}</span> : null}
        <span style={{ fontSize: ".75rem", opacity: 0.6 }}>
          Regenerate: python scripts/build_pose_debug.py &lt;image&gt;
        </span>
      </header>
      <div className="viewport">
        <div className="compareSplit">
          <div className="debugImagePanel">
            {bundle ? (
              <div
                className="debugImageWrap"
                style={{ aspectRatio: `${bundle.width} / ${bundle.height}` }}
              >
                <img src={bundle.image} alt="pose source" />
                <svg viewBox="0 0 1 1" preserveAspectRatio="none" className="debugSvg">
                  {bundle.connections.map(([a, b], k) => {
                    const pa = bundle.points[a];
                    const pb = bundle.points[b];
                    if (!pa || !pb || pa[2] < 0.3 || pb[2] < 0.3) return null;
                    return (
                      <line
                        key={k}
                        x1={pa[0]}
                        y1={pa[1]}
                        x2={pb[0]}
                        y2={pb[1]}
                        stroke="#e2e8f0aa"
                        strokeWidth={0.004}
                      />
                    );
                  })}
                  {bundle.points.map((p, i) => {
                    const isHover = i === hovered;
                    const mapped = Boolean(bundle.landmarkToBvh[String(i)]);
                    return (
                      <circle
                        key={i}
                        cx={p[0]}
                        cy={p[1]}
                        r={isHover ? 0.022 : mapped ? 0.014 : 0.008}
                        fill={isHover ? "#facc15" : colorFor(bundle, i)}
                        stroke={isHover ? "#000" : "none"}
                        strokeWidth={0.003}
                        style={{ cursor: "pointer" }}
                        onMouseEnter={() => setHovered(i)}
                        onMouseLeave={() => setHovered((h) => (h === i ? null : h))}
                      />
                    );
                  })}
                </svg>
              </div>
            ) : (
              <span style={{ opacity: 0.6 }}>Loading /debug/landmarks.json …</span>
            )}
          </div>
          <div className="canvasPanel">
            <ViewerCanvas
              doc={doc}
              matrices={matrices}
              character={selectedCharacter}
              overlaySkeleton={true}
              highlightIndex={highlightIndex}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
