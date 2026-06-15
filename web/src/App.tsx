import { useMemo, useState, useEffect, type ChangeEvent } from "react";
import { useLoader } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

import type { BvhDocument } from "./bvh/types.ts";
import { parseBvh } from "./bvh/parseBvh.ts";
import { poseFrame } from "./bvh/poseSkeleton.ts";
import type { CharacterManifestEntry } from "./characterManifest.ts";
import { PlaybackBar } from "./components/PlaybackBar.tsx";
import { ViewerCanvas } from "./components/ViewerCanvas.tsx";
import { DebugPage } from "./components/DebugPage.tsx";
import { characterModelKind } from "./modelFormat.ts";

function readManifest(payload: unknown): CharacterManifestEntry[] {
  if (!Array.isArray(payload)) return [];
  const out: CharacterManifestEntry[] = [];
  for (const row of payload) {
    if (!row || typeof row !== "object") continue;
    const r = row as Record<string, unknown>;
    const id = r.id,
      label = r.label,
      url = r.url;
    if (typeof id !== "string" || typeof label !== "string" || typeof url !== "string") continue;
    const idT = id.trim(),
      lbl = label.trim(),
      urlT = url.trim();
    if (!idT || !lbl || !urlT) continue;
    const entry: CharacterManifestEntry = {
      id: idT,
      label: lbl,
      url: urlT,
      ...(typeof r.bonePrefix === "string" && r.bonePrefix ? { bonePrefix: r.bonePrefix } : {}),
      ...(typeof r.hip === "string" && r.hip ? { hip: r.hip } : {}),
      ...(typeof r.scale === "number" && Number.isFinite(r.scale) ? { scale: r.scale } : {}),
      ...(typeof r.sceneScale === "number" && Number.isFinite(r.sceneScale)
        ? { sceneScale: r.sceneScale }
        : {}),
    };
    out.push(entry);
  }
  return out;
}

function ViewerApp() {
  const [doc, setDoc] = useState<BvhDocument | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [fileName, setFileName] = useState<string | null>(null);

  const [characters, setCharacters] = useState<CharacterManifestEntry[]>([]);
  const [characterId, setCharacterId] = useState<string>("");
  const [overlaySkeleton, setOverlaySkeleton] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/models/manifest.json")
      .then((r) => (r.ok ? r.json() : []))
      .then((payload) => {
        if (!cancelled) setCharacters(readManifest(payload));
      })
      .catch(() => {
        if (!cancelled) setCharacters([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setFrame(0);
  }, [doc]);

  useEffect(() => {
    if (!playing || !doc || doc.frameCount < 2) return;
    const ms = Math.max(4, (doc.frameTime * 1000) / Math.max(speed, 0.08));
    const id = window.setInterval(() => {
      setFrame((f) => (f + 1) % doc.frameCount);
    }, ms);
    return () => window.clearInterval(id);
  }, [playing, doc, speed]);

  const matrices = useMemo(() => {
    if (!doc || doc.motionData.length === 0) return null;
    const safe = Math.max(0, Math.min(frame, doc.frameCount - 1));
    const start = safe * doc.channelsPerFrame;
    const row = doc.motionData.subarray(start, start + doc.channelsPerFrame);
    return poseFrame(doc.flatJoints, doc.parentIndex, row);
  }, [doc, frame]);

  const selectedCharacter = useMemo(() => {
    if (!characterId.trim()) return null;
    return characters.find((c) => c.id === characterId) ?? null;
  }, [characterId, characters]);

  useEffect(() => {
    const url = selectedCharacter?.url;
    if (!url) return;
    const kind = characterModelKind(url);
    if (kind === "glb" || kind === "gltf") {
      void useGLTF.preload(url);
    } else if (kind === "fbx") {
      void useLoader.preload(FBXLoader, url);
    } else if (kind === "obj") {
      void useLoader.preload(OBJLoader, url);
    }
  }, [selectedCharacter?.url]);

  useEffect(() => {
    setOverlaySkeleton(false);
  }, [characterId]);

  const onPickFile = async (ev: ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    ev.target.value = "";
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseBvh(text);
      setDoc(parsed);
      setErrorText(null);
      setPlaying(true);
      setFileName(file.name);
    } catch (e) {
      setDoc(null);
      setErrorText(e instanceof Error ? e.message : String(e));
      setFileName(null);
    }
  };

  const fpsLabel = doc ? (1 / doc.frameTime).toFixed(2) : "---";

  return (
    <div className="app">
      <header className="toolbar">
        <label>
          BVH file
          <input type="file" accept=".bvh,.BVH,text/plain" onChange={onPickFile} />
        </label>
        {fileName ? (
          <span style={{ opacity: 0.85, fontSize: ".85rem" }}>{fileName}</span>
        ) : (
          <span style={{ opacity: 0.6, fontSize: ".85rem" }}>
            Pick a motion.bvh from keypoints/
          </span>
        )}
        <span style={{ fontSize: ".8rem", opacity: 0.75 }}>
          {doc ? `${doc.frameCount} frames · ~${fpsLabel} fps export` : "No file loaded"}
        </span>
        <a className="debugLink" href="#debug">
          Debug ↗
        </a>
        <label>
          Character
          <select
            className="toolbarSelect"
            value={characterId}
            disabled={characters.length === 0}
            onChange={(ev) => setCharacterId(ev.target.value)}
          >
            <option value="">Skeleton only</option>
            {characters.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        {characters.length === 0 ? (
          <span style={{ fontSize: ".75rem", opacity: 0.65 }}>
            Add models (.glb, .gltf, .fbx, .obj) under public/models + manifest.json — see README
          </span>
        ) : null}
        {selectedCharacter ? (
          <label className="inlineCheck">
            <input
              type="checkbox"
              checked={overlaySkeleton}
              onChange={(ev) => setOverlaySkeleton(ev.target.checked)}
            />
            Skeleton overlay
          </label>
        ) : null}
        {errorText ? <span className="errorText">{errorText}</span> : null}
        <PlaybackBar
          disabled={doc === null}
          frame={frame}
          frameCount={doc?.frameCount ?? 0}
          playing={playing}
          speed={speed}
          onTogglePlay={() => setPlaying((v) => !v)}
          onScrub={(f) => {
            setPlaying(false);
            setFrame(f);
          }}
          onSpeed={(s) => setSpeed(s)}
        />
      </header>
      <div className="viewport">
        <ViewerCanvas
          doc={doc}
          matrices={matrices}
          character={selectedCharacter}
          overlaySkeleton={overlaySkeleton}
        />
      </div>
    </div>
  );
}

/** Route between the BVH viewer and the pose-debug page via the URL hash (`#debug`). */
export default function App() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const onHash = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return hash === "#debug" ? <DebugPage /> : <ViewerApp />;
}
