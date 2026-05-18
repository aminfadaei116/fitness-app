import { useMemo, useState, useEffect, type ChangeEvent } from "react";

import type { BvhDocument } from "./bvh/types.ts";
import { parseBvh } from "./bvh/parseBvh.ts";
import { poseFrame } from "./bvh/poseSkeleton.ts";
import { PlaybackBar } from "./components/PlaybackBar.tsx";
import { ViewerCanvas } from "./components/ViewerCanvas.tsx";

export default function App() {
  const [doc, setDoc] = useState<BvhDocument | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [fileName, setFileName] = useState<string | null>(null);

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
          <span style={{ opacity: 0.6, fontSize: ".85rem" }}>Pick a motion.bvh from keypoints/</span>
        )}
        <span style={{ fontSize: ".8rem", opacity: 0.75 }}>
          {doc ? `${doc.frameCount} frames · ~${fpsLabel} fps export` : "No file loaded"}
        </span>
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
        <ViewerCanvas matrices={matrices} parentIndex={doc?.parentIndex ?? null} />
      </div>
    </div>
  );
}
