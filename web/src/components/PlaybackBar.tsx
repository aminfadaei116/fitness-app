type PlaybackBarProps = {
  readonly frame: number;
  readonly frameCount: number;
  readonly playing: boolean;
  readonly speed: number;
  readonly disabled: boolean;
  readonly onTogglePlay: () => void;
  readonly onScrub: (frame: number) => void;
  readonly onSpeed: (speed: number) => void;
};

export function PlaybackBar({
  frame,
  frameCount,
  playing,
  speed,
  disabled,
  onTogglePlay,
  onScrub,
  onSpeed,
}: PlaybackBarProps) {
  const max = Math.max(frameCount - 1, 0);
  const labelPlaying = disabled ? "---" : `Frame ${frame + 1} / ${frameCount}`;

  return (
    <div className="playbackControls">
      <button type="button" disabled={disabled} onClick={onTogglePlay}>
        {playing ? "Pause" : "Play"}
      </button>
      <span style={{ minWidth: "9rem", fontSize: ".85rem" }}>{labelPlaying}</span>
      <label style={{ gap: ".25rem", display: "inline-flex", alignItems: "center" }}>
        <span style={{ fontSize: ".8rem", opacity: 0.85 }}>Scrub</span>
        <input
          type="range"
          min={0}
          max={max}
          value={Math.min(frame, max)}
          disabled={disabled || frameCount < 2}
          onChange={(ev) => onScrub(Number.parseInt(ev.target.value, 10))}
        />
      </label>
      <label style={{ gap: ".25rem", display: "inline-flex", alignItems: "center" }}>
        <span style={{ fontSize: ".8rem", opacity: 0.85 }}>Speed</span>
        <input
          type="range"
          min={0}
          max={250}
          value={speed * 100}
          disabled={disabled}
          onChange={(ev) => onSpeed(Number.parseInt(ev.target.value, 10) / 100)}
        />
      </label>
      <span style={{ fontSize: ".8rem", opacity: 0.75 }}>{speed.toFixed(2)}×</span>
    </div>
  );
}
