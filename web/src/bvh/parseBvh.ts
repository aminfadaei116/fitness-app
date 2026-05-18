import type { BvhChannel, BvhDocument, BvhJoint, Vec3 } from "./types.ts";

function stripComment(line: string): string {
  const hash = line.indexOf("#");
  return (hash >= 0 ? line.slice(0, hash) : line).trimEnd();
}

function normalizedLines(raw: string): string[] {
  return raw.split(/\r?\n/).map((l) => stripComment(l).trim()).filter(Boolean);
}

class Lines {
  readonly arr: readonly string[];
  constructor(arr: readonly string[]) {
    this.arr = arr;
  }
  i = 0;
  peek(): string | undefined {
    return this.arr[this.i];
  }
  take(): string {
    const s = this.arr[this.i++];
    if (s === undefined) {
      throw new Error("Unexpected EOF in BVH file");
    }
    return s;
  }
}

function expectBrace(lp: Lines): void {
  const b = lp.take();
  if (b !== "{") {
    throw new Error(`Expected '{', got: ${b}`);
  }
}

function parseVec3FromOffsetLine(line: string): Vec3 {
  const w = line.split(/\s+/);
  if (w.length < 4 || !/^OFFSET$/i.test(w[0])) {
    throw new Error(`Invalid OFFSET line: ${line}`);
  }
  return [
    Number.parseFloat(w[1]),
    Number.parseFloat(w[2]),
    Number.parseFloat(w[3]),
  ];
}

function parseChannelsLine(line: string): BvhChannel[] {
  const w = line.split(/\s+/);
  if (!/^CHANNELS$/i.test(w[0])) {
    throw new Error(`Invalid CHANNELS line: ${line}`);
  }
  const count = Number.parseInt(w[1] ?? "", 10);
  if (!Number.isFinite(count) || count < 1) {
    throw new Error(`Bad CHANNEL count: ${line}`);
  }
  const names = w.slice(2);
  if (names.length !== count) {
    throw new Error(`CHANNELS ${count} vs ${names.length} names`);
  }
  return names.map((n) => n as BvhChannel);
}

function skipEndSite(lp: Lines): void {
  const h = lp.take();
  if (!/^End\s+Site$/i.test(h)) {
    throw new Error(`Expected End Site, got '${h}'`);
  }
  expectBrace(lp);
  for (;;) {
    const inner = lp.take();
    if (inner === "}") break;
    if (/^OFFSET\s+/i.test(inner)) continue;
    throw new Error(`Unexpected line inside End Site: ${inner}`);
  }
}

function parseJoint(lp: Lines): BvhJoint {
  const hdr = lp.take();
  const m = /^(ROOT|JOINT)\s+(\S+)$/.exec(hdr);
  if (!m) {
    throw new Error(`Expected ROOT/JOINT declaration, got '${hdr}'`);
  }
  expectBrace(lp);

  let offset: Vec3 = [0, 0, 0];
  let channels: BvhChannel[] = [];
  const children: BvhJoint[] = [];

  for (;;) {
    const p = lp.peek();
    if (!p) {
      throw new Error("Truncated joint block before '}'");
    }
    if (p === "}") {
      lp.take();
      break;
    }
    if (/^OFFSET\s+/i.test(p)) {
      offset = parseVec3FromOffsetLine(lp.take());
      continue;
    }
    if (/^CHANNELS\s+/i.test(p)) {
      channels = [...parseChannelsLine(lp.take())];
      continue;
    }
    if (/^End\s+Site$/i.test(p)) {
      skipEndSite(lp);
      continue;
    }
    if (/^(ROOT|JOINT)\s+/.test(p)) {
      children.push(parseJoint(lp));
      continue;
    }
    throw new Error(`Unsupported line in joint '${m[2]}': ${p}`);
  }

  if (channels.length === 0) {
    throw new Error(`Joint '${m[2]}' missing CHANNELS`);
  }

  return {
    name: m[2],
    isRoot: m[1] === "ROOT",
    offset,
    channels,
    children,
  };
}

export function motionChannelCount(j: BvhJoint): number {
  let sum = j.channels.length;
  for (const ch of j.children) {
    sum += motionChannelCount(ch);
  }
  return sum;
}

export function preorderWithParents(root: BvhJoint): {
  flatJoints: BvhJoint[];
  parentIndex: Int16Array;
} {
  const flatJoints: BvhJoint[] = [];
  const parents: number[] = [];

  const walk = (node: BvhJoint, parentIdx: number) => {
    const idx = flatJoints.length;
    flatJoints.push(node);
    parents.push(parentIdx);
    for (const c of node.children) walk(c, idx);
  };

  walk(root, -1);
  return { flatJoints, parentIndex: Int16Array.from(parents) };
}

function parseHierarchyLines(lines: readonly string[]): BvhJoint {
  const lp = new Lines(lines);
  const h = lp.take();
  if (h !== "HIERARCHY") {
    throw new Error(`Expected HIERARCHY, got '${h}'`);
  }
  const root = parseJoint(lp);
  return root;
}

function parseMotionLines(lines: readonly string[]): {
  frameCount: number;
  frameTime: number;
  rows: Float32Array[];
} {
  let frameCount = -1;
  let frameTime = -1;
  const numericRows: Float32Array[] = [];

  let i = 0;
  while (i < lines.length) {
    const ln = lines[i]!;
    if (/^Frames:\s+/i.test(ln)) {
      frameCount = Number.parseInt(ln.split(":")[1]!.trim(), 10);
      i += 1;
      continue;
    }
    if (/^Frame\s+Time:/i.test(ln)) {
      frameTime = Number.parseFloat(ln.split(":")[1]!.trim());
      i += 1;
      continue;
    }
    const floats = ln
      .split(/\s+/)
      .filter(Boolean)
      .map(Number.parseFloat);
    if (floats.length > 0) {
      if (floats.some((v) => !Number.isFinite(v))) {
        throw new Error(`Invalid numeric row: ${ln}`);
      }
      numericRows.push(Float32Array.from(floats));
    }
    i += 1;
  }

  if (numericRows.length === 0) {
    throw new Error("MOTION contains no numeric frame rows");
  }
  if (frameCount <= 0) {
    frameCount = numericRows.length;
  }
  if (numericRows.length !== frameCount) {
    throw new Error(
      `Frame count mismatch: Frames ${frameCount} vs ${numericRows.length} numeric rows`,
    );
  }
  if (!Number.isFinite(frameTime) || frameTime <= 0) {
    frameTime = 1 / 30;
  }

  return { frameCount, frameTime, rows: numericRows };
}

export function parseBvh(raw: string): BvhDocument {
  const text = raw.trimStart();
  const motionMatch = /\bMOTION\b/i.exec(text);
  if (!motionMatch) {
    throw new Error("BVH missing MOTION section");
  }
  const hierPart = text.slice(0, motionMatch.index).trimEnd();
  const motPart = text.slice(motionMatch.index + motionMatch[0].length).trim();

  const hierLines = normalizedLines(hierPart);
  const motionLinesRaw = normalizedLines(motPart);
  const motionLines =
    motionLinesRaw[0] === "MOTION" ? motionLinesRaw.slice(1) : motionLinesRaw;

  const root = parseHierarchyLines(hierLines);
  const { frameCount, frameTime, rows } = parseMotionLines(motionLines);
  const { flatJoints, parentIndex } = preorderWithParents(root);
  const channelsPerFrame = motionChannelCount(root);

  const motionData = new Float32Array(frameCount * channelsPerFrame);
  for (let f = 0; f < frameCount; f++) {
    const row = rows[f];
    if (row?.length !== channelsPerFrame) {
      throw new Error(
        `Frame ${f} has ${row?.length ?? 0} floats, expected ${channelsPerFrame}`,
      );
    }
    motionData.set(row, f * channelsPerFrame);
  }

  return {
    root,
    frameCount,
    frameTime,
    channelsPerFrame,
    flatJoints,
    parentIndex,
    motionData,
  };
}
