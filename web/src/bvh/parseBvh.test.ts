import { describe, expect, it } from "vitest";

import { motionChannelCount, parseBvh } from "./parseBvh.ts";
import { poseFrame } from "./poseSkeleton.ts";

const SAMPLE = `
HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Chest
  {
    OFFSET 0.0 1.0 0.0
    CHANNELS 3 Zrotation Yrotation Xrotation
  }
}
MOTION
Frames: 1
Frame Time: 0.04
0.0 0.0 0.0 0.0 0.0 0.0 12.5 21.75 44.125
`;

describe("parseBvh + poseSkeleton", () => {
  it("parses frames and FK contains no NaN", () => {
    const doc = parseBvh(SAMPLE.trim());
    expect(doc.frameCount).toBe(1);
    expect(motionChannelCount(doc.root)).toBe(doc.channelsPerFrame);
    expect(doc.channelsPerFrame).toBe(9);

    const row = doc.motionData.subarray(0, doc.channelsPerFrame);
    const mats = poseFrame(doc.flatJoints, doc.parentIndex, row);
    expect(mats).toHaveLength(2);

    let ok = true;
    for (const m of mats) {
      const e = m.elements;
      ok &&= Array.from(e).every(Number.isFinite);
    }
    expect(ok).toBe(true);
    expect(Number.isFinite(mats[1]!.elements[12])).toBe(true);
    expect(Number.isFinite(mats[1]!.elements[13])).toBe(true);
    expect(Number.isFinite(mats[1]!.elements[14])).toBe(true);
  });
});
