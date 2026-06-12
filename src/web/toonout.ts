/** Browser BiRefNet-ToonOut matting — onnxruntime-web (webgpu, wasm
 * fallback). Bundlers resolve `import ... from "sprited/toonout"` to this
 * file via the exports "browser" condition; Node gets the onnxruntime-node
 * glue instead. onnxruntime-web is an optional peer — install it in apps
 * that use this. The model (~470MB) is fetched from HF and kept in the
 * Cache API, so it downloads once per origin. */
import type { RawImage } from "../core/image.js";
import { TOONOUT_SIZE, TOONOUT_MODEL_URL, toonoutPreprocess, toonoutApplyMask } from "../core/toonout.js";

let cached: Promise<{ ort: any; sess: any }> | undefined;

async function fetchModel(url: string): Promise<ArrayBuffer> {
  if (typeof caches !== "undefined") {
    const cache = await caches.open("sprited-models");
    let res = await cache.match(url);
    if (!res) {
      await cache.add(url);
      res = await cache.match(url);
    }
    if (res) return res.arrayBuffer();
  }
  return (await fetch(url)).arrayBuffer();
}

function session(modelUrl: string): Promise<{ ort: any; sess: any }> {
  return (cached ??= (async () => {
    const m: any = await import("onnxruntime-web");
    const ort = m.default ?? m;
    const model = new Uint8Array(await fetchModel(modelUrl));
    for (const ep of ["webgpu", "wasm"]) {
      try {
        return { ort, sess: await ort.InferenceSession.create(model, { executionProviders: [ep] }) };
      } catch (e) {
        if (ep === "wasm") throw e;
      }
    }
    throw new Error("unreachable");
  })());
}

/** Matte cells through BiRefNet-ToonOut, fully client-side. */
export async function toonoutMatting(cells: RawImage[], modelUrl: string = TOONOUT_MODEL_URL): Promise<RawImage[]> {
  const { ort, sess } = await session(modelUrl);
  const out: RawImage[] = [];
  for (const cell of cells) {
    const chw = toonoutPreprocess(cell);
    const res = await sess.run({ image: new ort.Tensor("float32", chw, [1, 3, TOONOUT_SIZE, TOONOUT_SIZE]) });
    out.push(toonoutApplyMask(cell, res.mask.data as Float32Array));
  }
  return out;
}
