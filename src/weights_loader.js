/**
 * Build a { name -> { shape, data: Float32Array } } map from a single binary
 * blob and its manifest. Views share the underlying ArrayBuffer (no copy).
 *
 * manifest.tensors[*].offset and .length are in float-elements (not bytes).
 */
export function loadWeightsFromBuffers(arrayBuffer, manifest) {
  const out = {};
  for (const t of manifest.tensors) {
    out[t.name] = {
      shape: t.shape,
      data: new Float32Array(arrayBuffer, t.offset * 4, t.length),
    };
  }
  return out;
}

/**
 * Browser helper: fetch bin + manifest in parallel and hand back loaded weights.
 */
export async function loadWeights(binUrl, manifestUrl) {
  const [binResp, manifestResp] = await Promise.all([fetch(binUrl), fetch(manifestUrl)]);
  if (!binResp.ok) throw new Error(`weights bin: ${binResp.status}`);
  if (!manifestResp.ok) throw new Error(`weights manifest: ${manifestResp.status}`);
  const [arrayBuffer, manifest] = await Promise.all([
    binResp.arrayBuffer(),
    manifestResp.json(),
  ]);
  return loadWeightsFromBuffers(arrayBuffer, manifest);
}
