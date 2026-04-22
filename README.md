# silvero-vad-js

Pure-JavaScript SileroVAD v5 inference engine. No WASM, no ONNX Runtime — runs on all browsers with `AudioWorklet` support. Matches the official `silero-vad` Python wrapper's output frame-by-frame.

## Why this exists

Every JS SileroVAD library depends on `onnxruntime-web` (WASM). That works on most browsers, but ONNX Runtime WASM has been broken on iOS Safari since iOS 16.4+, silently failing on iPhones. This engine reimplements the model's forward pass in vanilla JS — no WASM dependency at all — so it works everywhere: Chrome, Firefox, Safari (desktop and iOS), and any browser that supports AudioWorklet.

## Install

```bash
npm install @jorastechnologies/silvero-vad-js
```

## Setup

Copy the runtime assets to your app's public/static directory:

```bash
cp node_modules/@jorastechnologies/silvero-vad-js/weights/silero_vad_v5.bin      public/vad/
cp node_modules/@jorastechnologies/silvero-vad-js/weights/silero_vad_v5.manifest.json public/vad/
cp node_modules/@jorastechnologies/silvero-vad-js/src/vad_processor.js            public/vad/
```

The weight files (~1.2 MB) and AudioWorklet script must be served as static files — they are fetched by the browser at runtime and cannot be inlined into a JS bundle.

## Usage

```javascript
import { VADRecorder } from '@jorastechnologies/silvero-vad-js';

const rec = new VADRecorder({
  weightsBinUrl:      '/vad/silero_vad_v5.bin',
  weightsManifestUrl: '/vad/silero_vad_v5.manifest.json',
  workletUrl:         '/vad/vad_processor.js',
  threshold: 0.5,
  silenceMs: 480,
});
rec.addEventListener('speech-start', () => console.log('speaking'));
rec.addEventListener('speech-end', (e) => sendToSTT(e.detail.audio));
await rec.start();
```

`speech-end` carries the accumulated audio as a `Float32Array` at 16kHz (`event.detail.audio` + `event.detail.sampleRate`).

## Deploying inside an existing Flask + nginx app

Drop the runtime files into Flask's static folder:

```
your-flask-app/static/vad/
├── silero_vad_v5.bin              # ~1.2 MB (16kHz-only weights)
├── silero_vad_v5.manifest.json    # ~1 KB
├── vad_processor.js               # AudioWorklet — MUST be same-origin
├── silero_vad.js
├── weights_loader.js
├── ops.js
├── vad_recorder.js
└── index.js
```

Deploy script:

```bash
cp node_modules/@jorastechnologies/silvero-vad-js/src/*.js                        your-flask-app/static/vad/
cp node_modules/@jorastechnologies/silvero-vad-js/weights/silero_vad_v5.bin       your-flask-app/static/vad/
cp node_modules/@jorastechnologies/silvero-vad-js/weights/silero_vad_v5.manifest.json your-flask-app/static/vad/
```

Template snippet:

```html
<script type="module">
  import { VADRecorder } from "{{ url_for('static', filename='vad/index.js') }}";
  const rec = new VADRecorder({
    weightsBinUrl:      "{{ url_for('static', filename='vad/silero_vad_v5.bin') }}",
    weightsManifestUrl: "{{ url_for('static', filename='vad/silero_vad_v5.manifest.json') }}",
    workletUrl:         "{{ url_for('static', filename='vad/vad_processor.js') }}",
  });
  rec.addEventListener('speech-end', (e) => sendToSTT(e.detail.audio));
  await rec.start();
</script>
```

### nginx checklist

1. **`.js` MIME type** — `curl -I https://yourdomain/static/vad/vad_processor.js` must return `Content-Type: application/javascript` (or `text/javascript`). Default `mime.types` covers this.

2. **Long cache on weights:**

   ```nginx
   location ~* /static/vad/.*\.(bin|json)$ {
       expires 1y;
       add_header Cache-Control "public, immutable";
   }
   ```

   On model upgrades, version via query string (`silero_vad_v5.bin?v=2`) — don't rename the file.

3. **Same-origin for the worklet** — AudioWorklet refuses cross-origin module loads even with CORS. Serving from `/static/vad/` (same origin as the page) is the simplest fix.

### iOS Safari requirements

- **HTTPS is mandatory** — iOS blocks `getUserMedia` on plain HTTP from non-localhost hosts. Use Caddy + Let's Encrypt, nginx + certbot, or Cloudflare in front of nginx.
- **iOS Safari 14.5+** is required for AudioWorklet.
- The `AudioContext` sample rate will be 48kHz on iOS; Chrome on macOS returns 16kHz when requested. Both work — the engine doesn't care about the source rate because the `AudioContext` is explicitly constructed at 16kHz and the browser resamples.

## Architecture

See `docs/ARCHITECTURE.md` for the per-layer breakdown of the 16kHz forward pass, including the non-obvious STFT input assembly (prepend 64 context samples, right-reflect by 64, total 640 samples into the STFT conv).

The engine ignores the 8kHz path in the ONNX graph and implements only the 16kHz branch — that's all iPhone audio capture needs.

## Development

Contributing or regenerating weights from the ONNX source:

```bash
git clone <repo-url> && cd silvero-vad-js
npm install
python3 -m pip install -r scripts/requirements.txt

# One-time: download ONNX model + generate weights + golden fixtures
curl -L -o weights/silero_vad_v5.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
npm run weights:export
npm run weights:golden

npm test           # run all tests (28 pass)
npm run test:watch # TDD
npm run demo       # http://localhost:8000/examples/index.html
```

All primitive ops (`src/ops.js`) are pure functions with unit tests. The integration test (`tests/silero_vad.test.js`) validates the forward pass against the official `silero-vad` Python wrapper output within 1e-4 per frame.

## License

MIT (matches the SileroVAD model license).
