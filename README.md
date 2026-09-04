# Whisper Transcribe

Transcribes voice memos and other audio into notes, entirely on your device — no server, no API key, no Python. Runs a Whisper model inside Obsidian itself via [transformers.js](https://github.com/huggingface/transformers.js) (ONNX, WebGPU with a WASM fallback).

## What it does

1. Watches a folder in your vault (or the whole vault, if you leave the setting empty).
2. When a new audio file appears — `.ogg`, `.oga`, `.m4a`, `.mp3`, `.wav`, `.webm`, `.opus` — it transcribes it.
3. Writes the transcript as a note named `YYYY-MM-DD Title.md`. The date and title come from the audio file's name when it carries a timestamp (`Title DD-MM-YYYY_HH-MM-SS.ogg`, the Telegram export format), otherwise from the file's own name and date.
4. Leaves the audio file untouched, and never overwrites an existing transcript.

There is also a command, **Transcribe current audio file**, for doing one on demand.

## Models

The model is downloaded once on first use and cached, so afterwards it works offline. Pick the trade-off in settings:

| Setting | Whisper model | Size |
| --- | --- | --- |
| Fastest, least accurate | tiny | 40 MB |
| Fast | base | 75 MB |
| Balanced | small | 250 MB |
| Accurate, slow | medium | 750 MB |
| Most accurate, desktop only | large-v3-turbo | 800 MB |

Accuracy differences are large on names, jargon and unclear speech. On a phone, stay with one of the first two.

Where WebGPU is available it is used; otherwise the plugin falls back to WASM, which works everywhere but is considerably slower.

## Language

Whisper detects the language by itself, so leave **Language** on *Detect automatically*. Pin a language only when detection guesses wrong, which mostly happens on very short or noisy recordings.

## Installation

Not in the community plugin directory yet. Either:

**With [BRAT](https://github.com/TfTHacker/obsidian42-brat):** BRAT settings → *Add Beta plugin* → `ostf14/obsidian-transcribe`.

**By hand:** from the [latest release](https://github.com/ostf14/obsidian-transcribe/releases/latest), download `whisper-transcribe.zip` and unpack it into `<vault>/.obsidian/plugins/` — it already contains a correctly named folder. Then enable the plugin under Settings → Community plugins.

To update afterwards, replacing `main.js` in that folder is enough.

## Known limitations

- Transcription currently runs on the main thread, so Obsidian's interface can stutter or freeze while a long recording is being processed.
- The first run needs an internet connection to fetch the model.
- Requires Obsidian 1.4.0 or newer.

## Development

```bash
npm install
npm run dev    # watch build
npm run build  # production build (main.js)
```

## License

MIT
