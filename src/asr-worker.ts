/**
 * Runs Whisper off the UI thread. Obsidian stays responsive while a recording is
 * being transcribed, which on longer files is the difference between a usable app
 * and a frozen one.
 *
 * The main thread decodes the audio (the Web Audio API is not available here) and
 * hands over plain PCM; everything model-shaped happens in this file.
 */
import { AsrPipeline, createAsrPipeline } from "./browser-process";

type IncomingMessage =
	| { type: "init"; modelId: string }
	| { type: "release" }
	| {
			type: "transcribe";
			id: number;
			modelId: string;
			audio: Float32Array;
			language?: string;
			forceWasm?: boolean;
	  };

/** Chunking the pipeline applies to long audio. Each successive window starts
 *  this many seconds later, which is what lets a timestamp inside a window be
 *  turned into a position in the whole recording. */
const CHUNK_LENGTH_S = 30;
const STRIDE_LENGTH_S = 5;
const CHUNK_JUMP_S = CHUNK_LENGTH_S - 2 * STRIDE_LENGTH_S;

let pipelinePromise: Promise<AsrPipeline> | null = null;
let loadedModelId: string | null = null;

function post(message: Record<string, unknown>, transfer?: Transferable[]) {
	(self as unknown as Worker).postMessage(message, transfer ?? []);
}

function getPipeline(modelId: string, forceWasm = false): Promise<AsrPipeline> {
	if (!pipelinePromise || loadedModelId !== modelId) {
		loadedModelId = modelId;
		pipelinePromise = createAsrPipeline(modelId, {
			forceWasm,
			onProgress: (percent) => post({ type: "progress", percent }),
			onWasmFallback: (err) =>
				post({ type: "wasm-fallback", error: String(err) }),
		}).catch((err) => {
			// Never cache a failed load: one bad attempt (offline, interrupted
			// download) would otherwise poison every later file.
			pipelinePromise = null;
			loadedModelId = null;
			throw err;
		});
	}
	return pipelinePromise!;
}

/** The id currently being worked on, so a failure that escapes the awaited chain
 *  can still be reported against it instead of leaving the caller waiting. */
let activeId: number | null = null;

function reportEscapedFailure(reason: unknown) {
	const detail = String(
		(reason as { message?: string })?.message ?? reason ?? "unknown error"
	);
	if (activeId !== null) {
		const id = activeId;
		activeId = null;
		post({ type: "error", id, error: detail });
	}
	// A half-initialised pipeline is not reusable after an out-of-memory failure.
	pipelinePromise = null;
	loadedModelId = null;
}

// Model loading detaches promises internally (streamed downloads, wasm init), so
// an out-of-memory RangeError can surface here rather than at our await. Without
// this the caller would wait forever on a request that already died.
self.addEventListener("unhandledrejection", (event: PromiseRejectionEvent) => {
	event.preventDefault();
	reportEscapedFailure(event.reason);
});

self.addEventListener("error", (event: ErrorEvent) => {
	reportEscapedFailure(event.message);
});

self.onmessage = async (event: MessageEvent<IncomingMessage>) => {
	const message = event.data;

	if (message.type === "init") {
		try {
			await getPipeline(message.modelId);
			post({ type: "ready" });
		} catch (err) {
			post({ type: "init-error", error: String(err) });
		}
		return;
	}

	if (message.type === "release") {
		await releasePipeline();
		return;
	}

	if (message.type === "transcribe") {
		activeId = message.id;
		try {
			const pipeline = await getPipeline(message.modelId, message.forceWasm);
			const duration = message.audio.length / 16000;
			post({ type: "transcribing", id: message.id, duration });

			const result = await pipeline(message.audio, {
				language: message.language,
				task: "transcribe",
				chunk_length_s: CHUNK_LENGTH_S,
				stride_length_s: STRIDE_LENGTH_S,
				streamer: await buildProgressStreamer(pipeline, message.id, duration),
			});

			if (activeId === message.id) {
				activeId = null;
				post({ type: "result", id: message.id, text: result.text });
			}
		} catch (err) {
			if (activeId === message.id) {
				activeId = null;
				post({ type: "error", id: message.id, error: String(err) });
			}
		}
	}
};

/**
 * Reports how far into the recording Whisper has got.
 *
 * The pipeline exposes no per-chunk hook, but Whisper's own streamer reports the
 * timestamps it emits. Those restart at zero in every 30-second window, so a
 * timestamp going backwards means a new window has begun, and the elapsed audio
 * is that window's offset plus the position inside it.
 */
async function buildProgressStreamer(
	pipeline: AsrPipeline,
	id: number,
	duration: number
) {
	const { WhisperTextStreamer } = await import("@huggingface/transformers");

	let windowIndex = 0;
	let lastTime = 0;
	let furthest = 0;

	const report = (time: number) => {
		if (time + 1e-6 < lastTime) windowIndex += 1; // timestamps reset per window
		lastTime = time;
		const elapsed = windowIndex * CHUNK_JUMP_S + time;
		if (elapsed <= furthest) return;
		furthest = elapsed;
		post({
			type: "transcribe-progress",
			id,
			seconds: Math.min(elapsed, duration),
			duration,
		});
	};

	return new WhisperTextStreamer(pipeline.tokenizer as any, {
		on_chunk_start: report,
		on_chunk_end: report,
	});
}

async function releasePipeline() {
	const pending = pipelinePromise;
	pipelinePromise = null;
	loadedModelId = null;
	if (!pending) return;
	try {
		const pipeline = await pending;
		await pipeline.dispose?.();
	} catch {
		/* nothing to release */
	}
}
