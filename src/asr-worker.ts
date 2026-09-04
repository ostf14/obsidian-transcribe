/**
 * Runs Whisper off the UI thread. Obsidian stays responsive while a recording is
 * being transcribed, which on longer files is the difference between a usable app
 * and a frozen one.
 *
 * The main thread decodes the audio (the Web Audio API is not available here) and
 * hands over plain PCM; everything model-shaped happens in this file.
 */
import { createAsrPipeline } from "./browser-process";

type AsrPipeline = (
	audio: Float32Array,
	options?: Record<string, unknown>
) => Promise<{ text: string }>;

type IncomingMessage =
	| { type: "init"; modelId: string }
	| {
			type: "transcribe";
			id: number;
			modelId: string;
			audio: Float32Array;
			language?: string;
	  };

let pipelinePromise: Promise<AsrPipeline> | null = null;
let loadedModelId: string | null = null;

function post(message: Record<string, unknown>, transfer?: Transferable[]) {
	(self as unknown as Worker).postMessage(message, transfer ?? []);
}

function getPipeline(modelId: string): Promise<AsrPipeline> {
	if (!pipelinePromise || loadedModelId !== modelId) {
		loadedModelId = modelId;
		pipelinePromise = createAsrPipeline(modelId, {
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
	return pipelinePromise;
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

	if (message.type === "transcribe") {
		activeId = message.id;
		try {
			const pipeline = await getPipeline(message.modelId);
			post({ type: "transcribing" });
			const result = await pipeline(message.audio, {
				language: message.language,
				task: "transcribe",
				chunk_length_s: 30,
				stride_length_s: 5,
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
