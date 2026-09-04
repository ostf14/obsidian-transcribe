import { createAsrPipeline } from "./browser-process";

/** Injected at build time: the bundled worker, inlined so the plugin ships as a
 *  single main.js. */
declare const __ASR_WORKER_SOURCE__: string;

export interface AsrHooks {
	onProgress?: (percent: number) => void;
	onWasmFallback?: () => void;
	onTranscribing?: () => void;
}

/** How long the worker may go completely silent — no progress, no result — before
 *  we treat it as dead. Long files are fine: inference still reports nothing for
 *  minutes, so this only has to be longer than any plausible quiet stretch. */
const SILENCE_TIMEOUT_MS = 15 * 60 * 1000;

interface Pending {
	resolve: (text: string) => void;
	reject: (err: Error) => void;
}

/**
 * Talks to the transcription worker, and transparently falls back to running the
 * model in-process if a worker cannot be started (some embedded webviews refuse
 * blob-backed module workers). The fallback keeps transcription working; it just
 * blocks the UI while it runs, which is exactly what the worker exists to avoid.
 */
export class AsrClient {
	private worker: Worker | null = null;
	private workerUrl: string | null = null;
	private workerUnavailable = false;
	private pending = new Map<number, Pending>();
	private nextId = 1;
	private watchdog: number | null = null;
	private inProcess: Promise<
		(audio: Float32Array, options?: Record<string, unknown>) => Promise<{ text: string }>
	> | null = null;

	constructor(private hooks: AsrHooks = {}) {}

	async transcribe(
		audio: Float32Array,
		modelId: string,
		language: string | undefined
	): Promise<string> {
		const worker = this.ensureWorker();
		if (worker) {
			try {
				return await this.runInWorker(worker, audio, modelId, language);
			} catch (err) {
				if (!this.workerUnavailable) throw err;
				// Worker died on startup — retry once in-process rather than failing.
			}
		}
		return this.runInProcess(audio, modelId, language);
	}

	/** Drops the loaded model, e.g. after the model setting changed. */
	reset() {
		this.terminate();
		this.inProcess = null;
		this.workerUnavailable = false;
	}

	terminate() {
		for (const pending of this.pending.values()) {
			pending.reject(new Error("Transcription cancelled"));
		}
		this.pending.clear();
		this.worker?.terminate();
		this.worker = null;
		if (this.workerUrl) {
			URL.revokeObjectURL(this.workerUrl);
			this.workerUrl = null;
		}
	}

	private ensureWorker(): Worker | null {
		if (this.workerUnavailable) return null;
		if (this.worker) return this.worker;

		try {
			const blob = new Blob([__ASR_WORKER_SOURCE__], {
				type: "text/javascript",
			});
			this.workerUrl = URL.createObjectURL(blob);
			const worker = new Worker(this.workerUrl, { type: "module" });
			worker.onmessage = (event) => this.handleMessage(event.data);
			worker.onerror = (event) => {
				console.error("whisper-transcribe: worker error", event.message);
				this.failAllPending(new Error(event.message || "Worker failed"));
				this.workerUnavailable = true;
				this.terminate();
			};
			this.worker = worker;
			return worker;
		} catch (err) {
			console.warn(
				"whisper-transcribe: could not start worker, running in-process",
				err
			);
			this.workerUnavailable = true;
			return null;
		}
	}

	private handleMessage(message: any) {
		this.noteActivity();
		switch (message?.type) {
			case "progress":
				this.hooks.onProgress?.(message.percent);
				return;
			case "wasm-fallback":
				this.hooks.onWasmFallback?.();
				return;
			case "transcribing":
				this.hooks.onTranscribing?.();
				return;
			case "result": {
				const pending = this.pending.get(message.id);
				this.pending.delete(message.id);
				pending?.resolve(message.text ?? "");
				return;
			}
			case "error": {
				const pending = this.pending.get(message.id);
				this.pending.delete(message.id);
				pending?.reject(new Error(message.error ?? "Transcription failed"));
				return;
			}
		}
	}

	private failAllPending(err: Error) {
		for (const pending of this.pending.values()) pending.reject(err);
		this.pending.clear();
		this.clearWatchdog();
	}

	/** Restarts the silence watchdog on every sign of life from the worker. */
	private noteActivity() {
		this.clearWatchdog();
		if (this.pending.size === 0) return;
		this.watchdog = window.setTimeout(() => {
			this.failAllPending(
				new Error(
					`The worker went silent for ${Math.round(
						SILENCE_TIMEOUT_MS / 60000
					)} minutes — it most likely ran out of memory. Try a smaller model.`
				)
			);
			// Whatever state it is in, it is not worth reusing.
			this.terminate();
		}, SILENCE_TIMEOUT_MS);
	}

	private clearWatchdog() {
		if (this.watchdog !== null) {
			window.clearTimeout(this.watchdog);
			this.watchdog = null;
		}
	}

	private runInWorker(
		worker: Worker,
		audio: Float32Array,
		modelId: string,
		language: string | undefined
	): Promise<string> {
		const id = this.nextId++;
		return new Promise<string>((resolve, reject) => {
			this.pending.set(id, { resolve, reject });
			// Deliberately copied rather than transferred: transferring detaches the
			// buffer here, so the in-process fallback below would be handed an empty
			// recording if the worker turned out to be unusable. A few megabytes of
			// PCM is a cheap price for that not to happen.
			worker.postMessage({ type: "transcribe", id, modelId, audio, language });
			this.noteActivity();
		});
	}

	private async runInProcess(
		audio: Float32Array,
		modelId: string,
		language: string | undefined
	): Promise<string> {
		if (!this.inProcess) {
			this.inProcess = createAsrPipeline(modelId, {
				onProgress: this.hooks.onProgress,
				onWasmFallback: () => this.hooks.onWasmFallback?.(),
			}).catch((err) => {
				this.inProcess = null;
				throw err;
			});
		}
		const pipeline = await this.inProcess;
		const result = await pipeline(audio, {
			language,
			task: "transcribe",
			chunk_length_s: 30,
			stride_length_s: 5,
		});
		return result.text;
	}
}
