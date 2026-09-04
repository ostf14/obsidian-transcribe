/**
 * Run `fn` with `process` looking non-Node, so that transformers.js and the ONNX
 * Runtime it loads both take their browser code paths.
 *
 * Two separate checks force this, and both are hit in Obsidian's desktop app,
 * because an Electron renderer is a browser that also exposes Node:
 *
 * 1. transformers.js, at module-evaluation time:
 *        IS_NODE_ENV = typeof process !== 'undefined' && process?.release?.name === 'node'
 *    On that branch it does `ONNX = ONNX_NODE.default ?? ONNX_NODE` — the native
 *    onnxruntime-node addon, which a distributed plugin cannot ship and which does
 *    not exist on mobile at all. ONNX stays undefined and session creation dies
 *    on `undefined.create`.
 *
 * 2. The ONNX Runtime wasm glue, at the top level of its module:
 *        var isNode = typeof globalThis.process?.versions?.node == 'string';
 *        if (isNode) isPthread = (await import('worker_threads')).workerData === ...
 *    That import cannot resolve in a browser context, so the module throws and
 *    every backend — WebGPU and WASM alike, they share this glue — reports
 *    "no available backend found". (A sibling check in the same file does guard
 *    for Electron via `"renderer" != process.type`; this one does not.)
 *
 * Check 2 fires while the wasm backend initialises, i.e. during session creation,
 * not merely on import — so the mask has to cover the whole model load.
 *
 * Both values are replaced with String *objects* rather than removed: `typeof` no
 * longer reports "string" and `===` against a literal fails, which is exactly what
 * these two checks test, while any other code reading them during the window still
 * sees something that concatenates and compares loosely like the real version.
 *
 * Inside a worker this is normally a no-op — there is no `process` there — which is
 * the better reason to run the model in one.
 */
export async function withBrowserLikeProcess<T>(fn: () => Promise<T>): Promise<T> {
	const proc =
		typeof process !== "undefined"
			? (process as unknown as Record<string, any>)
			: undefined;
	if (!proc) return fn();

	const restores: Array<() => void> = [];

	const mask = (owner: Record<string, any>, key: string, realValue: unknown) => {
		const descriptor = Object.getOwnPropertyDescriptor(owner, key);
		if (descriptor && !descriptor.configurable) return;
		Object.defineProperty(owner, key, {
			value: new String(String(realValue ?? "")),
			configurable: true,
			writable: true,
			enumerable: descriptor?.enumerable ?? true,
		});
		restores.push(() => {
			if (descriptor) Object.defineProperty(owner, key, descriptor);
			else delete owner[key];
		});
	};

	if (typeof proc.release?.name === "string") {
		mask(proc.release, "name", proc.release.name);
	}
	if (typeof proc.versions?.node === "string") {
		mask(proc.versions, "node", proc.versions.node);
	}

	try {
		return await fn();
	} finally {
		for (const restore of restores.reverse()) restore();
	}
}

import { dtypeFor, findModel } from "./models";

/** Builds the ASR pipeline, preferring WebGPU and falling back to WASM. Shared by
 *  the worker and by the in-process path used when a worker cannot be created. */
export async function createAsrPipeline(
	modelId: string,
	hooks: {
		onProgress?: (percent: number) => void;
		onWasmFallback?: (err: unknown) => void;
	} = {}
): Promise<(audio: Float32Array, options?: Record<string, unknown>) => Promise<{ text: string }>> {
	return withBrowserLikeProcess(async () => {
		const { pipeline, env } = await import("@huggingface/transformers");
		env.allowLocalModels = false;
		env.useBrowserCache = true;

		// Multi-threaded WASM needs SharedArrayBuffer, which requires cross-origin
		// isolation that Obsidian's app:// context does not provide.
		const wasmBackend = env.backends?.onnx?.wasm;
		if (
			wasmBackend &&
			!(typeof SharedArrayBuffer !== "undefined" && globalThis.crossOriginIsolated)
		) {
			wasmBackend.numThreads = 1;
		}

		const progress_callback = (p: { status: string; progress?: number }) => {
			if (p.status === "progress" && typeof p.progress === "number") {
				hooks.onProgress?.(p.progress);
			}
		};

		const model = findModel(modelId);

		if (typeof navigator !== "undefined" && "gpu" in navigator) {
			try {
				const gpu = await pipeline("automatic-speech-recognition", modelId, {
					device: "webgpu",
					dtype: dtypeFor("webgpu", model),
					progress_callback,
				});
				return gpu as any;
			} catch (err) {
				// A machine can advertise navigator.gpu and still fail to bring up a
				// usable adapter (old drivers, blocklisted GPU), and a large model can
				// run the machine out of memory. WASM is slower but far less demanding.
				hooks.onWasmFallback?.(err);
			}
		}

		const cpu = await pipeline("automatic-speech-recognition", modelId, {
			device: "wasm",
			dtype: dtypeFor("wasm", model),
			progress_callback,
		});
		return cpu as any;
	});
}
