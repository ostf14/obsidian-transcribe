/**
 * Whisper's own checkpoint names (tiny/base/small/medium/large) say nothing about
 * what a user actually chooses between, so the dropdown is labelled by the
 * trade-off — speed against accuracy — with the download size, which is the other
 * thing that matters, especially on a phone.
 */
export interface ModelSpec {
	id: string;
	label: string;
	/** Big enough that fp32 weights risk exhausting memory while downloading. */
	heavy?: boolean;
	/** Weights actually fetched, in MB, measured from the model repositories for
	 *  the precisions this plugin loads (see dtypeFor). */
	gpuMb: number;
	wasmMb: number;
}

// Download sizes are the weights this plugin actually fetches for its own
// precision choices (see dtypeFor), measured from the model repositories — not
// the nominal size of the original checkpoint, which is what everyone quotes and
// which is nowhere near what ends up on disk.
export const MODELS: ReadonlyArray<ModelSpec> = [
	{
		id: "Xenova/whisper-tiny",
		label: "Fastest, least accurate — ~115 MB",
		gpuMb: 114,
		wasmMb: 39,
	},
	{
		id: "Xenova/whisper-base",
		label: "Fast — ~200 MB",
		gpuMb: 197,
		wasmMb: 73,
	},
	{
		id: "Xenova/whisper-small",
		label: "Balanced — ~560 MB",
		gpuMb: 558,
		wasmMb: 238,
	},
	{
		id: "Xenova/whisper-medium",
		label: "Accurate, slow — ~1 GB",
		heavy: true,
		gpuMb: 1035,
		wasmMb: 740,
	},
	{
		id: "onnx-community/whisper-large-v3-turbo",
		label: "Most accurate — ~1.5 GB",
		heavy: true,
		gpuMb: 1534,
		wasmMb: 1035,
	},
];

/** Weights are not the whole story: inference also needs working space for
 *  activations and the decoder's cache. */
const RUNTIME_OVERHEAD = 1.4;

/** The address space a 32-bit WASM runtime can ever reach, minus room to work in.
 *  This is a hard architectural ceiling, not a property of the machine. */
const WASM_CEILING_MB = 4096 * 0.6;

export type Verdict = "ok" | "tight" | "no";

export interface ModelAssessment {
	verdict: Verdict;
	reason: string;
}

/**
 * Judges whether a model is worth choosing on this machine. Deliberately errs
 * towards "tight" rather than promising something will work: the measurement is
 * an estimate, and being wrong in that direction only costs a warning.
 */
export function assessModel(
	model: ModelSpec,
	device: {
		hasWebGpu: boolean;
		usableGpuBytes: number | null;
		maxBufferBytes: number | null;
	}
): ModelAssessment {
	const mb = (bytes: number) => bytes / (1024 * 1024);

	if (!device.hasWebGpu) {
		const needed = model.wasmMb * RUNTIME_OVERHEAD;
		if (needed > WASM_CEILING_MB) {
			return {
				verdict: "no",
				reason: `needs about ${Math.round(
					needed
				)} MB, past what a 32-bit WASM runtime can address at all`,
			};
		}
		if (model.heavy) {
			return {
				verdict: "tight",
				reason: "runs without a GPU, but expect minutes per recording",
			};
		}
		return { verdict: "ok", reason: "fine on the CPU path" };
	}

	const usable = device.usableGpuBytes;
	if (usable === null) {
		return { verdict: "tight", reason: "could not measure this GPU" };
	}

	const needed = model.gpuMb * RUNTIME_OVERHEAD;
	const available = mb(usable);

	if (needed > available) {
		return {
			verdict: "no",
			reason: `needs about ${Math.round(needed)} MB, GPU offered ${Math.round(
				available
			)} MB`,
		};
	}
	if (needed > available * 0.7) {
		return {
			verdict: "tight",
			reason: `needs about ${Math.round(needed)} MB of the ${Math.round(
				available
			)} MB this GPU offered`,
		};
	}
	return {
		verdict: "ok",
		reason: `about ${Math.round(needed)} MB of ${Math.round(available)} MB`,
	};
}

/**
 * Weight precision to load with.
 *
 * Left unset, transformers.js defaults to fp32 on WebGPU, which downloads and
 * holds the largest possible weights — enough to fail with "Array buffer
 * allocation failed" before inference even starts, especially as the model host
 * sends no content-length, so the download buffer grows by reallocating and
 * copying and briefly needs twice the file size.
 *
 * A quantised decoder is where most of the saving is, at very little cost to
 * accuracy; the encoder stays at full precision on smaller models, and drops to
 * fp16 on the large ones where fp32 is what breaks. WASM keeps the q8 that
 * transformers.js already picks there by default.
 */
/** The precisions transformers.js accepts. */
export type Dtype =
	| "auto"
	| "q8"
	| "fp16"
	| "fp32"
	| "q4"
	| "uint8"
	| "int8"
	| "bnb4"
	| "q4f16";

export function dtypeFor(
	device: "webgpu" | "wasm",
	model: ModelSpec | undefined
): Dtype | Record<string, Dtype> {
	if (device === "wasm") return "q8";
	return {
		encoder_model: model?.heavy ? "fp16" : "fp32",
		decoder_model_merged: "q4",
	};
}

export function findModel(id: string): ModelSpec | undefined {
	return MODELS.find((m) => m.id === id);
}
