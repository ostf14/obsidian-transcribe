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
}

export const MODELS: ReadonlyArray<ModelSpec> = [
	{ id: "Xenova/whisper-tiny", label: "Fastest, least accurate — 40 MB" },
	{ id: "Xenova/whisper-base", label: "Fast — 75 MB" },
	{ id: "Xenova/whisper-small", label: "Balanced — 250 MB" },
	{ id: "Xenova/whisper-medium", label: "Accurate, slow — 750 MB", heavy: true },
	{
		id: "onnx-community/whisper-large-v3-turbo",
		label: "Most accurate, desktop only — 800 MB",
		heavy: true,
	},
];

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
