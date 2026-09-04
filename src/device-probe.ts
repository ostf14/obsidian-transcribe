/**
 * Works out what this machine can actually run.
 *
 * WebGPU deliberately exposes no video-memory figure — that would be a
 * fingerprinting surface — so the only honest way to find out how much is
 * available is to ask for it: allocate buffers until the driver reports an
 * out-of-memory condition, then hand them all back. The spec provides exactly
 * this in `pushErrorScope("out-of-memory")`, so it is a supported question to
 * ask, not a trick.
 *
 * The result is deliberately conservative and reported as an estimate. It tells
 * a user which models are worth choosing without them having to discover the
 * answer as a failed transcription twenty minutes in.
 */

export interface DeviceReport {
	hasWebGpu: boolean;
	/** Vendor/architecture strings, when the browser reveals them. */
	description: string | null;
	/** Largest single allocation the adapter permits. A model whose biggest
	 *  weight tensor exceeds this cannot run regardless of free memory. */
	maxBufferBytes: number | null;
	/** Roughly how much the GPU let us hold at once, in bytes. */
	usableGpuBytes: number | null;
	/** Browser's coarse system-RAM hint, in GB, when available. */
	systemMemoryGb: number | null;
	/** Set when probing failed outright. */
	error: string | null;
}

const PROBE_CHUNK_BYTES = 128 * 1024 * 1024; // 128 MB
// The largest model needs roughly 2 GB, so there is nothing to learn past 4, and
// squeezing a machine dry is rude when someone is working on it.
const PROBE_CEILING_BYTES = 4 * 1024 * 1024 * 1024;

export async function probeDevice(): Promise<DeviceReport> {
	const report: DeviceReport = {
		hasWebGpu: false,
		description: null,
		maxBufferBytes: null,
		usableGpuBytes: null,
		systemMemoryGb:
			typeof (navigator as any).deviceMemory === "number"
				? (navigator as any).deviceMemory
				: null,
		error: null,
	};

	const gpu = (navigator as any).gpu;
	if (!gpu) return report;

	try {
		const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
		if (!adapter) {
			report.error = "No GPU adapter available";
			return report;
		}
		report.hasWebGpu = true;
		report.maxBufferBytes = adapter.limits?.maxBufferSize ?? null;

		const info = adapter.info ?? (await adapter.requestAdapterInfo?.());
		if (info) {
			const parts = [info.vendor, info.architecture, info.device, info.description]
				.filter((p: string) => p)
				.join(" ");
			report.description = parts || null;
		}

		const device = await adapter.requestDevice();
		report.usableGpuBytes = await measureUsableMemory(
			device,
			report.maxBufferBytes
		);
		device.destroy?.();
	} catch (err) {
		report.error = String((err as Error)?.message ?? err);
	}

	return report;
}

/** Allocates in steps until the driver refuses, then releases everything. */
async function measureUsableMemory(
	device: any,
	maxBufferBytes: number | null
): Promise<number> {
	// An adapter may cap a single buffer below our step size, in which case every
	// allocation would fail and the machine would look like it had no memory at all.
	const step = Math.min(
		PROBE_CHUNK_BYTES,
		maxBufferBytes && maxBufferBytes > 0 ? maxBufferBytes : PROBE_CHUNK_BYTES
	);
	const storageUsage =
		typeof (globalThis as any).GPUBufferUsage?.STORAGE === "number"
			? (globalThis as any).GPUBufferUsage.STORAGE
			: 0x0080;

	const buffers: any[] = [];
	let total = 0;
	try {
		while (total < PROBE_CEILING_BYTES) {
			device.pushErrorScope("out-of-memory");
			let buffer: any;
			try {
				buffer = device.createBuffer({ size: step, usage: storageUsage });
			} catch {
				await device.popErrorScope();
				break;
			}
			const oom = await device.popErrorScope();
			if (oom) {
				buffer.destroy?.();
				break;
			}
			buffers.push(buffer);
			total += step;
		}
	} finally {
		// Hand it all straight back: this runs on a machine someone is using.
		for (const buffer of buffers) buffer.destroy?.();
	}
	return total;
}
