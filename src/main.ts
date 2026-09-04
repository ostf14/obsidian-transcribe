import {
	AbstractInputSuggest,
	App,
	Notice,
	Plugin,
	PluginSettingTab,
	Setting,
	TFile,
	TFolder,
	normalizePath,
} from "obsidian";
import { WHISPER_LANGUAGES } from "./languages";

const AUDIO_EXTENSIONS = ["ogg", "oga", "m4a", "mp3", "wav", "webm", "opus"];

/**
 * Whisper's own checkpoint names (tiny/base/small/medium/large) say nothing about
 * what a user actually chooses between, so the dropdown is labelled by the
 * trade-off — speed against accuracy — with the download size, which is the other
 * thing that matters, especially on a phone.
 */
const MODELS: ReadonlyArray<{
	id: string;
	label: string;
}> = [
	{ id: "Xenova/whisper-tiny", label: "Fastest, least accurate — 40 MB" },
	{ id: "Xenova/whisper-base", label: "Fast — 75 MB" },
	{ id: "Xenova/whisper-small", label: "Balanced — 250 MB" },
	{ id: "Xenova/whisper-medium", label: "Accurate, slow — 750 MB" },
	{
		id: "onnx-community/whisper-large-v3-turbo",
		label: "Most accurate, desktop only — 800 MB",
	},
];

const AUTO_LANGUAGE = "auto";

interface WhisperTranscribeSettings {
	watchFolder: string;
	outputFolder: string;
	modelId: string;
	/** Whisper language name, or AUTO_LANGUAGE to let the model detect it. */
	language: string;
	noteHeader: string;
	autoTranscribe: boolean;
}

const DEFAULT_SETTINGS: WhisperTranscribeSettings = {
	watchFolder: "",
	outputFolder: "",
	modelId: "Xenova/whisper-base",
	language: AUTO_LANGUAGE,
	noteHeader: "[Source: voice memo «{{filename}}»]",
	autoTranscribe: true,
};

// Lazily loaded — transformers.js is a large dependency, only pull it in
// once transcription is actually needed, not on every Obsidian startup.
type AsrPipeline = (
	audio: Float32Array,
	options?: Record<string, unknown>
) => Promise<{ text: string }>;

export default class WhisperTranscribePlugin extends Plugin {
	settings!: WhisperTranscribeSettings;

	private asrPipelinePromise: Promise<AsrPipeline> | null = null;
	private queue: TFile[] = [];
	private processing = false;
	private statusEl!: HTMLElement;

	async onload() {
		await this.loadSettings();
		this.addSettingTab(new WhisperTranscribeSettingTab(this.app, this));

		this.statusEl = this.addStatusBarItem();
		this.statusEl.setText("");

		// Only start listening once the vault has finished its initial load:
		// "create" also fires for every existing file while Obsidian indexes the
		// vault (and in bulk when sync pulls a batch), which would otherwise
		// queue up the entire audio folder at once.
		this.app.workspace.onLayoutReady(() => {
			this.registerEvent(
				this.app.vault.on("create", (file) => {
					if (!this.settings.autoTranscribe) return;
					if (!(file instanceof TFile)) return;
					if (!this.isInWatchFolder(file)) return;
					if (!this.isAudioFile(file)) return;
					this.enqueue(file);
				})
			);
		});

		this.addCommand({
			id: "transcribe-active-file",
			name: "Transcribe current audio file",
			checkCallback: (checking) => {
				const file = this.app.workspace.getActiveFile();
				const isAudio = !!file && this.isAudioFile(file);
				if (checking) return isAudio;
				if (file) this.enqueue(file);
				return true;
			},
		});
	}

	onunload() {
		this.queue = [];
	}

	private isInWatchFolder(file: TFile): boolean {
		const watch = normalizePath(this.settings.watchFolder);
		if (!watch) return true;
		return (
			file.path === watch ||
			file.path.startsWith(watch + "/")
		);
	}

	private isAudioFile(file: TFile): boolean {
		return AUDIO_EXTENSIONS.includes(file.extension.toLowerCase());
	}

	private enqueue(file: TFile) {
		if (this.queue.find((f) => f.path === file.path)) return;
		this.queue.push(file);
		void this.processQueue();
	}

	private async processQueue() {
		if (this.processing) return;
		this.processing = true;
		try {
			while (this.queue.length > 0) {
				const file = this.queue.shift();
				if (!file) continue;
				await this.transcribeFile(file);
			}
		} finally {
			this.processing = false;
			this.statusEl.setText("");
		}
	}

	private async transcribeFile(file: TFile) {
		const outPath = this.buildOutputPath(file);
		if (await this.app.vault.adapter.exists(outPath)) {
			return; // already transcribed, never overwrite
		}

		const queued = this.queue.length;
		this.statusEl.setText(
			`Transcribing ${file.name}${queued > 0 ? ` (+${queued} queued)` : ""}…`
		);
		const notice = new Notice(`Transcribing "${file.name}"…`, 0);

		try {
			const audio = await this.decodeToPcm16k(file);
			const pipeline = await this.getPipeline();
			const result = await pipeline(audio, {
				// Undefined lets Whisper detect the language itself, which is its
				// native behaviour; a value pins it and skips detection.
				language:
					this.settings.language === AUTO_LANGUAGE
						? undefined
						: this.settings.language,
				task: "transcribe",
				chunk_length_s: 30,
				stride_length_s: 5,
			});

			const text = result.text.trim();
			const header = this.settings.noteHeader
				.replace(/\{\{filename\}\}/g, file.name)
				.trim();
			const content = header ? `${header}\n\n${text}\n` : `${text}\n`;

			await this.ensureFolder(this.outputFolderFor(file));
			await this.app.vault.create(outPath, content);

			notice.setMessage(`Transcribed "${file.name}"`);
		} catch (err) {
			console.error("whisper-transcribe: failed on", file.path, err);
			notice.setMessage(
				`Could not transcribe "${file.name}" — see the developer console`
			);
		} finally {
			window.setTimeout(() => notice.hide(), 4000);
		}
	}

	/** Empty setting means "next to the audio file", which needs no configuration
	 *  to make sense in a vault this plugin knows nothing about. */
	private outputFolderFor(file: TFile): string {
		const configured = this.settings.outputFolder.trim();
		if (configured) return normalizePath(configured);
		return file.parent?.path ?? "";
	}

	private buildOutputPath(file: TFile): string {
		const { date, title } = parseVoiceMemoName(file.basename);
		// Telegram-style names carry their own timestamp; anything else falls back
		// to the file's own date rather than a placeholder.
		const resolvedDate = date ?? formatDate(file.stat?.ctime ?? Date.now());
		const safeTitle = title.replace(/[\\/:*?"<>|]/g, "_");
		const outFolder = this.outputFolderFor(file);
		const name = `${resolvedDate} ${safeTitle}.md`;
		return normalizePath(outFolder ? `${outFolder}/${name}` : name);
	}

	private async ensureFolder(path: string) {
		if (!path) return; // vault root
		const normalized = normalizePath(path);
		const existing = this.app.vault.getAbstractFileByPath(normalized);
		if (existing instanceof TFolder) return;
		await this.app.vault.createFolder(normalized).catch(() => {
			/* already exists — race with another create, ignore */
		});
	}

	private async getPipeline(): Promise<AsrPipeline> {
		if (!this.asrPipelinePromise) {
			this.asrPipelinePromise = this.loadPipeline().catch((err) => {
				// Never cache a failed load: otherwise one bad attempt (offline,
				// interrupted download) would poison every later file until restart.
				this.asrPipelinePromise = null;
				throw err;
			});
		}
		return this.asrPipelinePromise;
	}

	private loadPipeline(): Promise<AsrPipeline> {
		// The mask has to stay in place for the whole load: the ONNX Runtime wasm
		// glue only runs its Node check when the backend initialises, which happens
		// during session creation inside pipeline(), not at import time.
		return withBrowserLikeProcess(() => this.buildPipeline());
	}

	private async buildPipeline(): Promise<AsrPipeline> {
		const { pipeline, env } = await import("@huggingface/transformers");
		env.allowLocalModels = false;
		env.useBrowserCache = true;

		// Multi-threaded WASM needs SharedArrayBuffer, which requires cross-origin
		// isolation that Obsidian's app:// context does not provide. Fall back to a
		// single thread there instead of failing to spin up the worker pool.
		const wasmBackend = env.backends?.onnx?.wasm;
		if (
			wasmBackend &&
			!(typeof SharedArrayBuffer !== "undefined" && globalThis.crossOriginIsolated)
		) {
			wasmBackend.numThreads = 1;
		}

		const progress_callback = (p: { status: string; progress?: number }) => {
			if (p.status === "progress" && typeof p.progress === "number") {
				this.statusEl.setText(
					`Downloading model… ${Math.round(p.progress)}%`
				);
			}
		};

		const hasWebGpu = typeof navigator !== "undefined" && "gpu" in navigator;

		this.statusEl.setText("Loading model…");
		if (hasWebGpu) {
			try {
				const gpuAsr = await pipeline(
					"automatic-speech-recognition",
					this.settings.modelId,
					{ device: "webgpu", progress_callback }
				);
				return gpuAsr as unknown as AsrPipeline;
			} catch (err) {
				// A machine can advertise navigator.gpu and still fail to bring up a
				// usable adapter (old drivers, blocklisted GPU). WASM always works.
				console.warn(
					"whisper-transcribe: WebGPU unavailable, falling back to WASM",
					err
				);
				this.statusEl.setText("WebGPU unavailable, using WASM…");
			}
		}

		const asr = await pipeline(
			"automatic-speech-recognition",
			this.settings.modelId,
			{ device: "wasm", progress_callback }
		);
		return asr as unknown as AsrPipeline;
	}

	/** Decode any browser-supported audio file into mono 16kHz PCM, as Whisper expects. */
	private async decodeToPcm16k(file: TFile): Promise<Float32Array> {
		const arrayBuffer = await this.app.vault.readBinary(file);
		const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
		const probeCtx = new AudioCtx();
		let decoded: AudioBuffer;
		try {
			decoded = await probeCtx.decodeAudioData(arrayBuffer.slice(0));
		} finally {
			void probeCtx.close();
		}

		const offline = new OfflineAudioContext(
			1,
			Math.ceil(decoded.duration * 16000),
			16000
		);
		const source = offline.createBufferSource();
		source.buffer = decoded;
		source.connect(offline.destination);
		source.start(0);
		const rendered = await offline.startRendering();
		return rendered.getChannelData(0);
	}

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings() {
		await this.saveData(this.settings);
	}
}

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
 */
async function withBrowserLikeProcess<T>(fn: () => Promise<T>): Promise<T> {
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

function formatDate(timestamp: number): string {
	const d = new Date(timestamp);
	const pad = (n: number) => String(n).padStart(2, "0");
	return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** Parses the Telegram-export naming convention:
 *  "Title DD-MM-YYYY_HH-MM-SS.ext" -> { date: "YYYY-MM-DD", title: "Title" }.
 *  Returns a null date when the name carries no timestamp, so the caller can fall
 *  back to the file's own date. */
function parseVoiceMemoName(basename: string): {
	date: string | null;
	title: string;
} {
	const match = basename.match(
		/@?(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})\s*$/
	);
	if (!match) {
		return { date: null, title: basename.trim() || "Voice memo" };
	}
	const [, dd, mm, yyyy] = match;
	const title = basename.slice(0, match.index).replace(/@\s*$/, "").trim();
	return { date: `${yyyy}-${mm}-${dd}`, title: title || "Voice memo" };
}


/** Folder autocomplete for a plain text input, so a vault path is picked rather
 *  than typed from memory. */
class FolderSuggest extends AbstractInputSuggest<TFolder> {
	constructor(app: App, private input: HTMLInputElement) {
		super(app, input);
	}

	getSuggestions(query: string): TFolder[] {
		const lower = query.toLowerCase();
		return this.app.vault
			.getAllFolders(true)
			.filter((folder) => folder.path.toLowerCase().includes(lower))
			.slice(0, 100);
	}

	renderSuggestion(folder: TFolder, el: HTMLElement): void {
		el.setText(folder.path === "" ? "/" : folder.path);
	}

	selectSuggestion(folder: TFolder): void {
		this.input.value = folder.path;
		this.input.trigger("input");
		this.close();
	}
}

class WhisperTranscribeSettingTab extends PluginSettingTab {
	plugin: WhisperTranscribePlugin;

	constructor(app: App, plugin: WhisperTranscribePlugin) {
		super(app, plugin);
		this.plugin = plugin;
	}

	display(): void {
		const { containerEl } = this;
		containerEl.empty();

		new Setting(containerEl)
			.setName("Transcribe automatically")
			.setDesc(
				"Transcribe each new audio file as soon as it appears in the watched folder."
			)
			.addToggle((toggle) =>
				toggle
					.setValue(this.plugin.settings.autoTranscribe)
					.onChange(async (value) => {
						this.plugin.settings.autoTranscribe = value;
						await this.plugin.saveSettings();
					})
			);

		new Setting(containerEl)
			.setName("Watched folder")
			.setDesc(
				"New audio files here (and in its subfolders) get transcribed. Leave empty to watch the whole vault."
			)
			.addText((text) => {
				text
					.setPlaceholder("Whole vault")
					.setValue(this.plugin.settings.watchFolder)
					.onChange(async (value) => {
						this.plugin.settings.watchFolder = value.trim();
						await this.plugin.saveSettings();
					});
				new FolderSuggest(this.app, text.inputEl);
			});

		new Setting(containerEl)
			.setName("Save transcripts to")
			.setDesc(
				"Where the transcript notes go. Leave empty to put each one next to its audio file."
			)
			.addText((text) => {
				text
					.setPlaceholder("Next to the audio file")
					.setValue(this.plugin.settings.outputFolder)
					.onChange(async (value) => {
						this.plugin.settings.outputFolder = value.trim();
						await this.plugin.saveSettings();
					});
				new FolderSuggest(this.app, text.inputEl);
			});

		new Setting(containerEl)
			.setName("Model")
			.setDesc(
				"Larger models are more accurate but slower, and are downloaded once before first use. On a phone, prefer one of the first two."
			)
			.addDropdown((dropdown) => {
				for (const model of MODELS) {
					dropdown.addOption(model.id, model.label);
				}
				dropdown
					.setValue(this.plugin.settings.modelId)
					.onChange(async (value) => {
						this.plugin.settings.modelId = value;
						await this.plugin.saveSettings();
						new Notice(
							"Model changed. It is downloaded on the next transcription."
						);
					});
			});

		new Setting(containerEl)
			.setName("Language")
			.setDesc(
				"Whisper detects the language on its own. Pick one only to pin it — useful for short or noisy recordings, where detection can guess wrong."
			)
			.addDropdown((dropdown) => {
				dropdown.addOption(AUTO_LANGUAGE, "Detect automatically");
				for (const [, name] of WHISPER_LANGUAGES) {
					dropdown.addOption(name.toLowerCase(), name);
				}
				dropdown
					.setValue(this.plugin.settings.language)
					.onChange(async (value) => {
						this.plugin.settings.language = value;
						await this.plugin.saveSettings();
					});
			});

		new Setting(containerEl)
			.setName("Note header")
			.setDesc(
				"First line of each transcript note. {{filename}} is replaced with the audio file's name. Leave empty for no header."
			)
			.addText((text) =>
				text
					.setPlaceholder("[Source: voice memo «{{filename}}»]")
					.setValue(this.plugin.settings.noteHeader)
					.onChange(async (value) => {
						this.plugin.settings.noteHeader = value;
						await this.plugin.saveSettings();
					})
			);
	}
}
