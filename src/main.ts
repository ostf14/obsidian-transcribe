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
import { AsrClient } from "./asr-client";
import { DeviceReport, probeDevice } from "./device-probe";
import { MODELS, assessModel } from "./models";
import { WHISPER_LANGUAGES } from "./languages";

const AUDIO_EXTENSIONS = ["ogg", "oga", "m4a", "mp3", "wav", "webm", "opus"];

const AUTO_LANGUAGE = "auto";

/** How long a loaded model may sit unused before it is released. */
const IDLE_RELEASE_MS = 5 * 60 * 1000;

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

export default class WhisperTranscribePlugin extends Plugin {
	settings!: WhisperTranscribeSettings;

	private asr!: AsrClient;
	private queue: TFile[] = [];
	private processing = false;
	private statusEl!: HTMLElement;
	private activeProgress: NoticeProgress | null = null;

	async onload() {
		await this.loadSettings();
		this.addSettingTab(new WhisperTranscribeSettingTab(this.app, this));

		this.statusEl = this.addStatusBarItem();
		this.statusEl.setText("");

		this.asr = new AsrClient({
			onProgress: (percent) => {
				this.statusEl.setText(`Downloading model… ${Math.round(percent)}%`);
				this.activeProgress?.setStage("Downloading model", percent / 100);
			},
			onWasmFallback: () => {
				this.statusEl.setText("Using WASM (slower)…");
				this.activeProgress?.setStage("Switching to WASM (slower)", null);
			},
			// The model is only loaded once, so telling the two stages apart matters:
			// a long wait during the download means something different to the user
			// than a long wait during inference.
			onTranscribing: (duration) => {
				this.statusEl.setText("Transcribing…");
				this.activeProgress?.startTranscribing(duration);
			},
			onTranscribeProgress: (seconds, duration) => {
				const share = duration > 0 ? seconds / duration : 0;
				this.statusEl.setText(`Transcribing… ${Math.round(share * 100)}%`);
				this.activeProgress?.setStage("Transcribing", share, seconds, duration);
			},
		});

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
		this.asr?.terminate();
	}

	/** Called when the model setting changes, so the next run loads the new one. */
	resetModel() {
		this.asr?.reset();
	}

	private isInWatchFolder(file: TFile): boolean {
		const watch = normalizePath(this.settings.watchFolder.trim());
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
		const notice = new Notice("", 0);
		const progress = new NoticeProgress(notice, file.name, queued);
		this.activeProgress = progress;

		try {
			const audio = await this.decodeToPcm16k(file);
			const transcript = await this.asr.transcribe(
				audio,
				this.settings.modelId,
				// Undefined lets Whisper detect the language itself, which is its
				// native behaviour; a value pins it and skips detection.
				this.settings.language === AUTO_LANGUAGE
					? undefined
					: this.settings.language
			);

			const text = transcript.trim();
			const header = this.settings.noteHeader
				.replace(/\{\{filename\}\}/g, file.name)
				.trim();
			const content = header ? `${header}\n\n${text}\n` : `${text}\n`;

			await this.ensureFolder(this.outputFolderFor(file));
			await this.app.vault.create(outPath, content);

			progress.finish(`Transcribed "${file.name}"`);
		} catch (err) {
			console.error("whisper-transcribe: failed on", file.path, err);
			progress.finish(
				`Could not transcribe "${file.name}": ${
					(err as Error)?.message ?? "see the developer console"
				}`
			);
		} finally {
			this.activeProgress = null;
			window.setTimeout(() => notice.hide(), 6000);
			// Nothing left to do soon? Let the model go rather than pinning hundreds
			// of megabytes for the rest of the Obsidian session.
			if (this.queue.length === 0) this.asr.scheduleRelease(IDLE_RELEASE_MS);
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
 * A progress bar drawn inside the notice.
 *
 * Both stages report real numbers — bytes for the download, position in the
 * recording for the transcription — so the bar never invents motion it does not
 * know about. The one genuinely unknown stretch, loading the weights into the
 * runtime, is shown as a stripeless bar rather than a fake percentage.
 */
class NoticeProgress {
	private labelEl: HTMLElement;
	private barEl: HTMLElement;
	private fillEl: HTMLElement;

	constructor(notice: Notice, private fileName: string, queued: number) {
		const root = notice.noticeEl;
		root.empty();
		root.createEl("div", {
			text: `${fileName}${queued > 0 ? ` (+${queued} queued)` : ""}`,
			attr: { style: "font-weight:600;margin-bottom:4px" },
		});
		this.labelEl = root.createEl("div", {
			text: "Preparing…",
			attr: { style: "font-size:var(--font-ui-smaller);opacity:.8" },
		});
		this.barEl = root.createEl("div", {
			attr: {
				style:
					"margin-top:6px;height:4px;border-radius:2px;overflow:hidden;background:var(--background-modifier-border)",
			},
		});
		this.fillEl = this.barEl.createEl("div", {
			attr: {
				style:
					"height:100%;width:0%;border-radius:2px;background:var(--interactive-accent);transition:width .2s linear",
			},
		});
	}

	setStage(
		label: string,
		share: number | null,
		seconds?: number,
		duration?: number
	) {
		const suffix =
			seconds !== undefined && duration !== undefined
				? ` — ${formatClock(seconds)} of ${formatClock(duration)}`
				: share !== null
					? ` — ${Math.round(share * 100)}%`
					: "";
		this.labelEl.setText(`${label}${suffix}`);
		this.fillEl.style.width = share === null ? "100%" : `${Math.round(share * 100)}%`;
		this.fillEl.style.opacity = share === null ? "0.35" : "1";
	}

	startTranscribing(duration: number) {
		this.setStage("Transcribing", 0, 0, duration);
	}

	finish(message: string) {
		this.labelEl.setText(message);
		this.fillEl.style.width = "100%";
		this.fillEl.style.opacity = "1";
	}
}

function formatClock(seconds: number): string {
	const total = Math.max(0, Math.round(seconds));
	const m = Math.floor(total / 60);
	const s = total % 60;
	return `${m}:${String(s).padStart(2, "0")}`;
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
	/** Kept across re-renders so the probe only runs when asked for. */
	private deviceReport: DeviceReport | null = null;

	constructor(app: App, plugin: WhisperTranscribePlugin) {
		super(app, plugin);
		this.plugin = plugin;
	}

	/**
	 * Which models this machine can actually carry. Whether a model fits is the
	 * one thing a user cannot tell from its name or size, and finding out the
	 * hard way costs a failed run, so it is worth asking the hardware directly.
	 */
	private renderCompatibility(containerEl: HTMLElement) {
		const setting = new Setting(containerEl)
			.setName("What this device can run")
			.setDesc(
				"Checks the GPU and reports which models fit. WebGPU never reveals how much video memory a card has, so this measures what the driver is willing to hand out instead. Treat it as an estimate: on laptops and integrated graphics, where video memory is shared with the system, a driver may promise more than it can comfortably deliver."
			);

		const results = containerEl.createEl("div", {
			attr: { style: "margin:-6px 0 18px 0;font-size:var(--font-ui-smaller)" },
		});

		const render = () => {
			results.empty();
			const report = this.deviceReport;
			if (!report) return;

			const summary = report.hasWebGpu
				? `GPU: ${report.description ?? "unnamed adapter"}${
						report.usableGpuBytes
							? ` — offered ${Math.round(
									report.usableGpuBytes / (1024 * 1024)
								)} MB`
							: ""
					}`
				: report.error
					? `No GPU acceleration (${report.error}) — models run on the CPU`
					: "No GPU acceleration available — models run on the CPU, slowly";

			results.createEl("div", {
				text: summary,
				attr: { style: "opacity:.8;margin-bottom:6px" },
			});

			for (const model of MODELS) {
				const { verdict, reason } = assessModel(model, report);
				const mark = verdict === "ok" ? "✓" : verdict === "tight" ? "!" : "✕";
				const colour =
					verdict === "ok"
						? "var(--text-success)"
						: verdict === "tight"
							? "var(--text-warning)"
							: "var(--text-error)";
				const row = results.createEl("div", {
					attr: { style: "display:flex;gap:8px;padding:1px 0" },
				});
				row.createEl("span", {
					text: mark,
					attr: { style: `color:${colour};width:1em;font-weight:600` },
				});
				row.createEl("span", { text: model.label.split(" — ")[0] });
				row.createEl("span", {
					text: reason,
					attr: { style: "opacity:.6" },
				});
			}
		};

		setting.addButton((button) =>
			button.setButtonText("Check").onClick(async () => {
				button.setButtonText("Checking…").setDisabled(true);
				try {
					this.deviceReport = await probeDevice();
					render();
				} finally {
					button.setButtonText("Check again").setDisabled(false);
				}
			})
		);

		render();
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
				"Larger models are more accurate but slower. The weights are downloaded once and then kept, so the size matters mostly the first time; the amount of memory they need while running matters every time. Sizes shown are for the GPU path — without a GPU the plugin loads smaller, more compressed weights. On a phone, prefer one of the first two."
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
						// Drop the loaded model, otherwise the old one keeps serving
						// until Obsidian restarts.
						this.plugin.resetModel();
						new Notice(
							"Model changed. It is downloaded on the next transcription."
						);
					});
			});

		this.renderCompatibility(containerEl);

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
			.setClass("whisper-transcribe-note-header")
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
