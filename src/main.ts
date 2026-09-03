import {
	App,
	Notice,
	Plugin,
	PluginSettingTab,
	Setting,
	TFile,
	TFolder,
	normalizePath,
} from "obsidian";

const AUDIO_EXTENSIONS = ["ogg", "oga", "m4a", "mp3", "wav", "webm", "opus"];

interface WhisperTranscribeSettings {
	watchFolder: string;
	outputFolder: string;
	modelId: string;
	language: string;
	autoTranscribe: boolean;
}

const DEFAULT_SETTINGS: WhisperTranscribeSettings = {
	watchFolder: "Телеграм аудио",
	outputFolder: "inbox",
	modelId: "Xenova/whisper-base",
	language: "russian",
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

		this.registerEvent(
			this.app.vault.on("create", (file) => {
				if (!this.settings.autoTranscribe) return;
				if (!(file instanceof TFile)) return;
				if (!this.isInWatchFolder(file)) return;
				if (!this.isAudioFile(file)) return;
				this.enqueue(file);
			})
		);

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

		this.statusEl.setText(`Whisper: расшифровываю ${file.name}…`);
		const notice = new Notice(
			`Расшифровываю «${file.name}»…`,
			0
		);

		try {
			const audio = await this.decodeToPcm16k(file);
			const pipeline = await this.getPipeline();
			const result = await pipeline(audio, {
				language: this.settings.language,
				task: "transcribe",
				chunk_length_s: 30,
				stride_length_s: 5,
			});

			const text = result.text.trim();
			const content = `[Источник: голосовое «${file.name}»]\n\n${text}\n`;

			await this.ensureFolder(this.settings.outputFolder);
			await this.app.vault.create(outPath, content);

			notice.setMessage(`Готово: ${file.name}`);
		} catch (err) {
			console.error("whisper-transcribe: failed on", file.path, err);
			notice.setMessage(`Ошибка расшифровки «${file.name}» — см. консоль`);
		} finally {
			window.setTimeout(() => notice.hide(), 4000);
		}
	}

	private buildOutputPath(file: TFile): string {
		const { date, title } = parseVoiceMemoName(file.basename);
		const safeTitle = title.replace(/[\\/:*?"<>|]/g, "_");
		const outFolder = normalizePath(this.settings.outputFolder);
		return normalizePath(`${outFolder}/${date} ${safeTitle}.md`);
	}

	private async ensureFolder(path: string) {
		const normalized = normalizePath(path);
		const existing = this.app.vault.getAbstractFileByPath(normalized);
		if (existing instanceof TFolder) return;
		await this.app.vault.createFolder(normalized).catch(() => {
			/* already exists — race with another create, ignore */
		});
	}

	private async getPipeline(): Promise<AsrPipeline> {
		if (!this.asrPipelinePromise) {
			this.asrPipelinePromise = this.loadPipeline();
		}
		return this.asrPipelinePromise;
	}

	private async loadPipeline(): Promise<AsrPipeline> {
		const { pipeline, env } = await import("@huggingface/transformers");
		env.allowLocalModels = false;
		env.useBrowserCache = true;

		const hasWebGpu = "gpu" in navigator;
		const device = hasWebGpu ? "webgpu" : "wasm";

		this.statusEl.setText("Whisper: загружаю модель…");
		const asr = await pipeline(
			"automatic-speech-recognition",
			this.settings.modelId,
			{
				device,
				progress_callback: (p: { status: string; progress?: number }) => {
					if (p.status === "progress" && typeof p.progress === "number") {
						this.statusEl.setText(
							`Whisper: загрузка модели ${Math.round(p.progress)}%`
						);
					}
				},
			}
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

/** Mirrors the naming convention used by the one-off Python batch transcription:
 *  "Title DD-MM-YYYY_HH-MM-SS.ext" -> { date: "YYYY-MM-DD", title: "Title" } */
function parseVoiceMemoName(basename: string): { date: string; title: string } {
	const match = basename.match(
		/@?(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})\s*$/
	);
	if (!match) {
		return { date: "0000-00-00", title: basename.trim() };
	}
	const [, dd, mm, yyyy] = match;
	const title = basename.slice(0, match.index).replace(/@\s*$/, "").trim();
	return { date: `${yyyy}-${mm}-${dd}`, title: title || "Голосовое" };
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
			.setName("Автоматически транскрибировать")
			.setDesc(
				"Запускать расшифровку сразу, как только в отслеживаемую папку попадает новый аудиофайл."
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
			.setName("Отслеживаемая папка")
			.setDesc("Новые аудиофайлы в этой папке (и подпапках) запускают расшифровку.")
			.addText((text) =>
				text
					.setPlaceholder("Телеграм аудио")
					.setValue(this.plugin.settings.watchFolder)
					.onChange(async (value) => {
						this.plugin.settings.watchFolder = value.trim();
						await this.plugin.saveSettings();
					})
			);

		new Setting(containerEl)
			.setName("Папка для результата")
			.setDesc("Куда класть готовые .md-расшифровки.")
			.addText((text) =>
				text
					.setPlaceholder("inbox")
					.setValue(this.plugin.settings.outputFolder)
					.onChange(async (value) => {
						this.plugin.settings.outputFolder = value.trim();
						await this.plugin.saveSettings();
					})
			);

		new Setting(containerEl)
			.setName("Модель")
			.setDesc(
				"Больше — точнее, но медленнее и тяжелее для скачивания/памяти. На телефоне рекомендуется tiny или base."
			)
			.addDropdown((dropdown) =>
				dropdown
					.addOption("Xenova/whisper-tiny", "tiny (~40 МБ, быстрее всего)")
					.addOption("Xenova/whisper-base", "base (~74 МБ, баланс)")
					.addOption("Xenova/whisper-small", "small (~250 МБ, точнее)")
					.setValue(this.plugin.settings.modelId)
					.onChange(async (value) => {
						this.plugin.settings.modelId = value;
						await this.plugin.saveSettings();
					})
			);

		new Setting(containerEl)
			.setName("Язык")
			.setDesc("Язык речи в голосовых (английским словом, как ожидает Whisper).")
			.addText((text) =>
				text
					.setPlaceholder("russian")
					.setValue(this.plugin.settings.language)
					.onChange(async (value) => {
						this.plugin.settings.language = value.trim();
						await this.plugin.saveSettings();
					})
			);
	}
}
