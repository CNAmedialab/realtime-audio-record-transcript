from dotenv import load_dotenv
load_dotenv()

import logging
import logging.handlers
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from glob import glob
from pathlib import Path

import yaml
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from openai import OpenAI
from pydub.exceptions import CouldntDecodeError

from recording import record_audio, get_device_index
from whisper_srt import audio_to_text

try:
    sys.path.insert(0, "/Users/capo/Dropbox/py/CNA")
    from sheetFunc import get_cell, update_acell
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    log_path = f"logs/app_{datetime.now():%Y%m%d}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(fmt)
    stderr_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)

    return logging.getLogger("app")


logger = _setup_logging()

# ── Directories ───────────────────────────────────────────────────────────────

for _d in ("temp", "split_audio_temp", "failed", "logs"):
    Path(_d).mkdir(exist_ok=True)

# ── Timeout wrapper ───────────────────────────────────────────────────────────

def _call_with_timeout(fn, *args, timeout: float = 30, **kwargs):
    """
    Run fn(*args, **kwargs) in a thread and return its result.
    Raises concurrent.futures.TimeoutError if it exceeds `timeout` seconds.
    The OpenAI client already has its own socket timeout set; this provides
    an outer deadline so the transcript loop is never blocked indefinitely.
    """
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout)


# ── Google Docs ──────────────────────────────────────────────────────────────

_docs_service = None


def _get_docs_service():
    global _docs_service
    if _docs_service is None:
        creds = Credentials.from_service_account_file(
            os.getenv("serviceAccountFile"),
            scopes=["https://www.googleapis.com/auth/documents"],
        )
        _docs_service = build("docs", "v1", credentials=creds)
    return _docs_service


def create_gdoc(title: str, with_translation: bool) -> tuple[str, str, str | None]:
    """Create a new Google Doc, set anyone-with-link-can-edit, optionally add translation tab.

    Returns (doc_id, transcript_tab_id, translation_tab_id).
    """
    sa_file = os.getenv("serviceAccountFile")
    if not sa_file:
        raise RuntimeError("serviceAccountFile env var not set.")

    creds = Credentials.from_service_account_file(
        sa_file,
        scopes=[
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    drive = build("drive", "v3", credentials=creds)
    docs = build("docs", "v1", credentials=creds)

    # Create the document
    doc_file = drive.files().create(
        body={"name": f"Transcript: {title}", "mimeType": "application/vnd.google-apps.document"},
        fields="id",
    ).execute()
    doc_id = doc_file["id"]

    # Anyone with link can edit
    drive.permissions().create(
        fileId=doc_id,
        body={"type": "anyone", "role": "writer"},
    ).execute()

    # Get default tab ID
    doc = docs.documents().get(documentId=doc_id, includeTabsContent=True).execute()
    tabs = doc.get("tabs", [])
    transcript_tab_id: str = tabs[0]["tabProperties"]["tabId"] if tabs else "t.0"

    translation_tab_id: str | None = None
    if with_translation:
        try:
            result = docs.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": [{"createTab": {"tabProperties": {"title": "譯文"}}}]},
            ).execute()
            replies = result.get("replies", [])
            if replies and "createTab" in replies[0]:
                translation_tab_id = replies[0]["createTab"]["tabProperties"]["tabId"]
        except Exception:
            # createTab not supported by this API version; translation written to same tab
            logger.warning("createTab not supported; translation will share the transcript tab")

    logger.info("Created GDoc: %s (transcript=%s, translation=%s)", doc_id, transcript_tab_id, translation_tab_id)
    return doc_id, transcript_tab_id, translation_tab_id


def _get_end_index(doc_id: str, tab_id: str | None) -> int:
    service = _get_docs_service()
    doc = service.documents().get(documentId=doc_id, includeTabsContent=True).execute()
    tabs = doc.get("tabs", [])
    if tab_id:
        for tab in tabs:
            if tab["tabProperties"]["tabId"] == tab_id:
                return tab["documentTab"]["body"]["content"][-1]["endIndex"] - 1
        raise ValueError(f"Tab '{tab_id}' not found in document")
    # Tabbed documents have no root body; fall back to first tab
    if tabs:
        return tabs[0]["documentTab"]["body"]["content"][-1]["endIndex"] - 1
    return doc["body"]["content"][-1]["endIndex"] - 1


def append_to_gdoc(doc_id: str, text: str, tab_id: str | None = None) -> None:
    service = _get_docs_service()
    end_index = _get_end_index(doc_id, tab_id)
    location = {"index": end_index}
    if tab_id:
        location["tabId"] = tab_id
    service.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"location": location, "text": text + "\n"}}]},
    ).execute()


# ── Translation ───────────────────────────────────────────────────────────────

_TRIVIAL_MAP: dict[str, str] = {
    "thank you.": "謝謝。",
    "thank you": "謝謝。",
    "thanks.": "謝謝。",
    "thanks": "謝謝。",
    "applause": "（掌聲）",
    "[applause]": "（掌聲）",
    "(applause)": "（掌聲）",
    "thank you very much.": "非常感謝。",
    "thank you very much": "非常感謝。",
}


def _trivial_translate(text: str) -> str | None:
    """Return a hardcoded translation for very short social phrases.

    Prevents the LLM from misinterpreting audience applause or closing remarks
    as a message directed at itself.

    Args:
        text: Stripped transcript text.

    Returns:
        Pre-defined translation, or None if text should go through the LLM.
    """
    return _TRIVIAL_MAP.get(text.strip().lower())


def translate_to_zhtw(
    text: str,
    system_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=25.0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


_LANG_PROMPTS: dict[str, str] = {
    "zh-TW": (
        "你是專業譯者，擅長繁體中文（台灣用語）。"
        "將下方逐字稿翻譯為自然流暢的繁體中文，保留原意，不加評論，不省略內容。"
        "術語保留原文並加括號附中譯，例如：API（應用程式介面）。"
        "無論輸入多短，只輸出翻譯結果，不加解釋或確認語。"
    ),
    "en-US": (
        "You are a professional translator. "
        "Translate the transcript below into natural American English. "
        "Preserve all meaning. Do not add commentary or omit content. "
        "Output only the translation, no explanations."
    ),
    "ja": (
        "あなたはプロの翻訳者です。"
        "以下の文字起こしを自然な日本語に翻訳してください。"
        "意味をすべて保持し、内容を省略しないでください。"
        "翻訳結果のみを出力し、説明や確認は不要です。"
    ),
}


def translate(
    text: str,
    lang: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> str:
    """Translate text to the given BCP-47 language tag (zh-TW / en-US / ja)."""
    system_prompt = _LANG_PROMPTS.get(lang)
    if system_prompt is None:
        raise ValueError(f"Unsupported language: {lang}. Supported: {list(_LANG_PROMPTS)}")
    return translate_to_zhtw(text, system_prompt=system_prompt, model=model, temperature=temperature)


# ── Task config ───────────────────────────────────────────────────────────────

def load_task(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Record loop ───────────────────────────────────────────────────────────────

_record_cfg: dict = {}
_device_index: int = 0


def record() -> None:
    r_cfg = _record_cfg
    min_dur = r_cfg.get("min_duration", 10.0)
    max_dur = r_cfg.get("max_duration", 45.0)
    silence_ms = r_cfg.get("silence_ms", 600.0)
    silence_rms = r_cfg.get("silence_rms_threshold", 300.0)

    while True:
        start_time = datetime.now()
        out_path = f"temp/output-S{start_time:%Y_%m_%d＼%H_%M_%S}.mp3"
        try:
            record_audio(
                ind=_device_index,
                output_filename=out_path,
                min_duration=min_dur,
                max_duration=max_dur,
                silence_ms=silence_ms,
                silence_rms_threshold=silence_rms,
            )
            logger.info("Recorded → %s", out_path)
        except Exception:
            logger.error("record_audio() failed", exc_info=True)
            time.sleep(3)


# ── Transcript loop ───────────────────────────────────────────────────────────

def transcript(task: dict, sheet_name: str | None = None) -> None:
    w_cfg = task.get("whisper", {})
    t_cfg = task.get("translation", {})
    g_cfg = task.get("gdoc", {})
    s_cfg = task.get("sheets", {})

    doc_id = g_cfg.get("doc_id") or None
    transcript_tab = g_cfg.get("transcript_tab_id") or None
    translation_tab = g_cfg.get("translation_tab_id") or None
    worksheet_id = s_cfg.get("worksheet_id", "")
    min_dbfs = w_cfg.get("min_dbfs", -50.0)
    translate_timeout = t_cfg.get("timeout", 30)
    gdoc_timeout = g_cfg.get("timeout", 15)
    whisper_timeout = w_cfg.get("timeout", 90)

    while True:
        files = sorted(glob("temp/*.mp3"))

        if not files:
            logger.debug("No files in temp/, waiting...")
            time.sleep(5)
            continue

        for file_path in files:
            logger.info("Processing: %s", file_path)

            # ── Step 1: Transcribe ──────────────────────────────────────────
            output: str | None = None
            for attempt in range(3):
                try:
                    output = _call_with_timeout(
                        audio_to_text,
                        file_path,
                        language=w_cfg.get("language", "en"),
                        prompt=w_cfg.get("prompt", ""),
                        temperature=w_cfg.get("temperature", 0.1),
                        min_dbfs=min_dbfs,
                        timeout=whisper_timeout,
                    )
                    break
                except FuturesTimeout:
                    logger.error(
                        "Whisper timeout (attempt %d/3, timeout=%ss): %s",
                        attempt + 1,
                        whisper_timeout,
                        file_path,
                    )
                except CouldntDecodeError as e:
                    logger.error(
                        "Audio decode error (attempt %d/3): %s — %s",
                        attempt + 1,
                        file_path,
                        e,
                    )
                except Exception:
                    logger.error(
                        "Whisper unexpected error (attempt %d/3): %s",
                        attempt + 1,
                        file_path,
                        exc_info=True,
                    )

                if attempt < 2:
                    time.sleep(2)
                else:
                    dest = shutil.move(file_path, "failed/")
                    logger.error("Giving up on %s → moved to %s", file_path, dest)

            if output is None:
                continue

            output = output.strip()
            timestamp_match = re.search(r"-S[\d_＼]+-?", file_path)
            timestamp = timestamp_match.group(0)[2:].rstrip("-") if timestamp_match else file_path
            logger.info("%s\t%s", timestamp, output)

            # ── Step 2: Google Sheets ───────────────────────────────────────
            if SHEETS_AVAILABLE and sheet_name and worksheet_id:
                try:
                    update_to = _call_with_timeout(
                        get_cell,
                        os.getenv("serviceAccountFile"),
                        worksheet_id,
                        sheet_name,
                        "D1",
                        timeout=10,
                    )
                    _call_with_timeout(
                        update_acell,
                        os.getenv("serviceAccountFile"),
                        worksheet_id,
                        sheet_name,
                        f"A{update_to}",
                        timestamp,
                        timeout=10,
                    )
                    _call_with_timeout(
                        update_acell,
                        os.getenv("serviceAccountFile"),
                        worksheet_id,
                        sheet_name,
                        f"C{update_to}",
                        output,
                        timeout=10,
                    )
                except FuturesTimeout:
                    logger.warning("Google Sheets timeout, skipping Sheets update")
                except Exception:
                    logger.error("Google Sheets update failed", exc_info=True)

            # ── Step 3: Write original to GDoc ──────────────────────────────
            if doc_id:
                try:
                    _call_with_timeout(
                        append_to_gdoc,
                        doc_id,
                        output,
                        tab_id=transcript_tab,
                        timeout=gdoc_timeout,
                    )
                except FuturesTimeout:
                    logger.error(
                        "GDoc transcript write timeout (%ss): %s",
                        gdoc_timeout,
                        file_path,
                    )
                except Exception:
                    logger.error("GDoc transcript write failed: %s", file_path, exc_info=True)

                # ── Step 4: Translate ───────────────────────────────────────
                _sys_prompt = t_cfg.get("system_prompt", "").strip()
                translated: str | None = None
                if _sys_prompt:
                    trivial = _trivial_translate(output)
                    if trivial is not None:
                        translated = trivial
                        logger.info("[翻譯] %s (trivial)", translated)
                    else:
                        try:
                            translated = _call_with_timeout(
                                translate_to_zhtw,
                                output,
                                system_prompt=_sys_prompt,
                                model=t_cfg.get("model", "gpt-4o-mini"),
                                temperature=t_cfg.get("temperature", 0.3),
                                timeout=translate_timeout,
                            )
                            logger.info("[翻譯] %s", translated)
                        except FuturesTimeout:
                            logger.warning(
                                "Translation timeout (%ss), skipping translation for: %s",
                                translate_timeout,
                                file_path,
                            )
                        except Exception:
                            logger.warning(
                                "Translation failed, skipping: %s",
                                file_path,
                                exc_info=True,
                            )

                # ── Step 5: Write translation to GDoc ──────────────────────
                if translated:
                    try:
                        _call_with_timeout(
                            append_to_gdoc,
                            doc_id,
                            translated,
                            tab_id=translation_tab,
                            timeout=gdoc_timeout,
                        )
                    except FuturesTimeout:
                        logger.error(
                            "GDoc translation write timeout (%ss): %s",
                            gdoc_timeout,
                            file_path,
                        )
                    except Exception:
                        logger.error(
                            "GDoc translation write failed: %s",
                            file_path,
                            exc_info=True,
                        )

            # ── Cleanup ─────────────────────────────────────────────────────
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
            time.sleep(1)


# ── Watchdog ──────────────────────────────────────────────────────────────────

def _watchdog(threads: dict[str, threading.Thread], builders: dict[str, callable]) -> None:
    """
    Monitor worker threads every 10s and restart any that have died.
    `builders` maps thread name → zero-argument callable that returns a new Thread.
    """
    while True:
        time.sleep(10)
        for name, thread in list(threads.items()):
            if not thread.is_alive():
                logger.error("Thread '%s' died — restarting", name)
                new_thread = builders[name]()
                new_thread.start()
                threads[name] = new_thread


# ── Entry point ───────────────────────────────────────────────────────────────

def _resolve_device(hint: str) -> int:
    """Find input device index by name pattern (case-insensitive) or integer."""
    import pyaudio as _pyaudio
    pa = _pyaudio.PyAudio()
    try:
        try:
            idx = int(hint)
            info = pa.get_device_info_by_index(idx)
            if info["maxInputChannels"] < 1:
                raise RuntimeError(f"Device {idx} ({info['name']}) has no input channels.")
            logger.info("Using device %d: %s", idx, info["name"])
            return idx
        except ValueError:
            pass
        pattern = hint.lower()
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if pattern in dev["name"].lower() and dev["maxInputChannels"] > 0:
                logger.info("Auto-selected device %d: %s", i, dev["name"])
                return i
        raise RuntimeError(f"No input device matching '{hint}' found.")
    finally:
        pa.terminate()


def youtube_cmd(url: str, lang: str | None, src_lang: str) -> None:
    """One-shot: download YouTube audio → Whisper → optional translate → stdout."""
    import subprocess
    import tempfile

    for _d in ("temp", "split_audio_temp"):
        Path(_d).mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="yt_dl_") as tmpdir:
        out_template = str(Path(tmpdir) / "audio.%(ext)s")
        print(f"[1/3] Downloading: {url}", flush=True)
        result = subprocess.run(
            [
                "yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "0",
                "-o", out_template, "--no-playlist", "--quiet", "--no-warnings", url,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"ERROR yt-dlp: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)

        mp3_files = sorted(Path(tmpdir).glob("*.mp3"))
        if not mp3_files:
            all_files = sorted(Path(tmpdir).iterdir())
            if not all_files:
                print("ERROR: yt-dlp produced no output.", file=sys.stderr)
                sys.exit(1)
            audio_path = all_files[0]
        else:
            audio_path = mp3_files[0]

        print(f"[2/3] Transcribing ({src_lang})...", flush=True)
        whisper_lang = None if src_lang == "auto" else src_lang
        transcript_text = audio_to_text(
            str(audio_path),
            language=whisper_lang or "en",
            temperature=0.0,
        )

    if transcript_text is None:
        print("ERROR: Transcription failed (silent or hallucination detected).", file=sys.stderr)
        sys.exit(1)

    if lang:
        print(f"[3/3] Translating to {lang}...", flush=True)
        translation = translate(transcript_text, lang)
        sep = "─" * 60
        lang_labels = {"zh-TW": "繁體中文譯文", "en-US": "English Translation", "ja": "日本語訳"}
        print(f"\n=== 原文逐字稿 ===\n{sep}")
        print(transcript_text)
        print(f"\n=== {lang_labels[lang]} ===\n{sep}")
        print(translation)
    else:
        print(transcript_text)


if __name__ == "__main__":
    import argparse as _argparse
    _ap = _argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--youtube", metavar="URL", default=None)
    _ap.add_argument("--lang", choices=["zh-TW", "en-US", "ja"], default=None)
    _ap.add_argument("--src-lang", default="auto", dest="src_lang")
    _ap.add_argument("--task", metavar="PATH", default=None)
    _ap.add_argument("--device", metavar="NAME_OR_IDX", default=None)
    _ap.add_argument("--setup-gdoc", action="store_true", dest="setup_gdoc")
    _known, _rest = _ap.parse_known_args()

    if _known.youtube:
        youtube_cmd(_known.youtube, _known.lang, _known.src_lang)
        sys.exit(0)

    if _known.setup_gdoc:
        if not _known.task:
            print("ERROR: --setup-gdoc requires --task <path>", file=sys.stderr)
            sys.exit(1)
        _task = load_task(_known.task)
        _g = _task.get("gdoc", {})
        if _g.get("doc_id"):
            # Already has a doc_id, just print it
            print(f"https://docs.google.com/document/d/{_g['doc_id']}/edit")
            sys.exit(0)
        _with_trans = bool(_task.get("translation", {}).get("system_prompt", "").strip())
        _task_name = _task.get("task_name", Path(_known.task).stem)
        _doc_id, _trans_tab, _tl_tab = create_gdoc(_task_name, _with_trans)
        _task.setdefault("gdoc", {})
        _task["gdoc"]["doc_id"] = _doc_id
        _task["gdoc"]["transcript_tab_id"] = _trans_tab
        _task["gdoc"]["translation_tab_id"] = _tl_tab or ""
        with open(_known.task, "w", encoding="utf-8") as _f:
            yaml.dump(_task, _f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"https://docs.google.com/document/d/{_doc_id}/edit")
        sys.exit(0)

    task_dir = "tasks"

    if _known.task:
        task_path = _known.task
        if not Path(task_path).exists():
            logger.critical("Task file not found: %s", task_path)
            sys.exit(1)
        task = load_task(task_path)
        logger.info("載入任務: %s", task.get("task_name", task_path))
    else:
        task_files = sorted(glob(f"{task_dir}/*.yaml"))
        if not task_files:
            logger.critical("No task files found in tasks/")
            sys.exit(1)

        print("=== 選擇任務 ===")
        for i, f in enumerate(task_files):
            print(f"  {i}: {os.path.basename(f)}")
        choice = input("選擇 (預設 0): ").strip()
        task_path = task_files[int(choice) if choice else 0]
        task = load_task(task_path)
        logger.info("載入任務: %s", task.get("task_name", task_path))

    g_cfg = task.get("gdoc", {})
    doc_id = g_cfg.get("doc_id")
    print("\n=== 輸出設定 ===")
    if doc_id:
        print(f"  Google Doc  : {doc_id}")
        print(f"  原文 tab    : {g_cfg.get('transcript_tab_id', '(default)')}")
        print(f"  翻譯 tab    : {g_cfg.get('translation_tab_id', '(default)')}")
    else:
        print("  Google Doc  : 未設定")

    sheet_name: str | None = None
    if SHEETS_AVAILABLE:
        raw = input("\nGoogle Sheet tab 名稱（Enter 跳過）: ").strip()
        sheet_name = raw or None
    else:
        print("  Google Sheet: 不可用")

    r_cfg = task.get("recording", {})
    _record_cfg.update(r_cfg)
    if _known.device:
        _device_index = _resolve_device(_known.device)
    else:
        _device_index = get_device_index()

    threads: dict[str, threading.Thread] = {}

    def _make_record_thread() -> threading.Thread:
        t = threading.Thread(target=record, name="record", daemon=True)
        return t

    def _make_transcript_thread() -> threading.Thread:
        t = threading.Thread(
            target=transcript, args=(task, sheet_name), name="transcript", daemon=True
        )
        return t

    builders = {
        "record": _make_record_thread,
        "transcript": _make_transcript_thread,
    }

    threads["record"] = _make_record_thread()
    threads["transcript"] = _make_transcript_thread()
    threads["record"].start()
    threads["transcript"].start()

    logger.info("All threads started. Watchdog active.")

    try:
        _watchdog(threads, builders)
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")
        sys.exit(0)
