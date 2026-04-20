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


def _get_end_index(doc_id: str, tab_id: str | None) -> int:
    service = _get_docs_service()
    doc = service.documents().get(documentId=doc_id, includeTabsContent=True).execute()
    if tab_id:
        for tab in doc.get("tabs", []):
            if tab["tabProperties"]["tabId"] == tab_id:
                return tab["documentTab"]["body"]["content"][-1]["endIndex"] - 1
        raise ValueError(f"Tab '{tab_id}' not found in document")
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
                translated: str | None = None
                trivial = _trivial_translate(output)
                if trivial is not None:
                    translated = trivial
                    logger.info("[翻譯] %s (trivial)", translated)
                else:
                    try:
                        translated = _call_with_timeout(
                            translate_to_zhtw,
                            output,
                            system_prompt=t_cfg.get("system_prompt", "將以下文字翻譯成繁體中文，只回傳翻譯結果。"),
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

if __name__ == "__main__":
    task_dir = "tasks"
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
