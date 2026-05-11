#!/usr/bin/env python3
"""
watcher.py — Jarvis 分析層

監聽 app.py log，定時用 AI 分析逐字稿，推送摘要到 Discord 或桌面通知。

用法：
  python watcher.py
  python watcher.py --mode meeting --context "這是產品週會" --discord $WEBHOOK
  python watcher.py --interval 60 --min-segments 2 --notify --engine openai
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Prompt templates ──────────────────────────────────────────────────────────

_MEETING_PROMPT = """\
你是一位會議助理。以下是最近 {n_segments} 段的會議逐字稿。
{context_line}
請完成以下三件事：
1. **摘要**（2-3句）：這段時間內討論了什麼？
2. **待辦／決議**：有無具體行動項目或決定？若無則省略此項。
3. **需查資料**：有無專有名詞、數據、人名等需要背景資料？若無則省略此項。

逐字稿：
{text}

輸出格式（繁體中文）：
**摘要：** ...
**待辦：** ...
**查資料：** ..."""

_GENERAL_PROMPT = """\
以下是一段音訊逐字稿，請用 2-3 句話摘要重點（繁體中文）。

逐字稿：
{text}"""

# ── Analysis engines ──────────────────────────────────────────────────────────

def _analyze_openai(filled_prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": filled_prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def _analyze_claude(filled_prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": filled_prompt}],
    )
    return msg.content[0].text.strip()


def analyze(
    segments: list[tuple[str, str]],
    mode: str,
    context: str,
    engine: str,
) -> str:
    text = "\n".join(f"[{ts}] {txt}" for ts, txt in segments)

    if mode == "meeting":
        context_line = f"會議背景：{context}\n" if context else ""
        prompt = _MEETING_PROMPT.format(
            n_segments=len(segments),
            context_line=context_line,
            text=text,
        )
    else:
        prompt = _GENERAL_PROMPT.format(text=text)

    return _analyze_claude(prompt) if engine == "claude" else _analyze_openai(prompt)


# ── Output sinks ──────────────────────────────────────────────────────────────

def push_discord(webhook_url: str, message: str) -> None:
    import httpx
    try:
        httpx.post(webhook_url, json={"content": message}, timeout=10)
    except Exception as exc:
        print(f"[Jarvis] Discord 發送失敗: {exc}", file=sys.stderr)


def push_notify(message: str) -> None:
    safe = message[:200].replace('"', "'")
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{safe}" with title "Jarvis 摘要"'],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass


# ── Log tailer ────────────────────────────────────────────────────────────────

# Matches lines like:  2024-01-01 10:00:00 [INFO] app: 2024_01_01＼10_00_00\t<text>
_TRANSCRIPT_RE = re.compile(r"\[INFO\] app: ([\w_＼]+)\t(.+)")


class LogTailer:
    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._path: Path | None = None
        self._pos = 0

    def _latest(self) -> Path | None:
        logs = sorted(self._log_dir.glob("app_*.log"))
        return logs[-1] if logs else None

    def read_new(self) -> list[tuple[str, str]]:
        """Return new (timestamp, text) pairs since last call."""
        current = self._latest()
        if current is None:
            return []

        if current != self._path:
            self._path = current
            self._pos = 0

        results: list[tuple[str, str]] = []
        try:
            with open(self._path, encoding="utf-8") as f:
                f.seek(self._pos)
                for line in f:
                    m = _TRANSCRIPT_RE.search(line)
                    if m:
                        results.append((m.group(1), m.group(2).strip()))
                self._pos = f.tell()
        except FileNotFoundError:
            self._pos = 0

        return results


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Jarvis watcher — 即時逐字稿分析")
    ap.add_argument("--log-dir", default="logs", metavar="DIR",
                    help="app.py log 目錄（預設：logs/）")
    ap.add_argument("--interval", type=int, default=90, metavar="SEC",
                    help="最長分析間隔秒數（預設：90）")
    ap.add_argument("--min-segments", type=int, default=3, dest="min_segments",
                    metavar="N", help="最少累積幾段才觸發分析（預設：3）")
    ap.add_argument("--mode", choices=["meeting", "general"], default="meeting",
                    help="分析模式（預設：meeting）")
    ap.add_argument("--context", default="",
                    help="會議背景說明，例如 '這是產品週會'")
    ap.add_argument("--discord", default=os.getenv("DISCORD_WEBHOOK_URL"),
                    metavar="URL", help="Discord Webhook URL（可改用 DISCORD_WEBHOOK_URL env）")
    ap.add_argument("--notify", action="store_true",
                    help="同時發送 macOS 桌面通知")
    ap.add_argument("--engine", choices=["openai", "claude"], default="openai",
                    help="分析引擎（預設：openai）")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"[Jarvis] 找不到 log 目錄：{log_dir}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[Jarvis] 啟動  mode={args.mode}  engine={args.engine}"
        f"  interval={args.interval}s  min_segments={args.min_segments}"
    )
    if args.context:
        print(f"[Jarvis] 會議背景：{args.context}")
    if args.discord:
        print(f"[Jarvis] Discord webhook 已設定")
    print("─" * 50)

    tailer = LogTailer(log_dir)
    buffer: list[tuple[str, str]] = []
    last_analysis = time.time()

    try:
        while True:
            new = tailer.read_new()
            if new:
                buffer.extend(new)
                for ts, txt in new:
                    print(f"  [{ts}] {txt}")

            elapsed = time.time() - last_analysis
            should_analyze = len(buffer) >= args.min_segments or (
                elapsed >= args.interval and buffer
            )

            if should_analyze:
                segments = buffer[:]
                buffer.clear()
                last_analysis = time.time()

                print(f"\n[Jarvis] 分析 {len(segments)} 段...", flush=True)
                try:
                    result = analyze(segments, args.mode, args.context, args.engine)
                except Exception as exc:
                    print(f"[Jarvis] 分析失敗: {exc}", file=sys.stderr)
                    buffer = segments + buffer  # 放回 buffer
                    time.sleep(10)
                    continue

                ts_now = datetime.now().strftime("%H:%M")
                header = f"**[Jarvis {ts_now}]**"
                message = f"{header}\n{result}"

                print(f"\n{message}\n{'─' * 50}")

                if args.discord:
                    push_discord(args.discord, message)
                if args.notify:
                    push_notify(result)

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n[Jarvis] 停止")


if __name__ == "__main__":
    main()
