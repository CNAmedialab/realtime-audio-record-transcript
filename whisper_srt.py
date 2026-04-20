import logging
import math
import os
import re
from glob import glob

from openai import OpenAI
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def _strip_hallucination(text: str, max_repeat: int = 3) -> tuple[str, bool]:
    """Deduplicate repeated sentences from Whisper hallucination loops.

    Args:
        text: Raw transcript text.
        max_repeat: Max times a sentence may appear before being dropped.

    Returns:
        (cleaned_text, was_truncated)
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    seen: dict[str, int] = {}
    result = []
    for s in sentences:
        key = s.strip().lower()
        count = seen.get(key, 0) + 1
        seen[key] = count
        if count <= max_repeat:
            result.append(s)
    truncated = len(result) < len(sentences)
    return " ".join(result), truncated


def _is_hallucinating(text: str, min_unique_ratio: float = 0.4) -> bool:
    """Return True if repetition ratio is too high to be real speech.

    Args:
        text: Transcript text (after _strip_hallucination).
        min_unique_ratio: Below this ratio the segment is considered garbage.
    """
    sentences = [
        s.strip().lower()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]
    if len(sentences) < 5:
        return False
    return len(set(sentences)) / len(sentences) < min_unique_ratio

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0)


def parse_time(time_str: str):
    hours, minutes, seconds, milliseconds = map(float, re.split("[:|,]", time_str))
    from datetime import timedelta
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)


def extract_speech_text(srt_string: str) -> str:
    lines = srt_string.split("\n")
    return " ".join(
        line for line in lines
        if line.strip() and "-->" not in line and not re.search(r"^\d+$", line)
    )


def _check_audio_quality(audio: AudioSegment, min_dbfs: float) -> tuple[bool, str]:
    """
    Check audio quality gates.

    Returns:
        (ok, reason): ok=True means audio is acceptable for transcription.
    """
    if audio.dBFS == float("-inf"):
        return False, "completely silent (dBFS=-inf)"

    if audio.dBFS < min_dbfs:
        return False, f"too quiet (dBFS={audio.dBFS:.1f} < threshold {min_dbfs})"

    return True, ""


def audio_to_text(
    audio_file_path: str,
    language: str = "en",
    prompt: str = "",
    temperature: float = 0.1,
    min_dbfs: float = -50.0,
) -> str | None:
    """
    Transcribe an audio file using OpenAI Whisper.

    Applies quality gates before sending to API. Returns None if the file
    should be skipped (silent, too quiet) and removes the file in that case.

    Args:
        audio_file_path: Path to the MP3 file.
        language: BCP-47 language code.
        prompt: Whisper prompt to guide transcription.
        temperature: Sampling temperature (0.0 = greedy).
        min_dbfs: Files below this dBFS level are skipped.

    Returns:
        Transcribed text, or None if skipped.
    """
    audio = AudioSegment.from_file(audio_file_path)

    ok, reason = _check_audio_quality(audio, min_dbfs)
    if not ok:
        logger.warning("Skipping %s: %s", audio_file_path, reason)
        os.remove(audio_file_path)
        return None

    logger.debug(
        "Audio quality OK: %s (dBFS=%.1f, duration=%.1fs)",
        audio_file_path,
        audio.dBFS,
        audio.duration_seconds,
    )

    interval_ms = 60 * 1000
    num_splits = math.ceil(audio.duration_seconds / (interval_ms / 1000))
    combined_result = ""

    for i in range(num_splits):
        split_audio = audio[i * interval_ms : min((i + 1) * interval_ms, len(audio))]
        tmp_path = f"split_audio_temp/{i + 1}.mp3"
        split_audio.export(tmp_path, format="mp3")

        with open(tmp_path, "rb") as f:
            logger.info("Transcribing chunk %d/%d of %s", i + 1, num_splits, audio_file_path)
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                prompt=prompt,
                response_format="srt",
                temperature=temperature,
            )

        text = re.sub(r"\n\n", "\n", transcript).strip()
        combined_result += extract_speech_text(text) + "\n"

    for f in glob("split_audio_temp/*.mp3"):
        os.remove(f)

    clean, was_truncated = _strip_hallucination(combined_result.strip())
    if was_truncated:
        logger.warning("Hallucination stripped (repeated sentences removed): %s", audio_file_path)

    if _is_hallucinating(clean):
        logger.warning(
            "Hallucination detected (unique ratio too low), skipping: %s", audio_file_path
        )
        os.remove(audio_file_path)
        return None

    return clean
