import logging
import struct

import numpy as np
import pyaudio
from pydub import AudioSegment

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2  # paInt16 = 2 bytes
CHANNELS = 1
FRAMES_PER_BUFFER = 1024  # ~23ms per frame at 44100Hz


def get_device_index() -> int:
    audio = pyaudio.PyAudio()
    print("Available audio devices:")
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        print(f"{i}. {dev['name']} - {'Input' if dev['maxInputChannels'] > 0 else 'Output'}")
    ind = int(input("Select device: "))
    audio.terminate()
    return ind


def _rms(data: bytes) -> float:
    """Compute RMS energy of a raw PCM int16 frame."""
    count = len(data) // SAMPLE_WIDTH
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"<{count}h", data)
    return float(np.sqrt(np.mean(np.square(shorts, dtype=np.float64))))


def record_audio(
    ind: int,
    output_filename: str,
    min_duration: float = 10.0,
    max_duration: float = 45.0,
    silence_ms: float = 600.0,
    silence_rms_threshold: float = 300.0,
) -> None:
    """
    Record audio with VAD-based smart cut.

    Stops recording when silence of at least `silence_ms` is detected after
    `min_duration` seconds have elapsed. Forces a cut at `max_duration` seconds.

    Args:
        ind: PyAudio device index.
        output_filename: Path to write the MP3 file.
        min_duration: Minimum recording duration in seconds before a silence cut is allowed.
        max_duration: Maximum recording duration in seconds before a forced cut.
        silence_ms: Consecutive silence duration (ms) required to trigger a cut.
        silence_rms_threshold: RMS energy below which a frame is considered silent.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=ind,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    frames: list[bytes] = []
    ms_per_frame = FRAMES_PER_BUFFER / SAMPLE_RATE * 1000  # ~23ms
    silence_frames_needed = int(silence_ms / ms_per_frame)
    max_frames = int(max_duration * SAMPLE_RATE / FRAMES_PER_BUFFER)
    min_frames = int(min_duration * SAMPLE_RATE / FRAMES_PER_BUFFER)

    consecutive_silence = 0

    logger.debug(
        "Recording started: min=%.1fs max=%.1fs silence_threshold=%.0f rms_threshold=%.0f",
        min_duration,
        max_duration,
        silence_ms,
        silence_rms_threshold,
    )

    for i in range(max_frames):
        try:
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        except OSError as e:
            logger.warning("Stream read error (frame %d): %s", i, e)
            continue

        frames.append(data)
        rms = _rms(data)

        if rms < silence_rms_threshold:
            consecutive_silence += 1
        else:
            consecutive_silence = 0

        if i >= min_frames and consecutive_silence >= silence_frames_needed:
            logger.debug(
                "VAD silence cut at %.1fs (%.0f consecutive silence frames)",
                i * ms_per_frame / 1000,
                consecutive_silence,
            )
            break
    else:
        logger.debug("Max duration reached, forced cut at %.1fs", max_duration)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    sound = AudioSegment(
        data=b"".join(frames),
        sample_width=SAMPLE_WIDTH,
        frame_rate=SAMPLE_RATE,
        channels=CHANNELS,
    )
    sound.export(output_filename, format="mp3")
    logger.info("Saved: %s (%.1fs)", output_filename, len(frames) * ms_per_frame / 1000)
