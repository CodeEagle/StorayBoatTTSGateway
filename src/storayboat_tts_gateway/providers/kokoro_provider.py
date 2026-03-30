from __future__ import annotations

import base64
import io
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from mutagen import File as MutagenFile

from ..api_models import (
    AudioFormat,
    ProviderName,
    SpeechRequest,
    SynthesisResult,
    TimingSource,
    VoiceInfo,
    WordTiming,
)
from .base import TTSProvider

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[\u3040-\u30ff]|[\uac00-\ud7af]|\w+|[^\w\s]", re.UNICODE)


class KokoroProvider(TTSProvider):
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("KOKORO_FASTAPI_BASE_URL") or "http://127.0.0.1:8880").rstrip("/")
        self._timeout = timeout or float(os.environ.get("KOKORO_FASTAPI_TIMEOUT", "120"))

    @property
    def supported_formats(self) -> tuple[AudioFormat, ...]:
        return (AudioFormat.MP3, AudioFormat.WAV)

    async def synthesize(
        self,
        request: SpeechRequest,
        on_progress: Callable[[float], Awaitable[None]] | None = None,
    ) -> SynthesisResult:
        voice = request.voice or "af_sarah"
        payload: dict[str, Any] = {
            "model": request.model,
            "input": request.input,
            "voice": voice,
            "speed": request.speed,
            "response_format": request.response_format.value,
            "stream": False,
            "return_download_link": False,
            # Keep input text stable by default, but allow callers to override per request.
            "normalization_options": request.normalization_options or {"normalize": False},
        }
        if request.lang:
            payload["lang_code"] = request.lang

        if on_progress is not None:
            await on_progress(0.15)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/dev/captioned_speech", json=payload)
            response.raise_for_status()
            data = response.json()
        if on_progress is not None:
            await on_progress(0.8)

        audio_base64 = data.get("audio")
        if not isinstance(audio_base64, str) or not audio_base64:
            raise ValueError("Kokoro-FastAPI did not return base64 audio in the 'audio' field.")

        words = self._parse_timestamps(data.get("timestamps"))
        if not words:
            words = self._estimate_fallback_timings(request.input, audio_base64, request.response_format)
        if not words:
            raise ValueError("Kokoro-FastAPI returned no usable word timestamps, and fallback timing estimation failed.")
        if on_progress is not None:
            await on_progress(0.9)

        return SynthesisResult(
            format=request.response_format,
            audio_base64=audio_base64,
            words=words,
            timing_source=TimingSource.WORD_BOUNDARY if data.get("timestamps") else TimingSource.ESTIMATED,
            provider=ProviderName.KOKORO,
            voice=voice,
            model=request.model,
            estimated=not bool(data.get("timestamps")),
        )

    async def list_voices(self) -> list[VoiceInfo]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self._base_url}/v1/audio/voices")
            response.raise_for_status()
            data = response.json()

        voices = data.get("voices", [])
        items: list[VoiceInfo] = []
        for voice in voices:
            if isinstance(voice, str):
                items.append(
                    VoiceInfo(
                        id=voice,
                        name=voice,
                        provider=ProviderName.KOKORO,
                        locale=None,
                        gender=None,
                        tags=["kokoro-fastapi"],
                    )
                )
                continue

            if isinstance(voice, dict):
                voice_id = self._first_str(voice, "id", "voice", "name")
                if not voice_id:
                    continue
                items.append(
                    VoiceInfo(
                        id=voice_id,
                        name=self._first_str(voice, "display_name", "label", "name") or voice_id,
                        provider=ProviderName.KOKORO,
                        locale=self._first_str(voice, "language", "locale"),
                        gender=self._first_str(voice, "gender"),
                        tags=self._extract_tags(voice),
                    )
                )
        return items

    def _parse_timestamps(self, raw: Any) -> list[WordTiming]:
        if not isinstance(raw, list):
            return []

        words: list[WordTiming] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = self._first_str(item, "word", "text", "token")
            if not text:
                continue
            start_ms = self._extract_time_ms(item, "start_ms", "start", "start_time", "begin", "from")
            end_ms = self._extract_time_ms(item, "end_ms", "end", "end_time", "stop", "to")
            if start_ms is None or end_ms is None:
                continue
            words.append(WordTiming(text=text, start_ms=start_ms, end_ms=end_ms))
        return words

    def _extract_time_ms(self, payload: dict[str, Any], *keys: str) -> int | None:
        for key in keys:
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    continue
            if isinstance(value, (int, float)):
                if key.endswith("_ms"):
                    return int(round(float(value)))
                if isinstance(value, float):
                    return int(round(value * 1000))
                return int(value)
        return None

    def _first_str(self, payload: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _extract_tags(self, payload: dict[str, Any]) -> list[str]:
        tags = payload.get("tags")
        if isinstance(tags, list):
            return [str(tag) for tag in tags]
        return ["kokoro-fastapi"]

    def _estimate_fallback_timings(
        self,
        text: str,
        audio_base64: str,
        audio_format: AudioFormat,
    ) -> list[WordTiming]:
        tokens = TOKEN_RE.findall(text)
        if not tokens:
            return []

        duration_ms = self._audio_duration_ms(audio_base64, audio_format)
        if duration_ms <= 0:
            # Final fallback: rough average per token. Enough to avoid hard failure.
            duration_ms = max(400, len(tokens) * 240)

        weights = [max(1, len(token.encode("utf-8"))) for token in tokens]
        total_weight = sum(weights)
        cursor = 0
        items: list[WordTiming] = []

        for index, (token, weight) in enumerate(zip(tokens, weights, strict=False)):
            start_ms = cursor
            if index == len(tokens) - 1:
                end_ms = duration_ms
            else:
                slice_ms = max(1, round(duration_ms * (weight / total_weight)))
                end_ms = min(duration_ms, cursor + slice_ms)
            items.append(WordTiming(text=token, start_ms=start_ms, end_ms=end_ms))
            cursor = end_ms
        return items

    def _audio_duration_ms(self, audio_base64: str, audio_format: AudioFormat) -> int:
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception:
            return 0

        try:
            handle = io.BytesIO(audio_bytes)
            if audio_format == AudioFormat.MP3:
                handle.name = "audio.mp3"
            elif audio_format == AudioFormat.WAV:
                handle.name = "audio.wav"
            parsed = MutagenFile(handle)
            length = getattr(getattr(parsed, "info", None), "length", 0)
            if not length:
                return 0
            return int(round(length * 1000))
        except Exception:
            return 0
