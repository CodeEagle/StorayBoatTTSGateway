from __future__ import annotations

import os
from typing import Any

import httpx

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

    async def synthesize(self, request: SpeechRequest) -> SynthesisResult:
        voice = request.voice or "af_sarah"
        payload: dict[str, Any] = {
            "model": request.model,
            "input": request.input,
            "voice": voice,
            "speed": request.speed,
            "response_format": request.response_format.value,
            "stream": False,
            "return_download_link": False,
            # Keep input text stable so timestamp text aligns better with callers.
            "normalization_options": {"normalize": False},
        }
        if request.lang:
            payload["lang_code"] = request.lang

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/dev/captioned_speech", json=payload)
            response.raise_for_status()
            data = response.json()

        audio_base64 = data.get("audio")
        if not isinstance(audio_base64, str) or not audio_base64:
            raise ValueError("Kokoro-FastAPI did not return base64 audio in the 'audio' field.")

        words = self._parse_timestamps(data.get("timestamps"))
        if not words:
            raise ValueError("Kokoro-FastAPI returned no usable word timestamps.")

        return SynthesisResult(
            format=request.response_format,
            audio_base64=audio_base64,
            words=words,
            timing_source=TimingSource.WORD_BOUNDARY,
            provider=ProviderName.KOKORO,
            voice=voice,
            model=request.model,
            estimated=False,
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
