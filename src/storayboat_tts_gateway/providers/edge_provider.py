from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable
from typing import Any

import edge_tts
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

EDGE_VOICES_CATALOG_URL = "https://gist.githubusercontent.com/BettyJJ/17cbaa1de96235a7f5773b8690a20462/raw/05f2ff16fdf9fcb5920635d64eb83c3fa91a2427/list%2520of%2520voices%2520available%2520in%2520Edge%2520TTS.txt"

OPENAI_VOICE_ALIASES: dict[str, str] = {
    "alloy": "en-US-AvaNeural",
    "echo": "en-US-AndrewNeural",
    "fable": "en-GB-SoniaNeural",
    "nova": "en-US-AriaNeural",
    "onyx": "en-US-ChristopherNeural",
    "shimmer": "en-US-AnaNeural",
}


class EdgeProvider(TTSProvider):
    @property
    def supported_formats(self) -> tuple[AudioFormat, ...]:
        return (AudioFormat.MP3,)

    async def synthesize(
        self,
        request: SpeechRequest,
        on_progress: Callable[[float], Awaitable[None]] | None = None,
    ) -> SynthesisResult:
        if request.response_format != AudioFormat.MP3:
            raise ValueError("Edge provider currently supports mp3 only.")

        voice = self._resolve_voice(request.voice)
        rate = self._speed_to_rate(request.speed)
        communicator = edge_tts.Communicate(
            text=request.input,
            voice=voice,
            rate=rate,
            boundary="WordBoundary",
        )

        audio_chunks: list[bytes] = []
        words: list[WordTiming] = []
        text_units = max(len(request.input.strip()), 1)
        delivered_progress = 0.1
        if on_progress is not None:
            await on_progress(delivered_progress)
        async for chunk in communicator.stream():
            chunk_type = chunk.get("type")
            if chunk_type == "audio":
                audio_chunks.append(chunk["data"])
            elif chunk_type == "WordBoundary":
                timing = self._word_timing_from_chunk(chunk)
                if timing is not None:
                    words.append(timing)
                    if on_progress is not None:
                        consumed_units = sum(max(len(item.text.strip()), 1) for item in words)
                        synthesized_fraction = min(consumed_units / text_units, 1.0)
                        next_progress = min(0.85, 0.1 + synthesized_fraction * 0.7)
                        if next_progress > delivered_progress:
                            delivered_progress = next_progress
                            await on_progress(delivered_progress)

        audio_bytes = b"".join(audio_chunks)
        if on_progress is not None:
            await on_progress(0.9)
        return SynthesisResult(
            format=AudioFormat.MP3,
            audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
            words=words,
            timing_source=TimingSource.WORD_BOUNDARY,
            provider=ProviderName.EDGE,
            voice=voice,
            model=request.model,
            estimated=False,
        )

    async def list_voices(self) -> list[VoiceInfo]:
        catalog = await self._load_catalog()
        return sorted(catalog, key=lambda item: (item.locale or "", item.name, item.id))

    def _resolve_voice(self, requested: str | None) -> str:
        if not requested:
            return OPENAI_VOICE_ALIASES["alloy"]
        return OPENAI_VOICE_ALIASES.get(requested, requested)

    def _speed_to_rate(self, speed: float) -> str:
        delta = round((speed - 1.0) * 100)
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta}%"

    def _word_timing_from_chunk(self, chunk: dict[str, Any]) -> WordTiming | None:
        text = chunk.get("text")
        offset = chunk.get("offset")
        duration = chunk.get("duration")
        if not text or offset is None or duration is None:
            return None

        start_ms = int(offset / 10_000)
        end_ms = int((offset + duration) / 10_000)
        return WordTiming(text=text, start_ms=start_ms, end_ms=end_ms)

    async def _load_catalog(self) -> list[VoiceInfo]:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(EDGE_VOICES_CATALOG_URL)
                response.raise_for_status()
                voices = parse_edge_voices_catalog(response.text)
                if voices:
                    return voices
        except Exception:
            pass

        upstream = await edge_tts.list_voices()
        return [
            VoiceInfo(
                id=voice["ShortName"],
                name=edge_voice_display_name(raw_name=voice.get("FriendlyName"), fallback=voice["ShortName"]),
                provider=ProviderName.EDGE,
                locale=voice.get("Locale"),
                gender=voice.get("Gender"),
                language_name=edge_language_name(voice.get("Locale")),
                country=edge_country_code(voice.get("Locale")),
                tags=_flatten_voice_tags(voice),
            )
            for voice in upstream
        ]


def parse_edge_voices_catalog(text: str) -> list[VoiceInfo]:
    if not text:
        return []

    normalized = text.replace("\r\n", "\n")
    blocks = normalized.split("\n\n")
    voices: list[VoiceInfo] = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        name: str | None = None
        short_name: str | None = None
        gender: str | None = None
        voice_locale: str | None = None

        for line in lines:
            if line.startswith("Name:"):
                name = line[5:].strip()
            elif line.startswith("ShortName:"):
                short_name = line[10:].strip()
            elif line.startswith("Gender:"):
                gender = line[7:].strip()
            elif line.startswith("Locale:"):
                voice_locale = line[7:].strip()

        if not short_name:
            continue

        voices.append(
            VoiceInfo(
                id=short_name,
                name=edge_voice_display_name(raw_name=name, fallback=short_name),
                provider=ProviderName.EDGE,
                locale=voice_locale,
                gender=gender,
                language_name=edge_language_name(voice_locale),
                country=edge_country_code(voice_locale),
                tags=[],
            )
        )

    deduped: dict[str, VoiceInfo] = {}
    for voice in voices:
        deduped[voice.id] = voice
    return list(deduped.values())


def edge_voice_display_name(raw_name: str | None = None, fallback: str = "") -> str:
    candidate: str
    if raw_name and "(" in raw_name and ")" in raw_name:
        open_index = raw_name.rfind("(")
        close_index = raw_name.rfind(")")
        if open_index < close_index:
            inner = raw_name[open_index + 1 : close_index]
            parts = [part.strip() for part in inner.split(",")]
            candidate = next((part for part in reversed(parts) if part), raw_name)
        else:
            candidate = raw_name
    else:
        candidate = raw_name or fallback

    cleaned = (
        candidate.replace("MultilingualNeural", "")
        .replace("Neural", "")
        .replace("Multilingual", "")
        .strip()
    )
    if cleaned:
        return cleaned

    fallback_cleaned = (
        (fallback.split("-")[-1] if fallback else "")
        .replace("MultilingualNeural", "")
        .replace("Neural", "")
        .replace("Multilingual", "")
        .strip()
    )
    return fallback_cleaned or fallback


def edge_language_name(voice_locale: str | None) -> str | None:
    if not voice_locale:
        return None
    identifier = voice_locale.replace("_", "-")
    normalized = identifier.split("-")
    if normalized:
        try:
            import babel
            return babel.Locale.parse(identifier.replace("-", "_")).get_display_name("en")
        except Exception:
            pass
        try:
            import pycountry
            language = pycountry.languages.get(alpha_2=normalized[0])
            if language and getattr(language, "name", None):
                return language.name
        except Exception:
            pass
    return identifier


def edge_country_code(voice_locale: str | None) -> str | None:
    if not voice_locale:
        return None
    parts = voice_locale.replace("_", "-").split("-")
    if len(parts) >= 2:
        return parts[1].upper()
    return None


def _flatten_voice_tags(voice: dict[str, Any]) -> list[str]:
    tag_groups = [voice.get("VoiceTag", {}).get("ContentCategories", [])]
    return [tag for group in tag_groups for tag in group if isinstance(tag, str)]
