from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ProviderName(str, Enum):
    EDGE = "edge"
    KOKORO = "kokoro"


class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"


class TimingSource(str, Enum):
    WORD_BOUNDARY = "word_boundary"
    ESTIMATED = "estimated"


class SpeechRequest(BaseModel):
    provider: ProviderName
    model: str = "tts-1"
    input: str = Field(min_length=1)
    voice: str | None = None
    response_format: AudioFormat = AudioFormat.MP3
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    lang: str | None = None
    stream: bool | None = None
    normalization_options: dict[str, bool] | None = None


class WordTiming(BaseModel):
    text: str
    start_ms: int
    end_ms: int


class VoiceInfo(BaseModel):
    id: str
    name: str
    provider: ProviderName
    locale: str | None = None
    gender: str | None = None
    language_name: str | None = None
    country: str | None = None
    alias_of: str | None = None
    tags: list[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    format: AudioFormat
    audio_base64: str
    words: list[WordTiming]
    timing_source: TimingSource
    provider: ProviderName
    voice: str
    model: str
    estimated: bool = False


class ProviderInfo(BaseModel):
    id: ProviderName
    name: str
    default_model: str
    default_voice: str
    supports_real_word_timing: bool
    supports_estimated_word_timing: bool
    supported_formats: list[AudioFormat]
    supported_response_modes: list[str]
    voice_list_path: str
    synthesize_paths: list[str]
    accepted_voice_aliases: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class APIEndpointInfo(BaseModel):
    method: str
    path: str
    summary: str
    response_type: str
    provider_optional: bool = False


class APICatalog(BaseModel):
    service: str
    version: str
    providers: list[ProviderInfo]
    endpoints: list[APIEndpointInfo]


ProviderRoute = Literal["edge", "kokoro"]
