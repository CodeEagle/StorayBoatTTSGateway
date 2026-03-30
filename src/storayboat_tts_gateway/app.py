from __future__ import annotations

import base64
import json
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from .api_models import APICatalog, APIEndpointInfo, ProviderInfo, ProviderName, SpeechRequest, SynthesisResult, VoiceInfo
from .providers.edge_provider import EdgeProvider
from .providers.kokoro_provider import KokoroProvider

app = FastAPI(title="StorayBoat TTS Gateway", version="0.1.0")

providers = {
    ProviderName.EDGE: EdgeProvider(),
    ProviderName.KOKORO: KokoroProvider(),
}


def get_provider(name: ProviderName):
    return providers[name]


def build_api_catalog() -> APICatalog:
    provider_infos = [
        ProviderInfo(
            id=ProviderName.EDGE,
            name="Edge TTS",
            supports_real_word_timing=True,
            supports_estimated_word_timing=False,
            supported_formats=list(get_provider(ProviderName.EDGE).supported_formats),
        ),
        ProviderInfo(
            id=ProviderName.KOKORO,
            name="Kokoro",
            supports_real_word_timing=True,
            supports_estimated_word_timing=False,
            supported_formats=list(get_provider(ProviderName.KOKORO).supported_formats),
        ),
    ]
    endpoints = [
        APIEndpointInfo(method="GET", path="/healthz", summary="Health check", response_type="application/json"),
        APIEndpointInfo(method="GET", path="/v1/providers", summary="List providers and capabilities", response_type="application/json"),
        APIEndpointInfo(method="GET", path="/v1/voices?provider={provider}", summary="List voices for a provider", response_type="application/json", provider_optional=False),
        APIEndpointInfo(method="GET", path="/v1/catalog", summary="List all API endpoints and provider capabilities", response_type="application/json"),
        APIEndpointInfo(method="POST", path="/v1/audio/speech_with_timestamps", summary="Return JSON with audio_base64 and word timings", response_type="application/json", provider_optional=False),
        APIEndpointInfo(method="POST", path="/v1/{provider}/audio/speech_with_timestamps", summary="Provider-scoped JSON synthesis endpoint", response_type="application/json", provider_optional=True),
        APIEndpointInfo(method="POST", path="/v1/audio/speech", summary="Return compatibility JSON with audio_base64 and format", response_type="application/json", provider_optional=False),
        APIEndpointInfo(method="POST", path="/v1/audio/speech_bundle", summary="Return multipart metadata.json plus binary audio", response_type="multipart/mixed", provider_optional=False),
    ]
    return APICatalog(service=app.title, version=app.version, providers=provider_infos, endpoints=endpoints)


def build_multipart_bundle(result: SynthesisResult) -> tuple[bytes, str]:
    boundary = f"storayboat-{uuid4().hex}"
    metadata = result.model_dump(exclude={"audio_base64"})
    metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    audio_bytes = base64.b64decode(result.audio_base64)
    audio_filename = f"audio.{result.format.value}"
    audio_content_type = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
    }.get(result.format.value, "application/octet-stream")

    parts = [
        (
            f"--{boundary}\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            'Content-Disposition: attachment; name="metadata"; filename="metadata.json"\r\n\r\n'
        ).encode("utf-8")
        + metadata_bytes
        + b"\r\n",
        (
            f"--{boundary}\r\n"
            f"Content-Type: {audio_content_type}\r\n"
            f'Content-Disposition: attachment; name="audio"; filename="{audio_filename}"\r\n\r\n'
        ).encode("utf-8")
        + audio_bytes
        + b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), boundary


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/providers", response_model=list[ProviderInfo])
async def list_providers() -> list[ProviderInfo]:
    return build_api_catalog().providers


@app.get("/v1/catalog", response_model=APICatalog)
async def api_catalog() -> APICatalog:
    return build_api_catalog()


@app.get("/v1/voices", response_model=list[VoiceInfo])
async def list_voices(provider: ProviderName) -> list[VoiceInfo]:
    return await get_provider(provider).list_voices()


@app.post("/v1/audio/speech_with_timestamps", response_model=SynthesisResult)
async def speech_with_timestamps(request: SpeechRequest) -> SynthesisResult:
    try:
        return await get_provider(request.provider).synthesize(request)
    except Exception as exc:  # pragma: no cover - translated for API callers
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/{provider}/audio/speech_with_timestamps", response_model=SynthesisResult)
async def provider_speech_with_timestamps(provider: ProviderName, request: SpeechRequest) -> SynthesisResult:
    merged = request.model_copy(update={"provider": provider})
    return await speech_with_timestamps(merged)


@app.post("/v1/audio/speech")
async def speech_passthrough(request: SpeechRequest) -> dict[str, str]:
    result = await speech_with_timestamps(request)
    return {"audio_base64": result.audio_base64, "format": result.format.value}


@app.post("/v1/audio/speech_bundle")
async def speech_bundle(request: SpeechRequest) -> Response:
    result = await speech_with_timestamps(request)
    payload, boundary = build_multipart_bundle(result)
    return Response(
        content=payload,
        media_type=f'multipart/mixed; boundary="{boundary}"',
    )


def main() -> None:
    uvicorn.run(
        "storayboat_tts_gateway.app:app",
        host="0.0.0.0",
        port=5051,
        reload=False,
    )


if __name__ == "__main__":
    main()
