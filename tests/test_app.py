import base64
import json

from fastapi.testclient import TestClient

from storyboat_tts_gateway.api_models import AudioFormat, ProviderName, SynthesisResult, TimingSource
from storyboat_tts_gateway.app import app, build_api_catalog, build_multipart_bundle, jobs


def _job_request_payload() -> dict[str, object]:
    return {
        "provider": "edge",
        "model": "tts-1",
        "input": "hello world",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0,
    }


def test_build_multipart_bundle_contains_metadata_and_binary_audio() -> None:
    result = SynthesisResult(
        format=AudioFormat.MP3,
        audio_base64=base64.b64encode(b"abc123").decode("ascii"),
        words=[],
        timing_source=TimingSource.WORD_BOUNDARY,
        provider=ProviderName.EDGE,
        voice="alloy",
        model="tts-1",
        estimated=False,
    )

    payload, boundary = build_multipart_bundle(result)
    text = payload.decode("latin-1")

    assert boundary
    assert 'filename="metadata.json"' in text
    assert 'filename="audio.mp3"' in text
    assert "abc123" in text
    assert "audio_base64" not in text

    metadata_start = text.index("\r\n\r\n") + 4
    metadata_end = text.index("\r\n--", metadata_start)
    metadata = json.loads(text[metadata_start:metadata_end])
    assert metadata["format"] == "mp3"
    assert metadata["provider"] == "edge"


def test_build_api_catalog_includes_bundle_and_catalog_endpoints() -> None:
    catalog = build_api_catalog()
    paths = {endpoint.path for endpoint in catalog.endpoints}
    assert "/v1/catalog" in paths
    assert "/v1/audio/jobs" in paths
    assert "/v1/audio/jobs/{id}/events" in paths
    assert "/v1/audio/jobs/{id}/bundle" in paths
    assert "/v1/audio/speech_bundle" in paths
    assert "/v1/audio/speech_base64" in paths
    edge = next(provider for provider in catalog.providers if provider.id == "edge")
    assert edge.default_voice == "alloy"
    assert "multipart_bundle" in edge.supported_response_modes
    assert "job_stream" in edge.supported_response_modes
    assert "alloy" in edge.accepted_voice_aliases


def test_audio_job_endpoints_stream_events_and_bundle(monkeypatch) -> None:
    async def fake_synthesize(request, on_progress=None):
        if on_progress is not None:
            await on_progress(0.4)
            await on_progress(0.8)
        return SynthesisResult(
            format=AudioFormat.MP3,
            audio_base64=base64.b64encode(b"job-audio").decode("ascii"),
            words=[],
            timing_source=TimingSource.WORD_BOUNDARY,
            provider=ProviderName.EDGE,
            voice=request.voice or "alloy",
            model=request.model,
            estimated=False,
        )

    monkeypatch.setattr("storyboat_tts_gateway.app.providers", {"edge": type("StubProvider", (), {"synthesize": staticmethod(fake_synthesize)})(), "kokoro": object()})
    jobs.clear()

    client = TestClient(app)
    create_response = client.post("/v1/audio/jobs", json=_job_request_payload())
    assert create_response.status_code == 200
    job_id = create_response.json()["id"]

    state_response = client.get(f"/v1/audio/jobs/{job_id}")
    assert state_response.status_code == 200
    assert state_response.json()["id"] == job_id

    events_response = client.get(f"/v1/audio/jobs/{job_id}/events")
    assert events_response.status_code == 200
    assert "event: snapshot" in events_response.text
    assert "event: started" in events_response.text
    assert "event: synth_progress" in events_response.text
    assert "event: completed" in events_response.text
    assert f'"/v1/audio/jobs/{job_id}/bundle"' in events_response.text

    bundle_response = client.get(f"/v1/audio/jobs/{job_id}/bundle")
    assert bundle_response.status_code == 200
    assert bundle_response.headers["content-type"].startswith("multipart/mixed; boundary=")
    assert bundle_response.headers["accept-ranges"] == "bytes"
    assert bundle_response.content


def test_provider_scoped_direct_synthesis_accepts_provider_from_path_only(monkeypatch) -> None:
    async def fake_synthesize(request, on_progress=None):
        return SynthesisResult(
            format=AudioFormat.MP3,
            audio_base64=base64.b64encode(b"path-only").decode("ascii"),
            words=[],
            timing_source=TimingSource.WORD_BOUNDARY,
            provider=ProviderName.EDGE,
            voice=request.voice or "alloy",
            model=request.model,
            estimated=False,
        )

    monkeypatch.setattr("storyboat_tts_gateway.app.providers", {"edge": type("StubProvider", (), {"synthesize": staticmethod(fake_synthesize)})(), "kokoro": object()})

    client = TestClient(app)
    response = client.post(
        "/v1/edge/audio/speech_with_timestamps",
        json={
            "model": "tts-1",
            "input": "hello world",
            "voice": "alloy",
            "response_format": "mp3",
            "speed": 1.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "edge"


def test_direct_synthesis_requires_provider_on_unscoped_route() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/audio/speech_with_timestamps",
        json={
            "model": "tts-1",
            "input": "hello world",
            "voice": "alloy",
            "response_format": "mp3",
            "speed": 1.0,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "provider is required"


def test_audio_speech_returns_raw_audio_bytes(monkeypatch) -> None:
    async def fake_synthesize(request, on_progress=None):
        return SynthesisResult(
            format=AudioFormat.MP3,
            audio_base64=base64.b64encode(b"raw-audio").decode("ascii"),
            words=[],
            timing_source=TimingSource.WORD_BOUNDARY,
            provider=ProviderName.EDGE,
            voice=request.voice or "alloy",
            model=request.model,
            estimated=False,
        )

    monkeypatch.setattr("storyboat_tts_gateway.app.providers", {"edge": type("StubProvider", (), {"synthesize": staticmethod(fake_synthesize)})(), "kokoro": object()})

    client = TestClient(app)
    response = client.post("/v1/audio/speech", json=_job_request_payload())

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["x-audio-format"] == "mp3"
    assert response.content == b"raw-audio"


def test_audio_speech_base64_returns_compatibility_json(monkeypatch) -> None:
    async def fake_synthesize(request, on_progress=None):
        return SynthesisResult(
            format=AudioFormat.MP3,
            audio_base64=base64.b64encode(b"json-audio").decode("ascii"),
            words=[],
            timing_source=TimingSource.WORD_BOUNDARY,
            provider=ProviderName.EDGE,
            voice=request.voice or "alloy",
            model=request.model,
            estimated=False,
        )

    monkeypatch.setattr("storyboat_tts_gateway.app.providers", {"edge": type("StubProvider", (), {"synthesize": staticmethod(fake_synthesize)})(), "kokoro": object()})

    client = TestClient(app)
    response = client.post("/v1/audio/speech_base64", json=_job_request_payload())

    assert response.status_code == 200
    assert response.json() == {
        "audio_base64": base64.b64encode(b"json-audio").decode("ascii"),
        "format": "mp3",
    }
