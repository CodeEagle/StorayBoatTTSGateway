import base64

import pytest

from storayboat_tts_gateway.api_models import SpeechRequest
from storayboat_tts_gateway.providers.kokoro_provider import KokoroProvider


def test_parse_timestamp_entries_supports_seconds_and_text_key() -> None:
    provider = KokoroProvider(base_url="http://localhost:8880")
    result = provider._parse_timestamps(
        [
            {"text": "Hello", "start": 0.0, "end": 0.35},
            {"word": "world", "start_time": 0.36, "end_time": 0.72},
        ]
    )

    assert result[0].text == "Hello"
    assert result[0].start_ms == 0
    assert result[0].end_ms == 350
    assert result[1].text == "world"
    assert result[1].start_ms == 360
    assert result[1].end_ms == 720


def test_parse_timestamp_entries_supports_millisecond_keys() -> None:
    provider = KokoroProvider(base_url="http://localhost:8880")
    result = provider._parse_timestamps(
        [
            {"word": "one", "start_ms": 12, "end_ms": 34},
            {"text": "two", "begin": 35, "stop": 80},
        ]
    )

    assert result[0].start_ms == 12
    assert result[0].end_ms == 34
    assert result[1].start_ms == 35
    assert result[1].end_ms == 80


@pytest.mark.asyncio
async def test_kokoro_provider_sanitizes_input_before_upstream_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "audio": base64.b64encode(b"audio").decode("ascii"),
                "timestamps": [{"word": "你好", "start_ms": 0, "end_ms": 120}],
            }

    class DummyClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> DummyResponse:
            captured["url"] = url
            captured["json"] = json
            return DummyResponse()

    monkeypatch.setattr("storayboat_tts_gateway.providers.kokoro_provider.httpx.AsyncClient", DummyClient)

    provider = KokoroProvider(base_url="http://localhost:8880")
    request = SpeechRequest(provider="kokoro", input="你好\t世界\n第二行\x00")
    await provider.synthesize(request)

    assert captured["url"] == "http://localhost:8880/dev/captioned_speech"
    assert captured["json"]["input"] == "你好 世界 第二行"
