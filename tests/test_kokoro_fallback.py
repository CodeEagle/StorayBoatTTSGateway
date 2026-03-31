import base64

from storyboat_tts_gateway.api_models import AudioFormat
from storyboat_tts_gateway.providers.kokoro_provider import KokoroProvider


def test_estimate_fallback_timings_handles_chinese_text() -> None:
    provider = KokoroProvider(base_url="http://localhost:8880")
    provider._audio_duration_ms = lambda audio_base64, audio_format: 1200  # type: ignore[method-assign]

    result = provider._estimate_fallback_timings(
        "今天天气很好。",
        base64.b64encode(b"fake-audio").decode("ascii"),
        AudioFormat.MP3,
    )

    assert [item.text for item in result] == ["今", "天", "天", "气", "很", "好", "。"]
    assert result[0].start_ms == 0
    assert result[-1].end_ms == 1200
