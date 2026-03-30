import base64
import json

from storayboat_tts_gateway.api_models import AudioFormat, ProviderName, SynthesisResult, TimingSource
from storayboat_tts_gateway.app import build_multipart_bundle


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
