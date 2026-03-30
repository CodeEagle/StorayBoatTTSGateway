from storayboat_tts_gateway.api_models import SpeechRequest


def test_speech_request_accepts_normalization_options() -> None:
    request = SpeechRequest(
        provider="kokoro",
        input="你好，世界",
        normalization_options={"normalize": True},
    )

    assert request.normalization_options == {"normalize": True}
