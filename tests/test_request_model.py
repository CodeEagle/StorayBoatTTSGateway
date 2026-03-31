from storyboat_tts_gateway.api_models import SpeechRequest


def test_speech_request_accepts_normalization_options() -> None:
    request = SpeechRequest(
        provider="kokoro",
        input="你好，世界",
        normalization_options={"normalize": True},
    )

    assert request.normalization_options == {"normalize": True}


def test_speech_request_sanitizes_control_characters_for_upstream_json() -> None:
    request = SpeechRequest(
        provider="kokoro",
        input="你好\t世界\n第二行\x00",
    )

    assert request.sanitized_input() == "你好 世界 第二行"


def test_speech_request_normalizes_lang_code() -> None:
    request = SpeechRequest(
        provider="edge",
        input="hello",
        lang="ZH_CN",
    )

    assert request.normalized_lang() == "zh-cn"
