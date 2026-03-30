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
