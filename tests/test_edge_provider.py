from storayboat_tts_gateway.providers.edge_provider import edge_voice_display_name, parse_edge_voices_catalog


def test_parse_edge_voices_catalog_matches_app_style() -> None:
    text = """
Name: Microsoft Server Speech Text to Speech Voice (en-US, AvaMultilingualNeural)
ShortName: en-US-AvaMultilingualNeural
Gender: Female
Locale: en-US

Name: Microsoft Server Speech Text to Speech Voice (zh-CN, XiaoxiaoNeural)
ShortName: zh-CN-XiaoxiaoNeural
Gender: Female
Locale: zh-CN
""".strip()

    voices = parse_edge_voices_catalog(text)
    assert voices[0].id == "en-US-AvaMultilingualNeural"
    assert voices[0].name == "Ava"
    assert voices[0].locale == "en-US"
    assert voices[0].country == "US"
    assert voices[1].name == "Xiaoxiao"


def test_edge_voice_display_name_cleans_suffixes() -> None:
    assert edge_voice_display_name("Microsoft Voice (en-US, AvaMultilingualNeural)", "en-US-AvaMultilingualNeural") == "Ava"
