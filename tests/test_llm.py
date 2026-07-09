from aiapply_lite.core.llm import _parse_json_object


def test_parse_plain_json() -> None:
    assert _parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_fenced_json() -> None:
    raw = '```json\n{"a": 2}\n```'
    assert _parse_json_object(raw) == {"a": 2}


def test_parse_json_with_prose() -> None:
    raw = 'Here you go: {"a": 3} hope that helps'
    assert _parse_json_object(raw) == {"a": 3}


def test_parse_invalid_returns_empty() -> None:
    assert _parse_json_object("not json at all") == {}
