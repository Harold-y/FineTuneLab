import pytest

from finetunelab.errors import FineTuneLabError
from finetunelab.judges import parse_judge_response


def test_parse_fenced_judge_json() -> None:
    value = parse_judge_response(
        '```json\n{"winner": 1, "scores": [0.1, 0.9], "reason": "clear"}\n```', 2
    )
    assert value["winner"] == 1
    assert value["scores"] == [0.1, 0.9]


def test_invalid_winner_is_rejected() -> None:
    with pytest.raises(FineTuneLabError, match="outside"):
        parse_judge_response('{"winner": 4, "scores": [0, 1]}', 2)
