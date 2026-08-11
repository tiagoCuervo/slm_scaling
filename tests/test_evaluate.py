import torch

from slm.evaluate import _evaluation_pair, summarize


def test_paper_evaluation_excludes_terminal_eos():
    inputs, targets = _evaluation_pair(
        torch.tensor([4, 5, 9]),
        eos=9,
        include_terminal_eos=False,
    )
    assert inputs.tolist() == [9, 4]
    assert targets.tolist() == [4, 5]


def test_storycloze_is_grouped_by_metadata():
    metrics = summarize(
        "storycloze",
        [
            {"id": "a", "group_id": "q1", "correct": True, "nll": 1.0},
            {"id": "b", "group_id": "q1", "correct": False, "nll": 2.0},
            {"id": "c", "group_id": "q2", "correct": True, "nll": 3.0},
            {"id": "d", "group_id": "q2", "correct": False, "nll": 2.0},
        ],
    )
    assert metrics["accuracy"] == 0.5


def test_sblimp_uses_official_half_credit_for_ties():
    scores = [
        {"id": "a_correct", "group_id": "a", "correct": True, "nll": 1.0},
        {"id": "a_incorrect", "group_id": "a", "correct": False, "nll": 1.0},
    ]
    assert summarize("sblimp", scores)["accuracy"] == 0.5


def test_sblimp_macro_averages_phenomena():
    scores = [
        {"id": "a+", "group_id": "a", "pair_id": "a", "phenomenon": "common", "correct": True, "nll": 1.0},
        {"id": "a-", "group_id": "a", "pair_id": "a", "phenomenon": "common", "correct": False, "nll": 2.0},
        {"id": "b+", "group_id": "b", "pair_id": "b", "phenomenon": "common", "correct": True, "nll": 1.0},
        {"id": "b-", "group_id": "b", "pair_id": "b", "phenomenon": "common", "correct": False, "nll": 2.0},
        {"id": "c+", "group_id": "c", "pair_id": "c", "phenomenon": "rare", "correct": True, "nll": 2.0},
        {"id": "c-", "group_id": "c", "pair_id": "c", "phenomenon": "rare", "correct": False, "nll": 1.0},
    ]
    metrics = summarize("sblimp", scores)
    assert metrics["pair_accuracy"] == 2 / 3
    assert metrics["accuracy"] == 0.5
