"""HotpotQA exact-match scoring, ported from hotpot_evaluate_v1.py.

Matches AgentBench's hotpot_evaluate.py byte-for-byte so our EM aligns with the
canonical paper evaluator. Two subtleties baked in:
  1. The " until " -> "-" replacement (a HotpotQA-specific pre-processing hack)
  2. Order is lower -> remove_punc -> remove_articles -> whitespace_fix
"""
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lowercase, remove articles, strip punctuation, normalize whitespace.

    Ported verbatim from AgentBench/src/tools/hotpotqa_tools/hotpot_evaluate.py
    (which is itself the canonical paper implementation).
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    s = s.replace(" until ", "-")
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def hotpotqa_em(predicted: str, expected: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(expected)


def hotpotqa_f1(predicted: str, expected: str) -> float:
    """Token-level F1, ported from the canonical hotpot_evaluate_v1.py.

    Partial-credit sibling of EM: splits both answers into normalized tokens
    and scores their overlap, so a near-miss earns credit instead of 0. E.g.
    "15 August 1843" vs gold "1843" -> EM=0 but F1=0.5. Returns F1 in [0, 1].
    Uses the same normalize_answer() as EM so the two stay consistent.
    """
    norm_pred = normalize_answer(predicted)
    norm_gold = normalize_answer(expected)
    # yes/no/noanswer must match exactly (HotpotQA convention).
    for tok in ("yes", "no", "noanswer"):
        if (norm_pred == tok or norm_gold == tok) and norm_pred != norm_gold:
            return 0.0
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
