"""HotpotQA exact-match scoring, ported from hotpot_evaluate_v1.py.

Matches AgentBench's hotpot_evaluate.py byte-for-byte so our EM aligns with the
canonical paper evaluator. Two subtleties baked in:
  1. The " until " -> "-" replacement (a HotpotQA-specific pre-processing hack)
  2. Order is lower -> remove_punc -> remove_articles -> whitespace_fix
"""
import re
import string


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
