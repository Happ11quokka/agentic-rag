from measurement.eval import normalize_answer, hotpotqa_em

def test_normalize_strips_articles():
    assert normalize_answer("The Eiffel Tower") == "eiffel tower"
    assert normalize_answer("a cat") == "cat"
    assert normalize_answer("an apple") == "apple"

def test_normalize_strips_punctuation():
    assert normalize_answer("Paris, France.") == "paris france"

def test_normalize_lowercases():
    assert normalize_answer("PARIS") == "paris"

def test_em_passes_normalized_equal():
    assert hotpotqa_em("The Eiffel Tower", "eiffel tower") is True
    assert hotpotqa_em("Paris.", "paris") is True

def test_em_fails_substring():
    assert hotpotqa_em("Paris, France", "Paris") is False

def test_em_handles_empty():
    assert hotpotqa_em("", "") is True
    assert hotpotqa_em("answer", "") is False
