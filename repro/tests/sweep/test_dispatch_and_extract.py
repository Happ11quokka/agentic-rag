import pytest
from sweep.agent_runner import extract_final_answer

def test_extract_react_finish_action():
    result = {"raw_messages": [], "answer": "Action: Finish[Paris]"}
    # extract_final_answer should accept either the raw {answer: ...} or unwrap Finish[]
    assert extract_final_answer("react", result) == "Paris"

def test_extract_react_plain_text():
    result = {"answer": "Paris", "raw_messages": []}
    assert extract_final_answer("react", result) == "Paris"

def test_extract_lats_passthrough():
    result = {"answer": "Eiffel Tower", "raw_messages": []}
    assert extract_final_answer("lats", result) == "Eiffel Tower"

def test_extract_llmcompiler_strips_whitespace():
    result = {"answer": "  Paris\n", "raw_messages": []}
    assert extract_final_answer("llmcompiler", result) == "Paris"
