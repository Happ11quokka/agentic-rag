"""Pinned FanOutQA question loader for the local open-book experiment."""

from .dataset import Question, load_dev, select_questions

__all__ = ["Question", "load_dev", "select_questions"]
