"""Typing tests."""
from transformer_from_scratch.types import TensorShapeLabels


class TestTensorShapeLabels:
    """TensorShapeLabels tests."""

    def test_enum_elements_are_unique(self):
        """Verify that we don't have any repeated enum elements."""

        enum_values = [e.value for e in TensorShapeLabels]  # type: ignore
        unique_enum_values = set(enum_values)
        assert len(enum_values) == len(
            unique_enum_values
        ), f"Duplicate enum elements found: {enum_values}"
