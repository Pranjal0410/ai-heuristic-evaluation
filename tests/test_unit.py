"""Unit tests for core heuristic evaluation components.

These tests validate the data models, scoring logic, element parsing,
and utility functions WITHOUT requiring OmniParser weights or an OpenAI API key.
This enables reliable testing in CI environments.
"""

import pytest
import json
from datetime import datetime

from app.core.constants import NIELSEN_HEURISTICS, HeuristicId, SeverityLevel
from app.services.heuristic_engine import (
    HeuristicViolation,
    HeuristicScore,
    HeuristicEvaluationResult,
    HeuristicEvaluationEngine,
)
from app.services.omniparser_client import (
    UIElement,
    UIElementDetectionResult,
    infer_heading_level,
    calculate_height_variance,
)
from app.services.exceptions import (
    ServiceException,
    InvalidInputError,
    OmniParserError,
    ModelInferenceError,
    RAGKnowledgeBaseError,
)


# ---------------------------------------------------------------------------
# Constants & Heuristic definitions
# ---------------------------------------------------------------------------

class TestNielsenHeuristics:
    """Validate the heuristic constant definitions."""

    def test_all_ten_heuristics_defined(self):
        assert len(NIELSEN_HEURISTICS) == 10

    def test_every_heuristic_id_has_definition(self):
        for h_id in HeuristicId:
            assert h_id in NIELSEN_HEURISTICS, f"Missing definition for {h_id}"

    def test_each_heuristic_has_required_fields(self):
        for h_id, data in NIELSEN_HEURISTICS.items():
            assert "name" in data, f"{h_id} missing name"
            assert "description" in data, f"{h_id} missing description"
            assert "measurable_criteria" in data, f"{h_id} missing criteria"
            assert len(data["measurable_criteria"]) > 0, f"{h_id} has no criteria"

    def test_criteria_ids_match_parent_heuristic(self):
        for h_id, data in NIELSEN_HEURISTICS.items():
            prefix = h_id.value  # e.g. "H1"
            for criterion in data["measurable_criteria"]:
                assert criterion["id"].startswith(prefix), (
                    f"Criterion {criterion['id']} does not match parent {prefix}"
                )

    def test_criteria_have_severity_weights(self):
        expected_severities = {"critical", "major", "minor", "cosmetic"}
        for h_id, data in NIELSEN_HEURISTICS.items():
            for criterion in data["measurable_criteria"]:
                weights = criterion.get("severity_weights", {})
                assert set(weights.keys()) == expected_severities, (
                    f"Criterion {criterion['id']} has wrong severity keys: {set(weights.keys())}"
                )


class TestSeverityLevel:
    """Validate the SeverityLevel enum."""

    def test_has_four_levels(self):
        assert len(SeverityLevel) == 4

    def test_level_values(self):
        assert SeverityLevel.CRITICAL.value == "critical"
        assert SeverityLevel.MAJOR.value == "major"
        assert SeverityLevel.MINOR.value == "minor"
        assert SeverityLevel.COSMETIC.value == "cosmetic"

    def test_from_string(self):
        assert SeverityLevel("critical") == SeverityLevel.CRITICAL
        assert SeverityLevel("cosmetic") == SeverityLevel.COSMETIC


# ---------------------------------------------------------------------------
# UIElement
# ---------------------------------------------------------------------------

class TestUIElement:
    """Validate UIElement creation, properties, and serialization."""

    def _make_element(self, **overrides):
        defaults = {
            "element_type": "button",
            "bbox": [10, 20, 110, 60],
            "content": "Submit",
            "interactivity": True,
        }
        defaults.update(overrides)
        return UIElement(**defaults)

    def test_basic_creation(self):
        elem = self._make_element()
        assert elem.element_type == "button"
        assert elem.content == "Submit"
        assert elem.interactivity is True

    def test_text_alias(self):
        elem = self._make_element(content="Click me")
        assert elem.text == "Click me"

    def test_computed_width_height(self):
        elem = self._make_element(bbox=[10, 20, 110, 70])
        assert elem.width == 100
        assert elem.height == 50

    def test_zero_size_bbox(self):
        elem = self._make_element(bbox=[0, 0, 0, 0])
        assert elem.width == 0
        assert elem.height == 0

    def test_bounds_property(self):
        elem = self._make_element(bbox=[10, 20, 110, 70])
        bounds = elem.bounds
        assert bounds["x"] == 10
        assert bounds["y"] == 20
        assert bounds["width"] == 100
        assert bounds["height"] == 50

    def test_to_dict(self):
        elem = self._make_element()
        d = elem.to_dict()
        assert d["type"] == "button"
        assert d["bbox"] == [10, 20, 110, 60]
        assert d["content"] == "Submit"
        assert d["interactivity"] is True

    def test_from_dict_omniparser_format(self):
        raw = {
            "type": "input",
            "bbox": [0, 0, 200, 40],
            "content": "Email",
            "interactivity": True,
        }
        elem = UIElement.from_dict(raw)
        assert elem.element_type == "input"
        assert elem.content == "Email"
        assert elem.interactivity is True

    def test_from_dict_legacy_format(self):
        raw = {
            "element_type": "button",
            "bbox": [5, 5, 100, 45],
            "text": "OK",
            "interactive": False,
        }
        elem = UIElement.from_dict(raw)
        assert elem.element_type == "button"
        assert elem.text == "OK"
        assert elem.interactivity is False

    def test_from_dict_missing_fields_use_defaults(self):
        elem = UIElement.from_dict({})
        assert elem.element_type == "unknown"
        assert elem.bbox == [0, 0, 0, 0]
        assert elem.content == ""
        assert elem.interactivity is False


# ---------------------------------------------------------------------------
# UIElementDetectionResult
# ---------------------------------------------------------------------------

class TestUIElementDetectionResult:
    def test_to_dict(self):
        elements = [
            UIElement("button", [0, 0, 100, 40], "OK", True),
            UIElement("text", [0, 50, 200, 70], "Welcome", False),
        ]
        result = UIElementDetectionResult(
            elements=elements,
            layout_hierarchy={"root": "main"},
            metadata={"width": 1920, "height": 1080},
        )
        d = result.to_dict()
        assert len(d["elements"]) == 2
        assert d["layout_hierarchy"] == {"root": "main"}
        assert d["metadata"]["width"] == 1920

    def test_empty_elements(self):
        result = UIElementDetectionResult(elements=[], layout_hierarchy={})
        d = result.to_dict()
        assert d["elements"] == []
        assert d["metadata"] == {}


# ---------------------------------------------------------------------------
# infer_heading_level
# ---------------------------------------------------------------------------

class TestInferHeadingLevel:
    def _make_text_elements(self, heights):
        return [
            UIElement("text", [0, 0, 100, h], f"text_{i}")
            for i, h in enumerate(heights)
        ]

    def test_largest_element_is_h1(self):
        elements = self._make_text_elements([50, 30, 20, 15, 10, 8, 5, 5, 5, 5,
                                              5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        level = infer_heading_level(elements[0], elements)
        assert level == 1

    def test_small_element_returns_none(self):
        elements = self._make_text_elements([50, 40, 30, 20, 10, 5, 5, 5])
        smallest = elements[-1]  # height=5, well below median
        level = infer_heading_level(smallest, elements)
        assert level is None

    def test_non_text_element_returns_none(self):
        button = UIElement("button", [0, 0, 100, 50], "Click")
        text_elements = [UIElement("text", [0, 0, 100, 30], "Hi")]
        assert infer_heading_level(button, text_elements) is None

    def test_zero_height_returns_none(self):
        elem = UIElement("text", [0, 0, 0, 0], "empty")
        assert infer_heading_level(elem, [elem]) is None


# ---------------------------------------------------------------------------
# calculate_height_variance
# ---------------------------------------------------------------------------

class TestCalculateHeightVariance:
    def test_identical_heights_zero_variance(self):
        elements = [
            UIElement("button", [0, 0, 100, 40], "A"),
            UIElement("button", [0, 0, 100, 40], "B"),
        ]
        assert calculate_height_variance(elements) == 0.0

    def test_different_heights_nonzero_variance(self):
        elements = [
            UIElement("button", [0, 0, 100, 40], "A"),
            UIElement("button", [0, 0, 100, 80], "B"),
        ]
        assert calculate_height_variance(elements) > 0

    def test_single_element_zero_variance(self):
        elements = [UIElement("button", [0, 0, 100, 40], "A")]
        assert calculate_height_variance(elements) == 0.0

    def test_empty_list_zero_variance(self):
        assert calculate_height_variance([]) == 0.0


# ---------------------------------------------------------------------------
# HeuristicViolation
# ---------------------------------------------------------------------------

class TestHeuristicViolation:
    def _make_violation(self, **overrides):
        defaults = {
            "heuristic_id": "H1",
            "criterion_id": "H1.2",
            "severity": SeverityLevel.MAJOR,
            "description": "Button lacks hover feedback",
            "affected_elements": ["Submit", "Cancel"],
            "recommendation": "Add hover state",
        }
        defaults.update(overrides)
        return HeuristicViolation(**defaults)

    def test_to_dict(self):
        v = self._make_violation()
        d = v.to_dict()
        assert d["heuristic_id"] == "H1"
        assert d["criterion_id"] == "H1.2"
        assert d["severity"] == "major"
        assert len(d["affected_elements"]) == 2

    def test_critical_severity_serialization(self):
        v = self._make_violation(severity=SeverityLevel.CRITICAL)
        assert v.to_dict()["severity"] == "critical"

    def test_cosmetic_severity_serialization(self):
        v = self._make_violation(severity=SeverityLevel.COSMETIC)
        assert v.to_dict()["severity"] == "cosmetic"

    def test_empty_affected_elements(self):
        v = self._make_violation(affected_elements=[])
        assert v.to_dict()["affected_elements"] == []


# ---------------------------------------------------------------------------
# HeuristicScore
# ---------------------------------------------------------------------------

class TestHeuristicScore:
    def test_perfect_score(self):
        s = HeuristicScore(heuristic_id="H1", score=100)
        d = s.to_dict()
        assert d["score"] == 100
        assert d["percentage"] == 100.0
        assert d["violations"] == []

    def test_zero_score(self):
        s = HeuristicScore(heuristic_id="H3", score=0)
        d = s.to_dict()
        assert d["score"] == 0
        assert d["percentage"] == 0.0

    def test_score_with_violations(self):
        v = HeuristicViolation("H2", "H2.1", SeverityLevel.MINOR, "Bad icon", ["icon1"], "Fix it")
        s = HeuristicScore(heuristic_id="H2", score=85, violations=[v])
        d = s.to_dict()
        assert len(d["violations"]) == 1
        assert d["violations"][0]["criterion_id"] == "H2.1"

    def test_llm_explanation_included(self):
        s = HeuristicScore(heuristic_id="H1", score=90, llm_explanation="Looks good overall")
        assert s.to_dict()["llm_explanation"] == "Looks good overall"

    def test_llm_explanation_none_by_default(self):
        s = HeuristicScore(heuristic_id="H1", score=90)
        assert s.to_dict()["llm_explanation"] is None


# ---------------------------------------------------------------------------
# HeuristicEvaluationResult
# ---------------------------------------------------------------------------

class TestHeuristicEvaluationResult:
    def test_full_result_serialization(self):
        scores = [
            HeuristicScore(heuristic_id=f"H{i}", score=80 + i)
            for i in range(1, 11)
        ]
        result = HeuristicEvaluationResult(
            overall_score=85.5,
            heuristic_scores=scores,
            total_violations=3,
            critical_issues=1,
        )
        d = result.to_dict()
        assert d["overall_score"] == 85.5
        assert len(d["heuristic_scores"]) == 10
        assert d["total_violations"] == 3
        assert d["critical_issues"] == 1
        assert "timestamp" in d

    def test_timestamp_is_valid_iso_format(self):
        result = HeuristicEvaluationResult(
            overall_score=50,
            heuristic_scores=[],
            total_violations=0,
            critical_issues=0,
        )
        # Should not raise
        datetime.fromisoformat(result.timestamp)

    def test_metadata_defaults_to_empty_dict(self):
        result = HeuristicEvaluationResult(
            overall_score=50,
            heuristic_scores=[],
            total_violations=0,
            critical_issues=0,
        )
        assert result.evaluation_metadata == {}


# ---------------------------------------------------------------------------
# HeuristicEvaluationEngine — scoring logic (no LLM/API needed)
# ---------------------------------------------------------------------------

class TestHeuristicEngineScoring:
    """Test the calculate_score method which is pure logic, no external calls."""

    def setup_method(self):
        self.engine = HeuristicEvaluationEngine()

    def test_no_violations_gives_perfect_score(self):
        score, explanation = self.engine.calculate_score([], "H1")
        assert score == 100

    def test_single_major_violation_deducts_correctly(self):
        v = HeuristicViolation("H1", "H1.2", SeverityLevel.MAJOR, "No feedback", [], "Fix")
        score, explanation = self.engine.calculate_score([v], "H1")
        # H1.2 major weight = 6, so score = 100 - 6 = 94
        assert score == 94

    def test_critical_violation_deducts_more(self):
        v = HeuristicViolation("H1", "H1.2", SeverityLevel.CRITICAL, "Broken", [], "Fix")
        score, explanation = self.engine.calculate_score([v], "H1")
        # H1.2 critical weight = 10, so score = 100 - 10 = 90
        assert score == 90

    def test_cosmetic_violation_deducts_least(self):
        v = HeuristicViolation("H1", "H1.2", SeverityLevel.COSMETIC, "Slight", [], "Tweak")
        score, explanation = self.engine.calculate_score([v], "H1")
        # H1.2 cosmetic weight = 1, so score = 100 - 1 = 99
        assert score == 99

    def test_multiple_violations_accumulate(self):
        violations = [
            HeuristicViolation("H1", "H1.1", SeverityLevel.CRITICAL, "No loading", [], "Add"),
            HeuristicViolation("H1", "H1.2", SeverityLevel.MAJOR, "No feedback", [], "Add"),
            HeuristicViolation("H1", "H1.3", SeverityLevel.MINOR, "No progress", [], "Add"),
        ]
        score, explanation = self.engine.calculate_score(violations, "H1")
        # H1.1 critical=10, H1.2 major=6, H1.3 minor=2 → total=18, score=82
        assert score == 82

    def test_score_never_goes_below_zero(self):
        violations = [
            HeuristicViolation("H1", "H1.1", SeverityLevel.CRITICAL, f"Issue {i}", [], "Fix")
            for i in range(20)
        ]
        score, explanation = self.engine.calculate_score(violations, "H1")
        assert score == 0

    def test_unknown_heuristic_raises_on_invalid_id(self):
        with pytest.raises(ValueError):
            self.engine.calculate_score([], "H99")

    def test_explanation_contains_violation_descriptions(self):
        v = HeuristicViolation("H3", "H3.1", SeverityLevel.MAJOR, "No undo", ["btn"], "Add undo")
        _, explanation = self.engine.calculate_score([v], "H3")
        assert "No undo" in explanation
        assert "major" in explanation


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_service_exception_base(self):
        e = ServiceException("test error", {"key": "value"})
        assert e.message == "test error"
        assert e.details == {"key": "value"}
        assert str(e) == "test error"

    def test_service_exception_default_details(self):
        e = ServiceException("error")
        assert e.details == {}

    def test_invalid_input_is_service_exception(self):
        e = InvalidInputError("bad input")
        assert isinstance(e, ServiceException)

    def test_omniparser_error_is_service_exception(self):
        e = OmniParserError("parse failed")
        assert isinstance(e, ServiceException)

    def test_model_inference_error_is_service_exception(self):
        e = ModelInferenceError("model down")
        assert isinstance(e, ServiceException)

    def test_rag_error_is_service_exception(self):
        e = RAGKnowledgeBaseError("kb failed")
        assert isinstance(e, ServiceException)


# ---------------------------------------------------------------------------
# Engine initialization state
# ---------------------------------------------------------------------------

class TestHeuristicEngineInit:
    def test_engine_starts_uninitialized(self):
        engine = HeuristicEvaluationEngine()
        assert engine.initialized is False
        assert engine.llm_client is None
        assert engine.rag_kb is None