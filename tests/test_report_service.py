"""Tests for report generation service."""

import pytest
from app.services.report_service import generate_evaluation_summary, _score_to_rating, _heuristic_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluation(overall=75.0, heuristic_scores=None):
    """Build a minimal evaluation_data dict."""
    if heuristic_scores is None:
        heuristic_scores = []
    return {
        "overall_score": overall,
        "heuristic_scores": heuristic_scores,
        "total_violations": sum(
            len(hs.get("violations", [])) for hs in heuristic_scores
        ),
        "critical_issues": sum(
            sum(1 for v in hs.get("violations", []) if v.get("severity") == "critical")
            for hs in heuristic_scores
        ),
    }


def _make_heuristic_score(h_id="H1", score=80, violations=None):
    return {
        "heuristic_id": h_id,
        "score": score,
        "max_score": 100,
        "percentage": score,
        "violations": violations or [],
        "explanation": f"Score {score}/100",
    }


def _make_violation(criterion="H1.1", severity="major", desc="Test issue"):
    return {
        "criterion_id": criterion,
        "severity": severity,
        "description": desc,
        "affected_elements": ["element1"],
        "recommendation": "Fix this issue",
    }


# ---------------------------------------------------------------------------
# _score_to_rating
# ---------------------------------------------------------------------------

class TestScoreToRating:
    def test_excellent(self):
        assert _score_to_rating(95) == "Excellent"
        assert _score_to_rating(90) == "Excellent"

    def test_good(self):
        assert _score_to_rating(80) == "Good"
        assert _score_to_rating(75) == "Good"

    def test_needs_improvement(self):
        assert _score_to_rating(60) == "Needs Improvement"
        assert _score_to_rating(50) == "Needs Improvement"

    def test_poor(self):
        assert _score_to_rating(30) == "Poor"
        assert _score_to_rating(0) == "Poor"


# ---------------------------------------------------------------------------
# _heuristic_name
# ---------------------------------------------------------------------------

class TestHeuristicName:
    def test_known_heuristic(self):
        assert _heuristic_name("H1") == "Visibility of System Status"
        assert _heuristic_name("H10") == "Help and Documentation"

    def test_unknown_returns_id(self):
        assert _heuristic_name("H99") == "H99"


# ---------------------------------------------------------------------------
# generate_evaluation_summary
# ---------------------------------------------------------------------------

class TestGenerateEvaluationSummary:
    def test_empty_evaluation(self):
        data = _make_evaluation(overall=0, heuristic_scores=[])
        summary = generate_evaluation_summary(data)
        assert summary["overall"]["score"] == 0
        assert summary["overall"]["rating"] == "Poor"
        assert summary["overall"]["total_violations"] == 0
        assert summary["severity_distribution"] == {
            "critical": 0, "major": 0, "minor": 0, "cosmetic": 0
        }
        assert summary["priority_issues"] == []
        assert summary["strengths"] == []
        assert summary["weaknesses"] == []

    def test_perfect_evaluation(self):
        scores = [_make_heuristic_score(f"H{i}", 95) for i in range(1, 11)]
        data = _make_evaluation(overall=95, heuristic_scores=scores)
        summary = generate_evaluation_summary(data)
        assert summary["overall"]["rating"] == "Excellent"
        assert len(summary["strengths"]) == 10
        assert summary["weaknesses"] == []

    def test_severity_distribution_counted_correctly(self):
        violations = [
            _make_violation(severity="critical"),
            _make_violation(severity="critical"),
            _make_violation(severity="major"),
            _make_violation(severity="minor"),
        ]
        scores = [_make_heuristic_score("H1", 40, violations)]
        data = _make_evaluation(overall=40, heuristic_scores=scores)
        summary = generate_evaluation_summary(data)
        assert summary["severity_distribution"]["critical"] == 2
        assert summary["severity_distribution"]["major"] == 1
        assert summary["severity_distribution"]["minor"] == 1
        assert summary["severity_distribution"]["cosmetic"] == 0

    def test_heuristic_breakdown_sorted_worst_first(self):
        scores = [
            _make_heuristic_score("H1", 90),
            _make_heuristic_score("H2", 30),
            _make_heuristic_score("H3", 60),
        ]
        data = _make_evaluation(overall=60, heuristic_scores=scores)
        summary = generate_evaluation_summary(data)
        breakdown = summary["heuristic_breakdown"]
        assert breakdown[0]["heuristic_id"] == "H2"  # worst first
        assert breakdown[-1]["heuristic_id"] == "H1"  # best last

    def test_priority_issues_sorted_by_impact(self):
        violations = [
            _make_violation(severity="cosmetic", desc="minor styling"),
            _make_violation(severity="critical", desc="broken navigation"),
        ]
        scores = [_make_heuristic_score("H3", 50, violations)]
        data = _make_evaluation(overall=50, heuristic_scores=scores)
        summary = generate_evaluation_summary(data)
        issues = summary["priority_issues"]
        assert issues[0]["severity"] == "critical"
        assert issues[0]["impact_score"] == 10
        assert issues[1]["severity"] == "cosmetic"
        assert issues[1]["impact_score"] == 1

    def test_priority_issues_capped_at_10(self):
        violations = [_make_violation(severity="minor", desc=f"issue {i}") for i in range(20)]
        scores = [_make_heuristic_score("H1", 20, violations)]
        data = _make_evaluation(overall=20, heuristic_scores=scores)
        summary = generate_evaluation_summary(data)
        assert len(summary["priority_issues"]) == 10

    def test_recommendations_grouped_by_urgency(self):
        violations = [
            _make_violation(severity="critical", desc="crash"),
            _make_violation(severity="major", desc="confusing"),
            _make_violation(severity="cosmetic", desc="ugly"),
        ]
        scores = [_make_heuristic_score("H1", 30, violations)]
        data = _make_evaluation(overall=30, heuristic_scores=scores)
        summary = generate_evaluation_summary(data)
        recs = summary["recommendations"]
        assert "fix_immediately" in recs
        assert "fix_soon" in recs
        assert "consider_fixing" in recs
        assert len(recs["fix_immediately"]) == 1
        assert recs["fix_immediately"][0]["issue"] == "crash"

    def test_weaknesses_detected_below_50(self):
        scores = [
            _make_heuristic_score("H1", 95),
            _make_heuristic_score("H2", 40),
            _make_heuristic_score("H3", 25),
        ]
        data = _make_evaluation(overall=53, heuristic_scores=scores)
        summary = generate_evaluation_summary(data)
        assert len(summary["weaknesses"]) == 2
        assert any("H2" in w for w in summary["weaknesses"])
        assert any("H3" in w for w in summary["weaknesses"])

    def test_generated_at_is_present(self):
        data = _make_evaluation(overall=80, heuristic_scores=[])
        summary = generate_evaluation_summary(data)
        assert "generated_at" in summary

    def test_violations_carry_heuristic_name(self):
        violations = [_make_violation(severity="major", desc="no feedback")]
        scores = [_make_heuristic_score("H1", 70, violations)]
        data = _make_evaluation(overall=70, heuristic_scores=scores)
        summary = generate_evaluation_summary(data)
        issue = summary["priority_issues"][0]
        assert issue["heuristic_name"] == "Visibility of System Status"
