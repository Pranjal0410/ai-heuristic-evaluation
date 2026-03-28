"""Evaluation report and analytics service.

Transforms raw heuristic evaluation results into actionable summaries
with severity analytics, priority rankings, and fix recommendations.

This implements the 'Multi-Dimensional Reporting' feature from the
AI Heuristic Evaluation roadmap (Phase 3).
"""

import logging
from typing import Any, Dict, List
from datetime import datetime, timezone

from app.core.constants import NIELSEN_HEURISTICS, HeuristicId

logger = logging.getLogger(__name__)

# Severity impact weights for priority scoring
SEVERITY_IMPACT = {
    "critical": 10,
    "major": 6,
    "minor": 3,
    "cosmetic": 1,
}


def _score_to_rating(score: float) -> str:
    """Map a 0-100 score to a human-readable rating."""
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 50:
        return "Needs Improvement"
    else:
        return "Poor"


def _heuristic_name(heuristic_id: str) -> str:
    """Look up the human-readable name for a heuristic ID."""
    for h_id, h_def in NIELSEN_HEURISTICS.items():
        if h_id.value == heuristic_id:
            return h_def["name"]
    return heuristic_id


def generate_evaluation_summary(evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an actionable summary from raw evaluation results.

    Takes the output of HeuristicEvaluationResult.to_dict() and produces:
    - Overall rating with score interpretation
    - Severity distribution across all violations
    - Per-heuristic breakdown sorted by worst score first
    - Top priority issues ranked by impact (severity × weight)
    - Actionable recommendations grouped by urgency

    Args:
        evaluation_data: Dict from HeuristicEvaluationResult.to_dict()

    Returns:
        Structured summary dict ready for API response or report generation.
    """
    overall_score = evaluation_data.get("overall_score", 0)
    heuristic_scores = evaluation_data.get("heuristic_scores", [])

    # --- Severity distribution ---
    all_violations = []
    for hs in heuristic_scores:
        for v in hs.get("violations", []):
            all_violations.append({
                **v,
                "heuristic_id": hs["heuristic_id"],
                "heuristic_name": _heuristic_name(hs["heuristic_id"]),
            })

    severity_distribution = {"critical": 0, "major": 0, "minor": 0, "cosmetic": 0}
    for v in all_violations:
        sev = v.get("severity", "cosmetic")
        if sev in severity_distribution:
            severity_distribution[sev] += 1

    # --- Per-heuristic breakdown (worst first) ---
    heuristic_breakdown = []
    for hs in heuristic_scores:
        h_id = hs["heuristic_id"]
        score = hs.get("score", 0)
        violations = hs.get("violations", [])
        heuristic_breakdown.append({
            "heuristic_id": h_id,
            "name": _heuristic_name(h_id),
            "score": score,
            "rating": _score_to_rating(score),
            "violation_count": len(violations),
            "critical_count": sum(1 for v in violations if v.get("severity") == "critical"),
            "explanation": hs.get("llm_explanation") or hs.get("explanation"),
        })
    heuristic_breakdown.sort(key=lambda h: h["score"])

    # --- Priority issues (ranked by impact score) ---
    priority_issues = []
    for v in all_violations:
        sev = v.get("severity", "cosmetic")
        impact = SEVERITY_IMPACT.get(sev, 1)
        priority_issues.append({
            "impact_score": impact,
            "severity": sev,
            "heuristic_id": v.get("heuristic_id"),
            "heuristic_name": v.get("heuristic_name"),
            "criterion_id": v.get("criterion_id"),
            "description": v.get("description"),
            "recommendation": v.get("recommendation"),
            "affected_elements": v.get("affected_elements", []),
        })
    priority_issues.sort(key=lambda x: x["impact_score"], reverse=True)

    # --- Actionable recommendations grouped by urgency ---
    urgent = [p for p in priority_issues if p["severity"] == "critical"]
    important = [p for p in priority_issues if p["severity"] == "major"]
    minor = [p for p in priority_issues if p["severity"] in ("minor", "cosmetic")]

    recommendations = {}
    if urgent:
        recommendations["fix_immediately"] = [
            {
                "issue": p["description"],
                "fix": p["recommendation"],
                "heuristic": f"{p['heuristic_id']}: {p['heuristic_name']}",
            }
            for p in urgent
        ]
    if important:
        recommendations["fix_soon"] = [
            {
                "issue": p["description"],
                "fix": p["recommendation"],
                "heuristic": f"{p['heuristic_id']}: {p['heuristic_name']}",
            }
            for p in important
        ]
    if minor:
        recommendations["consider_fixing"] = [
            {
                "issue": p["description"],
                "fix": p["recommendation"],
                "heuristic": f"{p['heuristic_id']}: {p['heuristic_name']}",
            }
            for p in minor
        ]

    # --- Strengths (heuristics scoring 90+) ---
    strengths = [
        f"{h['heuristic_id']}: {h['name']}"
        for h in heuristic_breakdown
        if h["score"] >= 90
    ]

    # --- Weaknesses (heuristics scoring below 50) ---
    weaknesses = [
        f"{h['heuristic_id']}: {h['name']} (score: {h['score']})"
        for h in heuristic_breakdown
        if h["score"] < 50
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "score": overall_score,
            "rating": _score_to_rating(overall_score),
            "total_violations": len(all_violations),
            "heuristics_evaluated": len(heuristic_scores),
        },
        "severity_distribution": severity_distribution,
        "heuristic_breakdown": heuristic_breakdown,
        "priority_issues": priority_issues[:10],  # Top 10 by impact
        "recommendations": recommendations,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }
