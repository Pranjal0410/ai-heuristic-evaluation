"""Report generation API routes.

Provides endpoints for generating evaluation summaries and reports
from raw heuristic evaluation results.
"""

import logging
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any

from app.services.report_service import generate_evaluation_summary

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/summary")
async def get_evaluation_summary(
    evaluation_data: Dict[str, Any] = Body(
        ...,
        example={
            "overall_score": 72.5,
            "heuristic_scores": [
                {
                    "heuristic_id": "H1",
                    "score": 80,
                    "violations": [
                        {
                            "criterion_id": "H1.2",
                            "severity": "major",
                            "description": "Submit button lacks hover feedback",
                            "affected_elements": ["Submit"],
                            "recommendation": "Add hover and active states"
                        }
                    ],
                    "explanation": "Score 80/100 - 1 violation",
                    "llm_explanation": "The interface provides partial feedback."
                }
            ],
            "total_violations": 1,
            "critical_issues": 0
        }
    )
):
    """Generate an actionable summary from heuristic evaluation results.

    Accepts the raw output of the /evaluate endpoint and returns:
    - Overall rating with score interpretation
    - Severity distribution across all violations
    - Per-heuristic breakdown sorted by worst score first
    - Top 10 priority issues ranked by impact
    - Actionable recommendations grouped by urgency (fix immediately / fix soon / consider)
    - Identified strengths and weaknesses

    This endpoint does not require OmniParser or OpenAI — it operates
    purely on previously generated evaluation data.
    """
    try:
        summary = generate_evaluation_summary(evaluation_data)
        return {
            "success": True,
            "data": summary,
        }
    except KeyError as e:
        logger.warning(f"Missing field in evaluation data: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid evaluation data",
                "message": f"Missing required field: {e}",
            }
        )
    except Exception as e:
        logger.exception(f"Error generating evaluation summary: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": "Failed to generate evaluation summary",
            }
        )
