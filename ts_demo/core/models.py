from __future__ import annotations

from typing import Dict, List, Literal
from pydantic import BaseModel, ConfigDict, Field

Polarity = Literal["BULLISH", "BEARISH", "MIXED", "NEUTRAL"]
Modality = Literal["ASSERTION", "FORECAST", "INTENTION", "CONDITIONAL", "RISK", "OPINION"]
Evidential = Literal["REPORTED_NUMBER", "OPERATIONAL_OBSERVATION", "INTERNAL_METRIC", "UNSPECIFIED"]
SpeakerRole = Literal["COMPANY_OFFICIAL", "MANAGEMENT", "ANALYST", "OTHER"]


class HorizonProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    one_day: float = Field(..., ge=0.0, le=1.0, alias="1D")
    one_week: float = Field(..., ge=0.0, le=1.0, alias="1W")
    one_month: float = Field(..., ge=0.0, le=1.0, alias="1M")
    three_month: float = Field(..., ge=0.0, le=1.0, alias="3M")
    one_year: float = Field(..., ge=0.0, le=1.0, alias="1Y")
    three_year: float = Field(..., ge=0.0, le=1.0, alias="3Y")

    def as_dict(self) -> Dict[str, float]:
        return self.model_dump(by_alias=True)


class ExtractedClaim(BaseModel):
    claim: str
    polarity: Polarity
    materiality_0_1: float = Field(..., ge=0.0, le=1.0)
    credibility_0_1: float = Field(..., ge=0.0, le=1.0)
    surprise_0_1: float = Field(..., ge=0.0, le=1.0)
    horizon_profile: HorizonProfile
    rationale: str
    quote: str
    is_forward_looking: bool
    modality: Modality
    commitment_0_1: float = Field(..., ge=0.0, le=1.0)
    conditionality_0_1: float = Field(..., ge=0.0, le=1.0)
    evidential_basis: Evidential
    speaker_role: SpeakerRole


class ExtractedClaims(BaseModel):
    claims: List[ExtractedClaim]
