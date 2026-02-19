from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class InputMessage(BaseModel):
    text: str = Field(..., description="Input text chunk")
    flush: bool = Field(..., description="Whether to flush current buffer")


class AlignmentPayload(BaseModel):
    chars: List[str]
    char_start_times_ms: List[int]
    char_durations_ms: List[int]
    char_indices: List[int] = Field(default_factory=list)


class OutputMessage(BaseModel):
    audio: str
    alignment: AlignmentPayload
