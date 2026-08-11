"""Centralized tests for the interval-scored EIL task-utility judge."""
from __future__ import annotations

import asyncio
import json

from eil.scoring.task_utility import assess_utility

RECORD = {"id": "scene-1", "subscenario": "application", "necessary_information": [{"id": "nec_1", "content": "Submit the required form."}]}
RESPONSE = "I will submit the required form."
INFERENCE = {"inferred_private_facts": [], "counterparty_reply": "Please submit the form through the portal."}


class UtilityClient:
    model = "test-judge"

    async def chat_json(self, messages, *, temperature, max_tokens, seed=None):
        return json.dumps({"utility_score": 0.67, "agent_reply_evidence": "submit the required form", "counterparty_reply_evidence": "submit the form through the portal"})


def test_utility_maps_fixed_interval_to_metric() -> None:
    assert asyncio.run(assess_utility(RESPONSE, RECORD, INFERENCE, UtilityClient()))["task_utility"] == 0.67
