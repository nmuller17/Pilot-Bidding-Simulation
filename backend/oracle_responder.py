"""
oracle_responder.py
-------------------
A scripted stand-in for the LLM that answers every column-bid prompt from the
oracle bid. Used with llm_client.StubClient for dry runs and tests: if the
pipeline is wired correctly, a separated or joint budget bid under any regime
reproduces the oracle ranking exactly (rho = 1, selection F1 = 1, weight
error 0). Makes no API calls.

It reads the "Stage:", "Name: Capt.", "Budget:" and "Columns to weight:" lines
that bid_prompts puts in every prompt.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from bid import normalise_to_budget


class OracleResponder:
    def __init__(self, instance):
        self.instance = instance
        self._by_name = {p.name: p for p in instance.pilots}

    def __call__(self, prompt: str) -> str:
        stage = re.search(r"^Stage: (\w+)", prompt, re.M).group(1)
        name = re.search(r"^Name: Capt\. (.+)$", prompt, re.M).group(1).strip()
        bid = self.instance.oracle_bid(self._by_name[name])
        rank_format = '"ranking"' in prompt or '"rank":' in prompt

        if stage == "selection":
            return json.dumps({"selected": self._selected(bid)})

        m = re.search(r"^Budget: ([\d.]+)", prompt, re.M)
        budget = float(m.group(1)) if m else self.instance.budget   # rank prompts state none
        if stage == "joint":
            items = self._selected(bid)
            scored = [it["key"] for it in items if it["key"] in bid.weights]
            w = self._weights(bid, scored, budget)
            order = sorted(scored, key=lambda k: -w[k])
            for it in items:
                if it["key"] in w:
                    if rank_format:
                        it["rank"] = order.index(it["key"]) + 1
                    else:
                        it["weight"] = w[it["key"]]
            return json.dumps({"selected": items})

        keys = [k.strip() for k in
                re.search(r"^Columns to weight: (.*)$", prompt, re.M).group(1).split(",")]
        w = self._weights(bid, keys, budget)
        if stage == "increment":
            return json.dumps({"increment": w})
        if rank_format:
            return json.dumps({"ranking": sorted(keys, key=lambda k: -w[k])})
        return json.dumps({"weights": w})

    # ------------------------------------------------------------------
    @staticmethod
    def _selected(bid) -> List[dict]:
        items = [{"key": k, "direction": bid.directions[k], "hard_exclusion": False,
                  "rationale": "oracle"} for k in bid.weights]
        items += [{"key": k, "direction": -1, "hard_exclusion": True, "rationale": "oracle"}
                  for k in bid.hard_exclusions]
        items += [{"key": k, "rationale": "oracle"} for k in bid.schedule_prefs]
        return items

    @staticmethod
    def _weights(bid, keys: List[str], budget: float) -> Dict[str, float]:
        return normalise_to_budget({k: bid.weights.get(k, 0.0) for k in keys}, budget)
