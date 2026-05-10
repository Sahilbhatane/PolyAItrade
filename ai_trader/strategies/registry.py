"""Strategy plugin registry — maps YAML strategy names to voter callables."""

from __future__ import annotations

from typing import Any, Callable

from ai_trader.strategies.config_loader import StrategyConfig
from ai_trader.strategies.vote_types import VoteContext, VoteResult
from ai_trader.strategies.voters import (
    vote_fibonacci_confluence,
    vote_ma_crossover,
    vote_mean_reversion,
    vote_momentum_breakout,
    vote_rule_based_v1,
    vote_vwap_reversion,
)

VoterFn = Callable[[VoteContext, dict[str, Any]], VoteResult]

VOTER_REGISTRY: dict[str, VoterFn] = {
    "rule_based_v1": vote_rule_based_v1,
    "momentum_breakout": vote_momentum_breakout,
    "mean_reversion": vote_mean_reversion,
    "vwap_reversion": vote_vwap_reversion,
    "ma_crossover": vote_ma_crossover,
    "fibonacci_confluence": vote_fibonacci_confluence,
}


class StrategyRegistry:
    """Dispatch votes by registered strategy name."""

    def __init__(self, strategy_config: StrategyConfig):
        self._strategy_config = strategy_config

    def vote(self, name: str, ctx: VoteContext, params: dict[str, Any]) -> VoteResult:
        fn = VOTER_REGISTRY.get(name)
        if fn is None:
            return VoteResult(name=name, action="HOLD", confidence=0.0, reasoning=f"unknown_strategy:{name}")
        ctx.strategy_config = self._strategy_config
        return fn(ctx, params)
