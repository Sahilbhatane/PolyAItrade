import pytest

from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.rl.optimizer import RLOptimizer


def test_optimizer_validates_bounds():
    opt = RLOptimizer({"max_weight_delta": 0.1})
    bad = {"strategy_weights": {"a": 0.5}, "confidence_threshold_delta": 0.0}
    with pytest.raises(ValueError):
        opt.validate_payload(bad)


def test_optimizer_writes_proposal_sync():
    st = StateManager()
    opt = RLOptimizer({"max_weight_delta": 0.15})
    opt.propose_from_checkpoint(st, checkpoint_stem="missing_model")
    prop = st.read(StateKeys.RL_WEIGHT_PROPOSAL)
    assert prop is not None
    assert "strategy_weights" in prop
