import numpy as np

from ai_trader.rl.env import TradingEnv


def test_trading_env_reset_step():
    env = TradingEnv({"seed": 0, "n_strategies": 6})
    obs, _ = env.reset(seed=0)
    assert obs.shape == (16,)
    obs2, rew, term, trunc, info = env.step(np.zeros(7, dtype=np.float32))
    assert obs2.shape == (16,)
    assert isinstance(rew, float)
    assert term is False and trunc is False
    assert "weights" in info
