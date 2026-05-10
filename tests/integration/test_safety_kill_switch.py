from ai_trader.broker.kill_switch import KillSwitch


def test_kill_switch_volatility_and_runaway():
    ks = KillSwitch(
        auto_triggers={
            "volatility_spike_zscore": 3.0,
            "runaway_loss_pct": 0.1,
        }
    )
    ks.check_volatility_spike(5.0)
    assert ks.is_active
    ks.disengage()
    ks.check_runaway_loss(0.11)
    assert ks.is_active
