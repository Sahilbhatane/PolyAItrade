"""RL — policy version, checkpoint, deployment mode."""

from __future__ import annotations

from textual.containers import Grid
from textual.widgets import Static

from ai_trader.tui.screens.base import Pane
from ai_trader.tui.store import AppStore
from ai_trader.tui.widgets.metric_card import MetricCard


class RLPane(Pane):
    pane_id = "rl"

    DEFAULT_CSS = """
    RLPane { padding: 1; }
    RLPane Grid { grid-size: 3; grid-gutter: 1; height: auto; }
    RLPane #rl-detail { height: 1fr; color: $text-muted; padding-top: 1; }
    """

    def compose(self):
        with Grid():
            yield MetricCard("Policy Version", help_key="regime", id="rl-policy")
            yield MetricCard("Checkpoint", help_key="regime", id="rl-checkpoint")
            yield MetricCard("Deployment", help_key="regime", id="rl-deploy")
            yield MetricCard("Seed", help_key="confidence", id="rl-seed")
            yield MetricCard("Checkpoint Dir", help_key="regime", id="rl-dir")
            yield MetricCard("Status", help_key="kill_switch", id="rl-status")
        yield Static("", id="rl-detail")

    @property
    def key_hints(self) -> str:
        return "F12 help  ^P palette"

    def refresh_from_store(self, store: AppStore) -> None:
        data = store.rl
        if not data:
            return

        exists = data.get("checkpoint_exists", False)
        mode = str(data.get("deployment_mode", "shadow")).upper()

        self.query_one("#rl-policy", MetricCard).set_metric(
            str(data.get("policy_version", "n/a")), "info"
        )
        self.query_one("#rl-checkpoint", MetricCard).set_metric(
            "FOUND" if exists else "MISSING", "ok" if exists else "warn"
        )
        self.query_one("#rl-deploy", MetricCard).set_metric(mode, _mode_state(mode))
        self.query_one("#rl-seed", MetricCard).set_metric(str(data.get("seed", "—")), "info")
        self.query_one("#rl-dir", MetricCard).set_metric(
            str(data.get("checkpoint_dir", "—"))[:24], "off"
        )
        self.query_one("#rl-status", MetricCard).set_metric(
            "READY" if exists else "NO MODEL", "ok" if exists else "crit"
        )

        detail = self.query_one("#rl-detail", Static)
        cfg = data.get("config", {})
        lines = [f"  {k}: {v}" for k, v in sorted(cfg.items()) if v is not None]
        detail.update("\n".join(lines) if lines else "No additional RL config.")


def _mode_state(mode: str) -> str:
    if mode == "LIVE":
        return "crit"
    if mode == "PAPER":
        return "warn"
    return "ok"
