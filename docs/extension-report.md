# PolyVITrade extension report (multi-agent + RL)

## Test report

- Command: `pytest --tb=short -q`
- Result: **207 passed** (runtime ~22s on reference machine)
- Network-marked tests excluded by default (`addopts = ["-m", "not network"]` in `pyproject.toml`)

## Architecture summary

- **Consensus pipeline (default):** `MarketData → Signal → RegimeDetection → StrategySelection → Consensus → Risk → Execution`, then optional `Reflection`.
- **Legacy pipeline:** set `agents.consensus_enabled: false` in `config.yaml` (or pass `agents: {consensus_enabled: false}` into `Orchestrator` config) to restore `StrategyAgent` in the chain.
- **RL:** `ai_trader/rl/*` trains offline PPO (`stable-baselines3`) and proposes bounded deltas via `StateKeys.RL_WEIGHT_PROPOSAL` only; **no RL module writes orders or risk verdicts**.
- **Config:** YAML loaded first, merged with `.env` / process env via `pydantic-settings`, then `AI_TRADER_*` nested overrides win. Production live Angel One requires API key, client id, and TOTP secret.

## Scripts

- `python scripts/startup_validation.py` — config + import smoke check
- `python scripts/health_diagnostics.py` — async timing for Signal / Regime / Selection / Consensus on synthetic data

## Known limitations / follow-ups

- RL `TradingEnv` is intentionally lightweight for reproducible CI; plug in replay-derived observations for production realism.
- `ConsensusAgent` requires **≥2** directional strategies agreeing — improves safety but may increase HOLD rate; tune `consensus_min_weighted_confidence` and voter thresholds in `strategy_config.yaml`.
- Strategy-selection does not yet apply `confidence_threshold_delta` from RL proposals to `StrategyAgent`/`ConsensusAgent` thresholds (weights only); extend via shared config or state if needed.
- Full-package `pkgutil.walk_packages` smoke test was avoided to prevent pulling optional/heavy side-effect imports.

## Performance notes

- Consensus path adds Regime + Selection + registry voting CPU overhead linear in enabled strategies; profile with `scripts/health_diagnostics.py`.
- SB3 PPO training is CPU-heavy; keep `ml_config.yaml` `rl.total_timesteps` modest for dev loops.

## Recommendations

1. Persist `CONSENSUS_AUDIT` and regime payloads to structured logs / DB for audit dashboards.
2. Add dedicated fixtures that monkeypatch `.env` path for isolated config tests.
3. Extend `test/integration` with `Orchestrator.run_pipeline` using `PaperBroker` and `consensus_enabled` toggles.
4. Consider Redis/external store for `KillSwitch` in multi-process deployments.
