# PolyVITrade - Agentic AI Trading System

Production-grade, multi-agent AI trading system built with Python, FastAPI, and modular architecture.

## Architecture

```
ai_trader/
├── agents/        # Multi-agent system (BaseAgent, communication protocol)
├── models/        # ML model interfaces (signal generation, NOT decisions)
├── backtesting/   # Strategy evaluation engine (deterministic, cost-aware)
├── data/          # Market data providers (validated, timestamped)
├── broker/        # Broker integrations (order execution only)
├── config/        # YAML-based config with env var overrides
├── logs/          # Structured logging (JSON, contextual)
├── routes/        # FastAPI HTTP endpoints
└── utils/         # DI container, time utilities
```

## Design Principles

1. **Separation of Concerns** - Each module owns exactly one responsibility
2. **Interface-First** - All modules expose abstract base classes; implementations are swappable
3. **Dependency Injection** - No module directly instantiates its dependencies
4. **Config-Driven** - Zero hardcoded values; everything flows from `config.yaml` + env vars
5. **Fail-Safe** - Missing data or errors result in no-trade, never silent failures
6. **Audit Trail** - Every decision is logged with inputs, reasoning, and confidence

## Agent Pipeline

Trades flow through a strict pipeline — no shortcuts allowed:

```
MarketDataAgent → SignalAgent → StrategyAgent → RiskAgent → ExecutionAgent
                                                     ↑
                                              (can veto any trade)
```

- **MarketDataAgent** validates and timestamps all incoming data
- **SignalAgent** outputs probabilities, never binary decisions
- **StrategyAgent** combines signals into trade proposals with confidence scores
- **RiskAgent** enforces capital limits, stop losses, and drawdown constraints (highest authority)
- **ExecutionAgent** places only approved orders through the broker interface
- **ReflectionAgent** reviews outcomes and proposes bounded strategy updates

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn ai_trader.app:create_app --factory --reload

# Run tests
pytest
```

## Configuration

Configuration loads from `config.yaml` at the project root. Override any value with environment variables using the `AI_TRADER_` prefix and `__` as a nesting separator:

```bash
export AI_TRADER_DATABASE__URL="postgresql://user:pass@localhost/polyvitrade"
export AI_TRADER_TRADING__MAX_RISK_PER_TRADE=0.01
```

## Trading Rules (Enforced by System)

- Max 1-2% capital risk per trade
- Stop trading if daily loss exceeds 3-5%
- Stop after 3 consecutive losses
- Trade only during market hours (9:15 AM - 3:30 PM IST)
- Confidence threshold must be met before execution
- Trailing stop loss is mandatory
- Slippage simulation always included

## Testing

```bash
pytest --tb=short -q
```

All modules are independently testable. The DI container can be reset between tests for full isolation.
