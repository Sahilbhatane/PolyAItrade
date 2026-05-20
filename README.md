# PolyVITrade - Agentic AI Trading System

Production-grade, multi-agent AI trading system for Indian stock markets (NSE/BSE). Built with Python, FastAPI, PyTorch, and a modular agent architecture.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Architecture Overview](#architecture-overview)
3. [Setup Guide](#setup-guide)
4. [Configuration](#configuration)
5. [How To: Fetch Market Data](#how-to-fetch-market-data)
6. [How To: Backtest a Strategy](#how-to-backtest-a-strategy)
7. [How To: Configure Strategies](#how-to-configure-strategies)
8. [How To: Train the ML Model](#how-to-train-the-ml-model)
9. [How To: Run the Agent Pipeline](#how-to-run-the-agent-pipeline)
10. [How To: Go Live (Angel One)](#how-to-go-live-angel-one)
11. [How To: Approve/Reject Trades](#how-to-approvereject-trades)
12. [How To: Use the Kill Switch](#how-to-use-the-kill-switch)
13. [API Reference](#api-reference)
14. [Trading Rules (Enforced)](#trading-rules-enforced)
15. [Development Workflow](#development-workflow)
16. [Testing](#testing)
17. [What To Add Next](#what-to-add-next)

---

## What This Project Does

PolyVITrade is a fully automated (but human-approved) trading system that:

1. **Fetches and validates market data** (OHLCV from yfinance or broker)
2. **Generates signals** using technical indicators (RSI, MACD, VWAP, Moving Averages, Bollinger Bands, ATR)
3. **Predicts price movement** using an LSTM neural network (PyTorch)
4. **Combines signals into decisions** using a weighted rule engine
5. **Enforces risk controls** (position sizing, stop losses, daily limits, consecutive loss limits)
6. **Requires human approval** before placing any real order
7. **Executes trades** via Angel One SmartAPI (or paper broker for testing)
8. **Learns from past trades** via a ReflectionAgent that adjusts strategy weights
9. **Provides a kill switch** to immediately halt all trading

The system is designed so that **no trade is ever placed without passing through ALL safety gates**.

---

## Architecture Overview

```
ai_trader/
├── agents/           # Multi-agent pipeline
│   ├── base.py              # BaseAgent abstract class
│   ├── market_data_agent.py # Fetches + validates market data
│   ├── signal_agent.py      # Computes technical indicator signals
│   ├── strategy_agent.py    # Combines signals → BUY/SELL/HOLD
│   ├── risk_agent.py        # Enforces all risk controls (highest authority)
│   ├── execution_agent.py   # Paper execution with slippage simulation
│   ├── live_execution_agent.py  # Live execution with approval + kill switch
│   ├── reflection_agent.py  # Post-trade analysis + weight adjustment
│   ├── orchestrator.py      # Coordinates the full pipeline
│   ├── event_bus.py         # Async pub/sub for agent communication
│   └── state.py             # Central shared state (thread-safe)
├── models/           # ML prediction module
│   ├── lstm.py              # LSTM network + predictor
│   ├── features.py          # Feature engineering pipeline
│   ├── training.py          # Walk-forward training pipeline
│   ├── evaluation.py        # Accuracy, precision, recall, profit impact
│   └── config.py            # ML config loader
├── strategies/       # Rule-based strategy engine
│   ├── indicators.py        # RSI, MACD, VWAP, SMA, EMA, BB, ATR
│   ├── rule_engine.py       # Weighted signal combiner
│   ├── risk_controls.py     # Position sizing, cooldown, daily limits
│   └── config_loader.py     # Strategy YAML config
├── backtesting/      # Deterministic backtesting engine
│   ├── simulator.py         # Bar-by-bar trade simulation
│   ├── fees.py              # Brokerage, STT, GST, slippage model
│   ├── metrics.py           # Return, drawdown, Sharpe, profit factor
│   └── strategy.py          # BaseStrategy interface
├── data/             # Market data layer
│   ├── yfinance_provider.py # Historical data fetcher
│   ├── storage.py           # SQLite/PostgreSQL caching
│   └── provider.py          # Abstract data provider interface
├── broker/           # Broker integrations
│   ├── base.py              # Abstract broker interface
│   ├── paper.py             # Simulated broker (testing)
│   ├── angelone.py          # Angel One SmartAPI (live)
│   ├── approval.py          # Human approval gate
│   └── kill_switch.py       # Emergency trading halt
├── config/           # Configuration management
│   └── loader.py            # YAML + env var config loader
├── routes/           # FastAPI HTTP endpoints
│   ├── health.py            # Health checks
│   ├── data.py              # Data fetch endpoint
│   ├── backtest.py          # Backtest execution endpoint
│   ├── ml.py                # ML train/predict endpoints
│   └── trading.py           # Approvals, kill switch, positions, balance
├── logs/             # Structured JSON logging
│   └── logger.py
└── utils/            # Shared utilities
    ├── di.py                # Dependency injection container
    └── time.py              # IST time + market hours check
```

### Agent Pipeline Flow

```
MarketDataAgent → SignalAgent → StrategyAgent → RiskAgent → [Approval] → ExecutionAgent
                                                    ↑                          ↓
                                              (can VETO)              ReflectionAgent
                                                                     (adjusts weights)
```

---

## Setup Guide

### Prerequisites

- Python 3.12+
- Git
- (Optional) Angel One trading account for live trading

### Step 1: Clone and Create Virtual Environment

```bash
git clone https://github.com/Sahilbhatane/PolyAItrade.git
cd PolyAItrade

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

For live trading with Angel One, also install:
```bash
pip install smartapi-python pyotp websocket-client
```

### Step 3: Verify Installation

```bash
python -c "import ai_trader; print('OK')"
pytest --tb=short -q
```

### Step 4: Start the API Server

```bash
uvicorn ai_trader.app:create_app --factory --reload --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000`.

- Browser: open `http://localhost:8000/` (API info), `http://localhost:8000/docs` (Swagger UI), or `http://localhost:8000/health`
- CLI health check: `curl http://localhost:8000/health`

---

## Configuration

All config lives in three YAML files at the project root:

| File | Purpose |
|------|---------|
| `config.yaml` | Core app config (trading rules, database, broker, logging) |
| `strategy_config.yaml` | Strategy parameters (indicator thresholds, weights, risk) |
| `ml_config.yaml` | ML model architecture + training hyperparameters |

### Environment file (`.env`)

Copy `.env.example` to `.env` for local development. Flat keys (`APP_PORT`, `DATABASE_URL`, `ANGELONE_*`, API keys) are merged into YAML config **before** `AI_TRADER_*` overrides; secrets must never be committed.

Details: see `ai_trader/config/env.py` and `ConfigLoader.load()`.

### Environment Variable Overrides

Any config value can be overridden with environment variables using `AI_TRADER_` prefix and `__` separator:

```bash
# Override database URL
export AI_TRADER_DATABASE__URL="postgresql://user:pass@localhost/polyvitrade"

# Override max risk per trade
export AI_TRADER_TRADING__MAX_RISK_PER_TRADE=0.01

# Set broker to Angel One
export AI_TRADER_BROKER__NAME=angelone
export AI_TRADER_BROKER__API_KEY=your_api_key
export AI_TRADER_BROKER__CLIENT_ID=your_client_id
export AI_TRADER_BROKER__PASSWORD=your_password
export AI_TRADER_BROKER__TOTP_SECRET=your_totp_secret
```

### config.yaml — Key Sections

```yaml
trading:
  max_risk_per_trade: 0.02      # Max 2% capital per trade
  max_daily_loss: 0.05          # Stop if 5% daily loss
  max_consecutive_losses: 3     # Stop after 3 losses in a row
  confidence_threshold: 0.6     # Minimum confidence to trade
  market_open: "09:15"          # IST market open
  market_close: "15:30"         # IST market close

broker:
  name: paper                   # "paper" for testing, "angelone" for live
  max_retries: 3                # API call retry attempts
  retry_delay_s: 1.0            # Exponential backoff base delay
  product_type: INTRADAY        # INTRADAY or CNC (delivery)
  exchange: NSE                 # NSE or BSE

approval:
  enabled: true                 # Never disable in production
  timeout_s: 300.0              # 5 minutes to approve before auto-reject
  auto_approve_paper: true      # Only bypasses in paper mode

kill_switch:
  daily_loss_limit: 0.05        # Auto-engage if daily loss > 5%
  max_api_failures: 5           # Auto-engage after 5 consecutive API failures
```

---

## How To: Fetch Market Data

### Via API

```bash
curl -X POST http://localhost:8000/data/fetch \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE.NS",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "interval": "1d"
  }'
```

### Via Python

```python
from ai_trader.data.yfinance_provider import YFinanceProvider
from ai_trader.data.storage import MarketDataStore

provider = YFinanceProvider()
df = provider.fetch_historical("RELIANCE.NS", "2024-01-01", "2024-12-31")

# Cache to database
store = MarketDataStore("sqlite:///ai_trader.db")
store.save(df, symbol="RELIANCE.NS", interval="1d")

# Load from cache (fast)
cached = store.load("RELIANCE.NS", "1d", "2024-01-01", "2024-12-31")
```

### Supported Symbols (yfinance format)

- NSE: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`
- BSE: `RELIANCE.BO`, `TCS.BO`
- Indices: `^NSEI` (Nifty 50), `^BSESN` (Sensex)

### When To Fetch Data

- **Before backtesting**: Fetch sufficient historical data (at least 1 year recommended)
- **Before ML training**: Need 2+ years for walk-forward validation
- **Daily**: Fetch latest data before the trading session starts

---

## How To: Backtest a Strategy

### Via API

```bash
curl -X POST http://localhost:8000/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE.NS",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "strategy": "rule_based",
    "capital": 100000
  }'
```

### Via Python

```python
import pandas as pd
from ai_trader.backtesting.simulator import Simulator
from ai_trader.backtesting.fees import FeeModel
from ai_trader.strategies.rule_engine import RuleBasedStrategy

# Load your data
df = pd.read_csv("data.csv", parse_dates=["timestamp"])

# Create fee model (Indian market costs)
fees = FeeModel(
    brokerage_rate=0.0003,   # 0.03%
    stt_rate=0.00025,        # STT on sell
    gst_rate=0.18,           # 18% GST on brokerage
    slippage_rate=0.0001,    # 0.01% slippage
)

# Create strategy
strategy = RuleBasedStrategy(config_path="strategy_config.yaml")

# Run backtest
simulator = Simulator(capital=100000.0, fee_model=fees)
result = simulator.run(df, strategy)

# View results
print(f"Total Return: {result.metrics.total_return:.2%}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Win Rate: {result.metrics.win_rate:.2%}")
print(f"Profit Factor: {result.metrics.profit_factor:.2f}")
print(f"Total Trades: {result.metrics.total_trades}")
```

### Interpreting Results

| Metric | Good | Bad | Meaning |
|--------|------|-----|---------|
| Total Return | > 15% annually | < 0% | Overall profit/loss |
| Max Drawdown | < 10% | > 25% | Worst peak-to-trough drop |
| Sharpe Ratio | > 1.5 | < 0.5 | Risk-adjusted return |
| Win Rate | > 50% | < 40% | Percentage of profitable trades |
| Profit Factor | > 1.5 | < 1.0 | Gross profit / gross loss |

### When To Backtest

- After changing any strategy parameter
- After training a new ML model
- Before going live with any configuration
- Weekly, to validate strategy still works on recent data

---

## How To: Configure Strategies

### strategy_config.yaml — Full Reference

```yaml
name: rule_based_v1
version: "1.0.0"

indicators:
  rsi_period: 14              # RSI lookback period
  rsi_overbought: 70.0        # RSI > 70 = overbought (SELL signal)
  rsi_oversold: 30.0          # RSI < 30 = oversold (BUY signal)

  macd_fast: 12               # MACD fast EMA
  macd_slow: 26               # MACD slow EMA
  macd_signal: 9              # MACD signal line

  sma_fast: 10                # Fast moving average
  sma_slow: 30                # Slow moving average

  vwap_enabled: true          # Use VWAP (volume weighted average price)
  bollinger_period: 20        # Bollinger Band period
  bollinger_std: 2.0          # Bollinger Band std deviation

  atr_period: 14              # ATR for volatility-based stops
  atr_stop_multiplier: 2.0    # Stop loss = 2x ATR

risk:
  max_capital_per_trade: 0.02 # 2% max capital per trade
  stop_loss_pct: 0.03         # 3% hard stop loss
  trailing_stop_pct: 0.02     # 2% trailing stop
  daily_loss_limit: 0.05      # 5% daily max loss
  max_consecutive_losses: 3   # Stop after 3 consecutive losses

overtrading:
  max_trades_per_day: 5       # Maximum 5 trades per session
  cooldown_bars: 3            # Wait 3 bars between trades
  min_confidence: 0.5         # Don't trade below 50% confidence

weights:                      # How much each signal contributes (must sum to ~1.0)
  rsi: 0.25
  macd: 0.25
  ma_crossover: 0.25
  vwap: 0.25
  ml_prediction: 0.0          # Set > 0 to enable ML (after training)

ml:
  enabled: false              # Set true after training model
  model_path: "models/lstm_v1"  # Path to saved model
  sequence_length: 30
  confidence_boost: 0.1       # Boost confidence when ML agrees
```

### How Signal Combination Works

Each indicator produces a signal between -1 (strong SELL) and +1 (strong BUY). The strategy:

1. Computes all indicator signals
2. Multiplies each by its weight
3. Sums the weighted signals → composite score
4. If composite > `min_confidence` → BUY
5. If composite < -`min_confidence` → SELL
6. Otherwise → HOLD

### Tuning Tips

- **Conservative**: Increase `min_confidence` to 0.6-0.7, reduce `max_capital_per_trade` to 0.01
- **Aggressive**: Decrease `min_confidence` to 0.4, increase weights on trending indicators
- **ML-enhanced**: Set `ml.enabled: true` and give `ml_prediction` a weight of 0.15-0.25

---

## How To: Train the ML Model

### Step 1: Fetch Sufficient Data

You need at least 500+ trading days (2+ years) for meaningful walk-forward validation.

```bash
curl -X POST http://localhost:8000/data/fetch \
  -d '{"symbol": "RELIANCE.NS", "start_date": "2022-01-01", "end_date": "2024-12-31"}'
```

### Step 2: Train via API

```bash
curl -X POST http://localhost:8000/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE.NS",
    "start_date": "2022-01-01",
    "end_date": "2024-12-31",
    "save_path": "models/lstm_reliance"
  }'
```

### Step 3: Train via Python (more control)

```python
from ai_trader.models.training import TrainingPipeline, TrainingConfig
from ai_trader.data.yfinance_provider import YFinanceProvider

# Fetch data
provider = YFinanceProvider()
df = provider.fetch_historical("RELIANCE.NS", "2022-01-01", "2024-12-31")

# Configure training
config = TrainingConfig(
    epochs=50,
    batch_size=32,
    n_folds=3,           # Walk-forward folds
    min_train_size=200,  # Minimum bars in training window
)

# Train
pipeline = TrainingPipeline(config)
result = pipeline.run(df, save_path="models/lstm_reliance")

# View results per fold
for i, fold in enumerate(result.fold_results):
    print(f"Fold {i+1}: Accuracy={fold.accuracy:.3f}, AUC={fold.auc_roc:.3f}")

print(f"Average Accuracy: {result.avg_accuracy:.3f}")
```

### Step 4: Enable in Strategy

Edit `strategy_config.yaml`:
```yaml
weights:
  ml_prediction: 0.2  # Give ML 20% weight

ml:
  enabled: true
  model_path: "models/lstm_reliance"
```

### When To Retrain

- Every 60 trading days (configurable in `ml_config.yaml`)
- When strategy performance degrades noticeably
- After major market regime changes
- After any market structural changes (new regulations, etc.)

### What the ML Model Does (and Does NOT Do)

- **Does**: Outputs probability of price increase (0.0 to 1.0)
- **Does NOT**: Make trade decisions independently
- **Does NOT**: Override risk controls
- **Does NOT**: Execute trades
- It's one signal among many, weighted alongside RSI, MACD, etc.

---

## How To: Run the Agent Pipeline

### Paper Trading (Testing)

```python
import asyncio
from ai_trader.agents.orchestrator import Orchestrator
from ai_trader.agents.event_bus import EventBus
from ai_trader.agents.state import StateManager
from ai_trader.broker.paper import PaperBroker

async def main():
    bus = EventBus()
    state = StateManager()
    broker = PaperBroker(initial_balance=100_000.0)

    orchestrator = Orchestrator(
        event_bus=bus,
        state=state,
        broker=broker,
        config={
            "symbol": "RELIANCE.NS",
            "interval": "5m",
        }
    )

    result = await orchestrator.run_pipeline()
    print(f"Pipeline result: {result.status}")
    print(f"Trade decision: {state.read('trade_decision')}")
    print(f"Order result: {state.read('order_result')}")

asyncio.run(main())
```

### Live Trading (With Approval)

```python
import asyncio
from ai_trader.agents.orchestrator import Orchestrator
from ai_trader.agents.event_bus import EventBus
from ai_trader.agents.state import StateManager
from ai_trader.agents.live_execution_agent import LiveExecutionAgent
from ai_trader.broker.angelone import AngelOneBroker
from ai_trader.broker.approval import ApprovalGate
from ai_trader.broker.kill_switch import KillSwitch

async def main():
    bus = EventBus()
    state = StateManager()

    broker = AngelOneBroker(
        api_key="your_key",
        client_id="your_id",
        password="your_pass",
        totp_secret="your_totp",
    )

    approval_gate = ApprovalGate(timeout_s=300)  # 5 min to approve
    kill_switch = KillSwitch(auto_triggers={"daily_loss_limit": 0.05})

    agent = LiveExecutionAgent(
        event_bus=bus,
        state=state,
        broker=broker,
        approval_gate=approval_gate,
        kill_switch=kill_switch,
    )

    # The pipeline will BLOCK at approval gate
    # Use the /trading/approvals/respond API to approve/reject

asyncio.run(main())
```

---

## How To: Go Live (Angel One)

### Step 1: Get Angel One API Credentials

1. Open an Angel One account at [angelone.in](https://www.angelone.in)
2. Go to SmartAPI portal: [smartapi.angelone.in](https://smartapi.angelone.in)
3. Create an app → get API Key
4. Note your Client ID, Password
5. Enable TOTP → get TOTP secret

### Step 2: Configure

Edit `config.yaml`:
```yaml
broker:
  name: angelone
  api_key: "YOUR_API_KEY"
  client_id: "YOUR_CLIENT_ID"
  password: "YOUR_PASSWORD"
  totp_secret: "YOUR_TOTP_SECRET"
  product_type: INTRADAY    # or CNC for delivery
  exchange: NSE

approval:
  auto_approve_paper: false  # NEVER auto-approve in live mode
  timeout_s: 300
```

Or use environment variables (preferred for security):
```bash
export AI_TRADER_BROKER__NAME=angelone
export AI_TRADER_BROKER__API_KEY=your_key
export AI_TRADER_BROKER__CLIENT_ID=your_id
export AI_TRADER_BROKER__PASSWORD=your_pass
export AI_TRADER_BROKER__TOTP_SECRET=your_secret
```

### Step 3: Start Server and Monitor

```bash
uvicorn ai_trader.app:create_app --factory --host 0.0.0.0 --port 8000
```

### Step 4: Verify Connectivity

```bash
curl http://localhost:8000/trading/health
# → {"healthy": true}

curl http://localhost:8000/trading/balance
# → {"cash": 500000.0, "margin_used": 0.0, ...}
```

### Step 5: Monitor & Approve Trades

When the pipeline proposes a trade, it will appear in pending approvals:
```bash
curl http://localhost:8000/trading/approvals/pending
```

Approve or reject via API (see next section).

---

## How To: Approve/Reject Trades

### View Pending Trades

```bash
curl http://localhost:8000/trading/approvals/pending
```

Response:
```json
[
  {
    "request_id": "abc-123",
    "trade_details": {
      "symbol": "RELIANCE-EQ",
      "side": "BUY",
      "quantity": 10,
      "price": 1500.0,
      "stop_loss": 1470.0,
      "confidence": 0.72
    },
    "created_at": "2024-01-15T09:30:00Z"
  }
]
```

### Approve a Trade

```bash
curl -X POST http://localhost:8000/trading/approvals/respond \
  -H "Content-Type: application/json" \
  -d '{"request_id": "abc-123", "action": "approve"}'
```

### Reject a Trade

```bash
curl -X POST http://localhost:8000/trading/approvals/respond \
  -H "Content-Type: application/json" \
  -d '{"request_id": "abc-123", "action": "reject", "reason": "Market too volatile"}'
```

### View Approval History

```bash
curl http://localhost:8000/trading/approvals/history
```

### Approval Timeout

If you don't respond within `timeout_s` (default 300 seconds), the trade is automatically **rejected**. No trade is ever placed without explicit approval.

---

## How To: Use the Kill Switch

The kill switch immediately halts ALL trading activity.

### Engage (Emergency Stop)

```bash
curl -X POST http://localhost:8000/trading/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"action": "engage", "reason": "Market crash detected"}'
```

### Check Status

```bash
curl http://localhost:8000/trading/kill-switch/status
# → {"active": true, "reason": "Market crash detected", "triggered_by": "api_user"}
```

### Disengage (Resume Trading)

```bash
curl -X POST http://localhost:8000/trading/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"action": "disengage"}'
```

### Automatic Triggers

The kill switch auto-engages when:
- **Daily loss exceeds limit** (default 5%)
- **API failures exceed threshold** (default 5 consecutive failures)

These are configured in `config.yaml` under `kill_switch:`.

### View History

```bash
curl http://localhost:8000/trading/kill-switch/history
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| GET | `/ready` | Readiness probe |
| POST | `/data/fetch` | Fetch historical market data |
| POST | `/backtest/run` | Run a backtest |
| POST | `/ml/train` | Train the ML model |
| POST | `/ml/predict` | Get ML prediction |
| GET | `/trading/approvals/pending` | Pending trade approvals |
| POST | `/trading/approvals/respond` | Approve/reject a trade |
| GET | `/trading/approvals/history` | Approval audit log |
| GET | `/trading/kill-switch/status` | Kill switch state |
| POST | `/trading/kill-switch` | Engage/disengage kill switch |
| GET | `/trading/kill-switch/history` | Kill switch event log |
| GET | `/trading/positions` | Open positions from broker |
| GET | `/trading/balance` | Account balance |
| GET | `/trading/health` | Broker connectivity check |

---

## Trading Rules (Enforced)

These rules are enforced by the system — they cannot be bypassed:

| Rule | Enforced By | Config Key |
|------|-------------|------------|
| Max 2% capital per trade | RiskAgent | `trading.max_risk_per_trade` |
| Stop after 5% daily loss | RiskAgent + KillSwitch | `trading.max_daily_loss` |
| Stop after 3 consecutive losses | RiskAgent | `trading.max_consecutive_losses` |
| Trade only 9:15-15:30 IST | MarketDataAgent | `trading.market_open/close` |
| Min confidence threshold | StrategyAgent | `trading.confidence_threshold` |
| Max 5 trades per day | RiskController | `overtrading.max_trades_per_day` |
| Cooldown between trades | RiskController | `overtrading.cooldown_bars` |
| Stop loss mandatory | RiskAgent | `risk.stop_loss_pct` |
| Slippage always simulated | ExecutionAgent | `backtest.slippage_rate` |
| Human approval required | ApprovalGate | `approval.enabled` |
| No trade without data validation | MarketDataAgent | Always enforced |
| No trade on API failure | KillSwitch | `kill_switch.max_api_failures` |

---

## Development Workflow

### Phase Order (Follow This)

```
1. Data          → Fetch and validate market data
2. Backtesting   → Test strategy on historical data
3. Strategy      → Tune indicators and weights
4. ML            → Train model, validate with backtest
5. Agents        → Run full pipeline in paper mode
6. Live          → Deploy with approval gate
```

### Daily Workflow (When Live)

```
08:30  Start server, verify broker health
09:00  Fetch latest data, check kill switch status
09:15  Pipeline starts generating signals
09:15+ Monitor pending approvals, approve/reject
15:30  Market close — review day's trades
15:30+ ReflectionAgent evaluates all trades
       Check if weight adjustments are proposed
```

### Weekly Workflow

```
- Review ReflectionAgent reports
- Check if ML model needs retraining
- Backtest current config on latest data
- Verify all tests still pass
- Review kill switch history
```

---

## Testing

### Run All Tests

```bash
pytest --tb=short -q
```

### Run Specific Test Modules

```bash
# Broker integration tests
pytest tests/test_broker_integration.py -v

# Strategy tests
pytest tests/test_strategy.py -v

# ML model tests
pytest tests/test_ml.py -v

# Backtesting engine tests
pytest tests/test_backtesting.py -v

# Agent pipeline tests
pytest tests/test_agents_pipeline.py -v

# Reflection agent tests
pytest tests/test_reflection.py -v
```

### Current Test Count: 207 tests, all passing

---

## What To Add Next

### High Priority (Recommended Order)

1. **WebSocket notifications** — Push trade proposals to a frontend instead of polling
2. **Dashboard UI** — React/Next.js frontend for approvals, monitoring, P&L tracking
3. **Telegram/Discord bot** — Approve trades from your phone
4. **Multiple symbol support** — Run pipeline on a watchlist simultaneously
5. **Scheduled pipeline runner** — Cron/APScheduler to run pipeline every N minutes during market hours

### Medium Priority

6. **PostgreSQL migration** — Move from SQLite to PostgreSQL for production
7. **Redis caching** — Fast state sharing for distributed deployment
8. **Options support** — Extend strategy for options trading
9. **News sentiment agent** — Scrape and analyze market news (NLP)
10. **Portfolio-level risk** — Correlation-based position sizing across multiple holdings

### Lower Priority / Future

11. **LLM integration** — Use GPT/Claude for trade reasoning explanation
12. **Reinforcement learning** — Replace/supplement LSTM with RL agent
13. **Multi-broker support** — Zerodha (Kite), Upstox, IIFL alongside Angel One
14. **Docker deployment** — Containerize for cloud deployment
15. **Prometheus/Grafana** — Metrics and alerting stack

### When Adding New Features

- Always write tests first (or alongside)
- Add YAML config for any new parameters
- Follow the agent interface pattern (inherit `BaseAgent`)
- Never let a new component bypass the risk pipeline
- Every new module must be independently runnable
- Each feature gets its own commit

---

## Security Notes

- **Never commit credentials** — Use environment variables or `.env` files
- **Never disable approval in production** — `auto_approve_paper: true` is for paper mode ONLY
- **TOTP secrets are sensitive** — Store in a secrets manager in production
- **API keys should be rotated** periodically
- **The kill switch is your last line of defense** — Never remove it

---

## License

Private / Not yet open sourced.
