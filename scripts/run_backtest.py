"""
Run a backtest and print results.
Usage: python -m scripts.run_backtest [--tickers TSLA AAPL] [--period 2y]
"""
import argparse
import json
import structlog

from backtest.backtester import Backtester

logger = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LedgerGPT backtest")
    parser.add_argument("--tickers", nargs="+", default=["TSLA", "AAPL", "NVDA"])
    parser.add_argument("--period", default="2y")
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    backtester = Backtester(initial_cash=args.cash, commission=args.commission)
    results = backtester.run(
        tickers=args.tickers,
        period=args.period,
        strategy_params={"min_confidence": args.min_confidence},
        verbose=args.verbose,
    )

    # Save results
    import json
    from pathlib import Path
    out_path = Path("artifacts/backtest_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("backtest_results_saved", path=str(out_path))


if __name__ == "__main__":
    main()
