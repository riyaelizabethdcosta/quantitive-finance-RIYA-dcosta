"""
Portfolio Analysis Project (Quantitative Finance - ECO 43000)
Group: RIYA, SERENITY, & JULIANA

Goal:
- Build a portfolio of 7 NYSE stocks and compare it to DIA, SPY, IWM.
Tools:
- pandas, yfinance, matplotlib
Metrics:
- Volatility, Beta, Tracking Error, Sharpe Ratio
Period:
- 2015-01-01 to 2025-12-31

Outputs (saved to ./outputs/):
- table1_asset_factors.csv
- table2_portfolio_vs_benchmark.csv
- cumulative_growth.png
- adjusted_prices_selected_assets.png
- daily_returns_first_90_days.png
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    start: str = "2015-01-01"
    end: str = "2025-12-31"
    assets: List[str] = None
    benchmarks: List[str] = None
    risk_free_annual: float = 0.02
    trading_days: int = 252
    out_dir: str = "outputs"

    def __post_init__(self):
        if self.assets is None:
            self.assets = ['AAPL', 'MSFT', 'JPM', 'XOM', 'V', 'PFE', 'DIS']
        if self.benchmarks is None:
            self.benchmarks = ['DIA', 'SPY', 'IWM']


CFG = Config()


# -----------------------------
# Helpers
# -----------------------------
def download_adj_close(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download Adjusted Close prices for the given tickers."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all")
    return df


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns."""
    return prices.pct_change().dropna()


def annualize_vol(daily_returns: pd.Series, trading_days: int = 252) -> float:
    return float(daily_returns.std() * np.sqrt(trading_days))


def annualize_return(daily_returns: pd.Series, trading_days: int = 252) -> float:
    return float(daily_returns.mean() * trading_days)


def sharpe_ratio(daily_returns: pd.Series, rf_annual: float = 0.02, trading_days: int = 252) -> float:
    rf_daily = rf_annual / trading_days
    excess = daily_returns - rf_daily
    if excess.std() == 0:
        return np.nan
    return float((excess.mean() / excess.std()) * np.sqrt(trading_days))


def beta(asset_ret: pd.Series, benchmark_ret: pd.Series) -> float:
    """Beta = Cov(asset, benchmark) / Var(benchmark)."""
    aligned = pd.concat([asset_ret, benchmark_ret], axis=1).dropna()
    a = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    var_b = b.var()
    if var_b == 0:
        return np.nan
    return float(a.cov(b) / var_b)


def tracking_error(port_ret: pd.Series, bench_ret: pd.Series, trading_days: int = 252) -> float:
    """Annualized tracking error: std(port - benchmark) * sqrt(252)."""
    aligned = pd.concat([port_ret, bench_ret], axis=1).dropna()
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(diff.std() * np.sqrt(trading_days))


def make_equal_weights(tickers: List[str]) -> Dict[str, float]:
    w = 1.0 / len(tickers)
    return {ticker: w for ticker in tickers}


def portfolio_returns(asset_returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    w = pd.Series(weights).reindex(asset_returns.columns)
    w = w / w.sum()
    return (asset_returns * w).sum(axis=1)


def ensure_out_dir(path_str: str) -> Path:
    p = Path(path_str)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# Main
# -----------------------------
def main():
    out_dir = ensure_out_dir(CFG.out_dir)

    tickers_all = CFG.assets + CFG.benchmarks
    prices = download_adj_close(tickers_all, CFG.start, CFG.end)

    # Split prices
    asset_prices = prices[CFG.assets].dropna(how="all")
    bench_prices = prices[CFG.benchmarks].dropna(how="all")

    # Returns
    asset_ret = to_returns(asset_prices)
    bench_ret = to_returns(bench_prices)

    # Align dates
    common_index = asset_ret.index.intersection(bench_ret.index)
    asset_ret = asset_ret.loc[common_index]
    bench_ret = bench_ret.loc[common_index]

    # Weights (equal-weight)
    weights = make_equal_weights(CFG.assets)

    # Portfolio return series
    port_ret = portfolio_returns(asset_ret, weights)
    port_ret.name = "Portfolio"

    # -----------------------------
    # Table 1: Asset Factors
    # -----------------------------
    table1 = pd.DataFrame(index=CFG.assets)
    table1["Annual Volatility"] = [annualize_vol(asset_ret[t], CFG.trading_days) for t in CFG.assets]

    for b in CFG.benchmarks:
        table1[f"Beta vs {b}"] = [beta(asset_ret[t], bench_ret[b]) for t in CFG.assets]

    # Returns: 1Y and 3Y (annualized from daily mean over lookback)
    last_1y = asset_ret.tail(CFG.trading_days)
    last_3y = asset_ret.tail(CFG.trading_days * 3)

    table1["Return 1Y (ann.)"] = [annualize_return(last_1y[t], CFG.trading_days) for t in CFG.assets]
    table1["Return 3Y (ann.)"] = [annualize_return(last_3y[t], CFG.trading_days) for t in CFG.assets]

    table1_path = out_dir / "table1_asset_factors.csv"
    table1.to_csv(table1_path)

    # -----------------------------
    # Table 2: Portfolio vs Benchmark ETF
    # -----------------------------
    table2 = pd.DataFrame(index=CFG.benchmarks)
    vol_port = annualize_vol(port_ret, CFG.trading_days)
    sharpe_port = sharpe_ratio(port_ret, CFG.risk_free_annual, CFG.trading_days)

    for b in CFG.benchmarks:
        aligned = pd.concat([port_ret, bench_ret[b]], axis=1).dropna()
        table2.loc[b, "Correlation"] = aligned.corr().iloc[0, 1]
        table2.loc[b, "Covariance"] = aligned.cov().iloc[0, 1]
        table2.loc[b, "Tracking Error (ann.)"] = tracking_error(port_ret, bench_ret[b], CFG.trading_days)
        table2.loc[b, "Portfolio Sharpe (ann.)"] = sharpe_port
        table2.loc[b, f"{b} Sharpe (ann.)"] = sharpe_ratio(bench_ret[b], CFG.risk_free_annual, CFG.trading_days)

        vol_b = annualize_vol(bench_ret[b], CFG.trading_days)
        table2.loc[b, "Volatility Spread (Port - Bench)"] = vol_port - vol_b

    table2_path = out_dir / "table2_portfolio_vs_benchmark.csv"
    table2.to_csv(table2_path)

    # -----------------------------
    # Charts
    # -----------------------------
    # 1) Cumulative Growth of $1: Portfolio vs DIA/SPY/IWM
    cum = (1 + pd.concat([port_ret, bench_ret], axis=1).dropna()).cumprod()
    plt.figure(figsize=(10, 5))
    plt.plot(cum.index, cum["Portfolio"], label="Portfolio")
    for b in CFG.benchmarks:
        plt.plot(cum.index, cum[b], label=b)
    plt.title("Cumulative Growth of $1: Portfolio vs DIA / SPY / IWM")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_growth.png", dpi=200)
    plt.close()

    # 2) Adjusted Prices: Selected Assets (2015-2025)
    plt.figure(figsize=(10, 5))
    for ticker in CFG.assets:
        plt.plot(asset_prices.index, asset_prices[ticker], label=ticker)
    # Include SPY like the slide example (optional)
    if "SPY" in bench_prices.columns:
        plt.plot(bench_prices.index, bench_prices["SPY"], label="SPY")
    plt.title("Adjusted Prices: Selected Assets (2015-2025)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "adjusted_prices_selected_assets.png", dpi=200)
    plt.close()

    # 3) Daily Returns (First 90 Trading Days, Selected Assets)
    first_90 = asset_ret.head(90).copy()
    if "SPY" in bench_ret.columns:
        first_90["SPY"] = bench_ret["SPY"].head(90)

    plt.figure(figsize=(10, 5))
    for col in first_90.columns:
        plt.plot(first_90.index, first_90[col], label=col)
    plt.title("Daily Returns (First 90 Trading Days, Selected Assets)")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "daily_returns_first_90_days.png", dpi=200)
    plt.close()

    # -----------------------------
    # Console summary
    # -----------------------------
    print("\nSaved outputs to:", out_dir.resolve())
    print("\nTable 1 saved to:", table1_path.resolve())
    print("Table 2 saved to:", table2_path.resolve())
    print("Charts saved to:", (out_dir / "cumulative_growth.png").resolve())
    print("\nPortfolio (equal-weight) summary:")
    print("Annualized Return:", round(annualize_return(port_ret, CFG.trading_days), 4))
    print("Annualized Volatility:", round(vol_port, 4))
    print("Sharpe (ann.):", round(sharpe_port, 4))


if __name__ == "__main__":
    main()
