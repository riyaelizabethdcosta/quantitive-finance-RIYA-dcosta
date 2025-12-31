
# QQQ Momentum Strategy
# ECO 43000 – Quantitative Finance
# Professor: John Droescher
# Students: Serenity Sheppard, Riya Dcosta, Reina Osorio

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def main():
    # ---------------- PARAMETERS ----------------
    start_date = "2020-01-01"
    lookbacks = {
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "12M": 252
    }
    long_n = 15
    short_n = 15

    # ---------------- DATA ----------------
    qqq = yf.Ticker("QQQ")
    tickers = qqq.funds_data.top_holdings.index.tolist()

    prices = yf.download(tickers, start=start_date)["Adj Close"]
    returns = prices.pct_change()

    # ---------------- MOMENTUM FACTORS ----------------
    momentum = pd.DataFrame(index=tickers)

    for label, days in lookbacks.items():
        momentum[label] = prices.pct_change(days).iloc[-1]

    z_scores = (momentum - momentum.mean()) / momentum.std()
    momentum["Total Score"] = z_scores.sum(axis=1)
    momentum["Rank"] = momentum["Total Score"].rank(ascending=False)

    momentum["Basket"] = "Neutral"
    momentum.loc[momentum["Rank"] <= long_n, "Basket"] = "Long"
    momentum.loc[momentum["Rank"] > len(momentum) - short_n, "Basket"] = "Short"

    # ---------------- BACKTEST ----------------
    long_returns = returns[momentum[momentum["Basket"] == "Long"].index].mean(axis=1)
    short_returns = returns[momentum[momentum["Basket"] == "Short"].index].mean(axis=1)

    strategy_returns = long_returns - short_returns
    qqq_returns = yf.download("QQQ", start=start_date)["Adj Close"].pct_change()

    cum_strategy = (1 + strategy_returns).cumprod()
    cum_qqq = (1 + qqq_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cum_strategy, label="Momentum Strategy")
    plt.plot(cum_qqq, label="QQQ Benchmark", linestyle="--")
    plt.legend()
    plt.title("Cumulative Returns: Momentum Strategy vs QQQ")
    plt.show()


if __name__ == "__main__":
    main()
