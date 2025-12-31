
# Efficient Frontier Project
# ECO 43000 – Quantitative Finance (Fall 2025)
# Professor: John Droescher
# Students: Serenity Sheppard, Riya Dcosta, Reina Osorio

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def main():
    # ---------------- PARAMETERS ----------------
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "V", "PFE", "DIS"]
    benchmarks = ["DIA", "SPY", "IWM"]
    start_date = "2015-01-01"

    # ---------------- DATA ----------------
    prices = yf.download(tickers, start=start_date)["Adj Close"]
    returns = prices.pct_change().dropna()

    benchmark_prices = yf.download(benchmarks, start=start_date)["Adj Close"]
    benchmark_returns = benchmark_prices.pct_change().dropna()

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # ---------------- FUNCTIONS ----------------
    def portfolio_performance(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret / vol
        return ret, vol, sharpe

    def min_variance():
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * len(tickers)
        result = minimize(lambda w: portfolio_performance(w)[1],
                          len(tickers) * [1 / len(tickers)],
                          bounds=bounds,
                          constraints=constraints)
        return result.x

    def max_sharpe():
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * len(tickers)
        result = minimize(lambda w: -portfolio_performance(w)[2],
                          len(tickers) * [1 / len(tickers)],
                          bounds=bounds,
                          constraints=constraints)
        return result.x

    # ---------------- EFFICIENT FRONTIER ----------------
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        ret, vol, sharpe = portfolio_performance(weights)
        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = sharpe

    min_w = min_variance()
    max_w = max_sharpe()

    min_perf = portfolio_performance(min_w)
    max_perf = portfolio_performance(max_w)

    plt.figure(figsize=(10, 6))
    plt.scatter(results[0], results[1], c=results[2], cmap="viridis", s=3)
    plt.colorbar(label="Sharpe Ratio")
    plt.scatter(min_perf[1], min_perf[0], color="red", marker="*", s=200, label="Min Variance")
    plt.scatter(max_perf[1], max_perf[0], color="blue", marker="*", s=200, label="Max Sharpe")
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier – 7 NYSE Stocks")
    plt.legend()
    plt.show()

    # ---------------- CUMULATIVE RETURNS ----------------
    portfolio_returns = returns.dot(max_w)
    cum_portfolio = (1 + portfolio_returns).cumprod()
    cum_benchmarks = (1 + benchmark_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cum_portfolio, label="Optimal Portfolio")
    for b in benchmarks:
        plt.plot(cum_benchmarks[b], label=b)
    plt.legend()
    plt.title("Cumulative Growth of $1")
    plt.show()


if __name__ == "__main__":
    main()
