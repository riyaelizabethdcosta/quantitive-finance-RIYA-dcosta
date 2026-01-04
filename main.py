# ECO 43000 Final Project - Momentum Strategy
# Jan 2026
# Testing momentum trading on Nasdaq-100 stocks
# Group Members: Riya, Serenity & Reina

import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt

# create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Settings
YEARS_BACK = 5
LONG_N = 15
SHORT_N = 15

print("Getting ticker list...")

# Fixed Nasdaq-100 style universe (100+ stocks)
# Using a stable large-cap universe instead of scraping holdings
TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","GOOG","NVDA","META","TSLA","AVGO","COST",
    "NFLX","ADBE","PEP","CSCO","TMUS","CMCSA","INTC","QCOM","AMD","TXN",
    "INTU","AMAT","SBUX","ISRG","BKNG","GILD","MDLZ","ADP","REGN","MU",
    "LRCX","KLAC","SNPS","CDNS","ORLY","MAR","PANW","PYPL","ABNB","WDAY",
    "CSX","MNST","ROST","IDXX","BIIB","DXCM","HON","VRTX","MELI","ASML",
    "KDP","CRWD","FTNT","NXPI","PCAR","AEP","AZN","MRNA","ODFL","MRVL",
    "TEAM","CTSH","FAST","EA","XEL","ILMN","VRSK","ZS","PAYX","CHTR",
    "DLTR","CPRT","ANSS","CEG","DDOG","SIRI","BKR","GEHC","FANG","ROKU",
    "PDD","JD","NTES","BIDU","LCID","RIVN","ENPH","EXC","WBA","CSGP",
    "GFS","ON","MDB","NET","TTD","APP","ARM","SMCI","ADI","MCHP",
    "LULU","EBAY","AAL","UAL","DAL","NKE","KR","WMT"
]

# remove duplicates just in case
TICKERS = sorted(list(set(TICKERS)))
# Remove tickers that fail on Yahoo
BAD = {"WBA", "ANSS"}
TICKERS = [t for t in TICKERS if t not in BAD]

print(f"Using {len(TICKERS)} stocks")


# Download data
print("Downloading price data (takes a minute)...")
end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(days=365 * YEARS_BACK)

raw_data = yf.download(
    TICKERS,
    start=start_date,
    end=end_date,
    progress=False,
    group_by="ticker",
    auto_adjust=False,
    threads=True
)


# ---- Extract Adj Close prices into a clean DataFrame (date x ticker) ----
prices_list = []

for t in TICKERS:
    if t in raw_data.columns.get_level_values(0):
        # raw_data[t] is a DataFrame with columns like Open/High/Low/Close/Adj Close/Volume
        df_t = raw_data[t]
        col = "Adj Close" if "Adj Close" in df_t.columns else "Close"
        prices_list.append(df_t[[col]].rename(columns={col: t}))

prices = pd.concat(prices_list, axis=1)

# Drop tickers that came back completely empty
prices = prices.dropna(how="all", axis=1)


# Clean data
prices = prices.dropna(thresh=int(len(prices) * 0.5), axis=1)
prices = prices.ffill().bfill()


print(f"After cleaning: {len(prices.columns)} stocks with complete data")

if prices.shape[1] < 20:
    raise ValueError("Too few stocks after cleaning — check data download.")


# Monthly prices
monthly_prices = prices.resample("ME").last()

# Momentum calculations
mom_1m = monthly_prices.pct_change(1)
mom_3m = monthly_prices.pct_change(3)
mom_6m = monthly_prices.pct_change(6)
mom_12m = monthly_prices.pct_change(12)

def zscore_monthly(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)

z_1m = zscore_monthly(mom_1m)
z_3m = zscore_monthly(mom_3m)
z_6m = zscore_monthly(mom_6m)
z_12m = zscore_monthly(mom_12m)

combined_score = (z_1m + z_3m + z_6m + z_12m) / 4

# Trading algorithm
def get_top_picks(month_scores):
    valid = month_scores.dropna()
    if len(valid) == 0:
        return None, []
    ranked = valid.sort_values(ascending=False)
    return ranked.index[0], ranked.head(10).index.tolist()

print("Running backtest...")

monthly_returns = monthly_prices.pct_change()

# QQQ benchmark (clean, simple)
qqq_data = yf.download("QQQ", start=start_date, end=end_date, progress=False)
price_col = "Adj Close" if "Adj Close" in qqq_data.columns else "Close"
qqq_prices = qqq_data[price_col]

qqq_monthly = qqq_prices.resample("ME").last().pct_change()
qqq_monthly.name = "QQQ"

results = []
picks_log = []

months = combined_score.index

for i in range(len(months) - 1):
    month = months[i]
    next_month = months[i + 1]

    scores = combined_score.loc[month]
    best_pick, top_10 = get_top_picks(scores)

    if best_pick is None:
        continue

    valid_scores = scores.dropna().sort_values(ascending=False)
    longs = valid_scores.head(LONG_N).index.tolist()
    shorts = valid_scores.tail(SHORT_N).index.tolist()

    next_returns = monthly_returns.loc[next_month]

    long_ret = next_returns[longs].mean()
    short_ret = next_returns[shorts].mean()
    longshort_ret = (long_ret - short_ret) / 2

    qqq_ret = qqq_monthly.loc[next_month] if next_month in qqq_monthly.index else np.nan

    results.append({
        "Month": month,
        "Long_Return": long_ret,
        "Short_Return": short_ret,
        "LongShort_Return": longshort_ret,
        "QQQ_Return": qqq_ret
    })

    picks_log.append({
        "Month": month,
        "Best_Pick": best_pick,
        "Top_10": ", ".join(top_10)
    })

results_df = pd.DataFrame(results).set_index("Month")
picks_df = pd.DataFrame(picks_log)

print("Backtest complete!")

# Cumulative returns
results_df["LongShort_Cumulative"] = (1 + results_df["LongShort_Return"]).cumprod()
results_df["QQQ_Cumulative"] = (1 + results_df["QQQ_Return"]).cumprod()

# Save results
results_df.to_csv("results/backtest_results.csv")
picks_df.to_csv("results/monthly_picks.csv", index=False)

# Chart 1
plt.figure(figsize=(12,6))
plt.plot(results_df.index, results_df["LongShort_Return"], label="Long-Short Strategy")
plt.plot(results_df.index, results_df["QQQ_Return"], label="QQQ Benchmark", alpha=0.7)
plt.title("Monthly Returns: Strategy vs QQQ")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/chart1_monthly_returns.png")
plt.show()

# Chart 2: Monthly returns - Long basket vs QQQ
plt.figure(figsize=(12,6))
plt.plot(results_df.index, results_df["Long_Return"], label="Long Basket")
plt.plot(results_df.index, results_df["QQQ_Return"], label="QQQ Benchmark", alpha=0.7)
plt.title("Monthly Returns: Long Basket vs QQQ")
plt.xlabel("Date")
plt.ylabel("Monthly Return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/chart2_long_vs_qqq.png")
plt.show()


#Chart 3
plt.figure(figsize=(12,6))
plt.plot(results_df.index, results_df["LongShort_Cumulative"], label="Strategy", linewidth=2)
plt.plot(results_df.index, results_df["QQQ_Cumulative"], label="QQQ", linestyle="--", linewidth=2)
plt.title("Cumulative Returns: Strategy vs QQQ")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/chart2_cumulative.png")
plt.show()

print("All done! Check the results folder.")
