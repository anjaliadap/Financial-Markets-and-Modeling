import requests 
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt

# Getting data from 
START_DATE = "2024-09-30"
END_DATE = date.today().isoformat()

# Starting trades on 
TRADE_DATE = "2024-10-01"

NVDA_NOTIONAL = 1_000_000.0
AVGO_NOTIONAL = 1_000_000.0
INIT_DEPOSIT_CASH = 1_000_000.0

def fetch_sofr(start=START_DATE, end=END_DATE):
    s = pdr.DataReader('SOFR', 'fred', start, end)
    s = s.rename(columns={'SOFR': 'SOFR'}).astype(float) / 100.0 # FRED returns percent

    return s

sofr_df = fetch_sofr()
sofr_df

sofr_df.isna().value_counts()

# Handling missing values using backward-filling
sofr_df['SOFR'].bfill(inplace=True)

tickers = ['NVDA', 'AVGO']

price_df = yf.download(tickers=tickers
                       , start=START_DATE # Don't specify period with start-end
                       , end=END_DATE
                       , interval="1d"
                       , auto_adjust=True)["Close"]

# --- Params ---
REG_T_INITIAL_MARGIN = 0.50        # 50% initial margin (documented)
VARIATION_MARGIN_RATIO = 0.40      # 40% account variation margin requirement

# Interest rate earned on cash credit
def cash_credit_interest_rate(sofr_rate):
    """
    SOFR - 50 bps
    """
    adjusted_rate = sofr_rate - 0.005

    return max(adjusted_rate, 0)  # floor at 0 (no negative interest)

# Margin loan financing rate
def margin_loan_interest_rate(sofr_rate):
    """
    SOFR + 50 bps
    """
    return sofr_rate + 0.005

df = price_df.copy()
df = df.join(sofr_df, how="left")

df.isna().value_counts()

df["cash_credit_rate"] = df["SOFR"].apply(cash_credit_interest_rate)
df["margin_loan_rate"] = df["SOFR"].apply(margin_loan_interest_rate)
df["stock_borrow_rate"] = 0.005

# No of days between each trading day 
df["cal_days"] = df.index.to_series().diff().dt.days.fillna(0).astype(int)

# Entry prices & share counts
trade_ts = pd.to_datetime(TRADE_DATE)
if trade_ts not in df.index:
    raise RuntimeError(f"No price for trade date {TRADE_DATE}. Check holidays/start date.")

nvda_entry = float(df.loc[trade_ts, "NVDA"])
avgo_entry = float(df.loc[trade_ts, "AVGO"])
nvda_shares = NVDA_NOTIONAL / nvda_entry             # long shares
avgo_shares = AVGO_NOTIONAL / avgo_entry             # short shares

DAY_COUNT = 360  # ACT/360

# Daily market values (positive numbers for absolute MVs)
df["MV_long"]  = nvda_shares * df["NVDA"]
df["MV_short"] = avgo_shares * df["AVGO"]
df["Gross_Exposure"] = df["MV_long"] + df["MV_short"]

# --- State arrays ---
n = len(df)
start_idx = df.index.get_loc(trade_ts)

cash = np.zeros(n)
equity = np.zeros(n)
contrib = np.zeros(n)     # external top-ups (margin calls)
int_cash = np.zeros(n)    # interest on cash (credit or debit)
borrow_fee = np.zeros(n)  # stock borrow accrual
daily_roe = np.full(n, np.nan)  # contribution-adjusted

# Initial cash/equity (on trade date close)
# Deposit +1mm; buy long -1mm; receive short +1mm => net cash = +1mm
cash0 = INIT_DEPOSIT_CASH - NVDA_NOTIONAL + AVGO_NOTIONAL
eq0   = cash0 + df.iloc[start_idx]["MV_long"] - df.iloc[start_idx]["MV_short"]

cash[start_idx]   = cash0
equity[start_idx] = eq0

# --- EOD loop ---
for i in range(start_idx + 1, n):  # iterate every trading day after initial trade date
    prev = i - 1

    # Calendar days since last trading day (1 normally, 3 over weekends, etc.)
    days = int(df.iloc[i]["cal_days"])

    # --- interest on cash using YESTERDAY's balances/rates (a & b) ---
    c_prev = cash[prev]
    r_credit = df.iloc[prev]["cash_credit_rate"]
    r_debit  = df.iloc[prev]["margin_loan_rate"]
    if c_prev >= 0:
        i_cash = c_prev * r_credit * (days / DAY_COUNT)  # interest earned
    else:
        i_cash = c_prev * r_debit  * (days / DAY_COUNT)  # interest paid (negative)

    # --- stock borrow on YESTERDAY's short MV (c) ---
    mv_short_prev = df.iloc[prev]["MV_short"]
    r_borrow = df.iloc[prev]["stock_borrow_rate"]  # here = 0.005 constant
    i_borrow = - mv_short_prev * r_borrow * (days / DAY_COUNT)  # negative (reduces cash)

    # Update cash with accruals
    c_now = c_prev + i_cash + i_borrow

    # --- equity from first principles (e.i & e.ii) ---
    eq_now = c_now + df.iloc[i]["MV_long"] - df.iloc[i]["MV_short"]

    # --- variation margin 40% of gross exposure (d & e.iii) ---
    req_eq = VARIATION_MARGIN_RATIO * df.iloc[i]["Gross_Exposure"]
    addl = 0.0
    if eq_now < req_eq:
        addl = (req_eq - eq_now) + 1.0  # tiny buffer
        c_now += addl
        eq_now += addl

    # Store all the updated values for this day
    cash[i]        = c_now
    equity[i]      = eq_now
    contrib[i]     = addl
    int_cash[i]    = i_cash
    borrow_fee[i]  = i_borrow

    # --- daily ROE excluding same-day contributions (f) ---
    eq_prev = equity[prev]
    if eq_prev != 0:
        daily_roe[i] = (eq_now - eq_prev - addl) / eq_prev

# Attach to df
df["Cash"] = cash
df["Equity"] = equity
df["Contrib"] = contrib
df["Int_Cash"] = int_cash
df["Borrow_Fee"] = borrow_fee
df["Daily_ROE"] = daily_roe

# Slice from trade date onward (reporting period)
ledger = df.iloc[start_idx:].copy()

# --- performance stats ---
rets = ledger["Daily_ROE"].dropna()
avg  = rets.mean()
vol  = rets.std(ddof=1)   # ✅ sample stdev is standard for Sharpe on historical data
sharpe = (avg / vol) * np.sqrt(252) if vol > 0 else np.nan

print("Summary:")
print(f"Start: {ledger.index[0].date()}  End: {ledger.index[-1].date()}")
print(f"Final Equity: ${ledger['Equity'].iloc[-1]:,.2f}")
print(f"Total Contributions (margin calls): ${ledger['Contrib'].sum():,.2f}")
print(f"Avg Daily ROE: {avg:.6f}  |  Vol: {vol:.6f}  |  Sharpe(252): {sharpe:.2f}")

