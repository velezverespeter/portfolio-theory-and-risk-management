"""
Efficient Frontier, Value-at-Risk & Monte Carlo Portfolio Simulation
----------------------------------------------------------------------
Constructs a diversified multi-asset portfolio, derives the Markowitz
efficient frontier via random portfolio simulation, identifies the
maximum-Sharpe and minimum-volatility portfolios, computes historical and
parametric Value-at-Risk (VaR) and Conditional VaR (CVaR) for the optimal
portfolio, and runs a Monte Carlo simulation of forward portfolio value.

Usage:
    python portfolio_risk_model.py
    python portfolio_risk_model.py --fallback

Data source:
    Live daily price history pulled via yfinance for a 5-asset universe
    (SPY, QQQ, EFA, AGG, GLD) covering US equities, tech/growth, international
    developed equities, US bonds, and gold.

    If the API is unreachable, the model falls back to SYNTHETIC returns
    calibrated to sourced, long-run stylized historical parameters (see
    FALLBACK_PARAMS below). This fallback is for demonstration and testing
    only — it is not real historical price data — and is labeled as such
    everywhere it appears in output.

Author: Peter Velez Vereš
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ASSETS = ["SPY", "QQQ", "EFA", "AGG", "GLD"]
ASSET_LABELS = {
    "SPY": "US Large-Cap Equities",
    "QQQ": "US Tech / Growth",
    "EFA": "International Developed Equities",
    "AGG": "US Aggregate Bonds",
    "GLD": "Gold",
}

# ── Fallback calibration (used only if live data is unreachable) ────────────
# Long-run stylized annualized return/volatility approximations, informed by
# 10-year historical figures (PortfoliosLab, AverageAnnualReturn.com, dated
# 2026-07-19) and standard asset-class volatility approximations. These are
# NOT a live price series — they parameterize a synthetic multivariate normal
# return generator so the model can be demonstrated without a network call.
FALLBACK_PARAMS = {
    "annual_return": {"SPY": 0.11, "QQQ": 0.18, "EFA": 0.07, "AGG": 0.03, "GLD": 0.08},
    "annual_vol":    {"SPY": 0.155, "QQQ": 0.21, "EFA": 0.17, "AGG": 0.055, "GLD": 0.15},
    # Approximate long-run correlation matrix (order: SPY, QQQ, EFA, AGG, GLD)
    "correlation": np.array([
        [1.00, 0.93, 0.85, 0.15, 0.25],
        [0.93, 1.00, 0.75, 0.05, 0.20],
        [0.85, 0.75, 1.00, 0.20, 0.25],
        [0.15, 0.05, 0.20, 1.00, 0.20],
        [0.25, 0.20, 0.25, 0.20, 1.00],
    ]),
}

TRADING_DAYS = 252
RANDOM_SEED = 42


def fetch_live_returns(years: int = 5) -> pd.DataFrame:
    """Pull live daily adjusted close prices via yfinance and return daily
    log returns. Raises on failure so the caller can fall back cleanly."""
    import yfinance as yf

    data = yf.download(ASSETS, period=f"{years}y", auto_adjust=True)["Close"]
    if data.empty or data.isnull().all().any():
        raise ValueError("Live price data incomplete or empty.")
    returns = np.log(data / data.shift(1)).dropna()
    return returns


def generate_fallback_returns(n_days: int = TRADING_DAYS * 5) -> pd.DataFrame:
    """Generate synthetic daily returns via a multivariate normal
    distribution calibrated to sourced long-run annualized parameters.
    Reproducible via a fixed random seed."""
    rng = np.random.default_rng(RANDOM_SEED)

    ann_ret = np.array([FALLBACK_PARAMS["annual_return"][a] for a in ASSETS])
    ann_vol = np.array([FALLBACK_PARAMS["annual_vol"][a] for a in ASSETS])
    corr = FALLBACK_PARAMS["correlation"]

    daily_mean = ann_ret / TRADING_DAYS
    daily_vol = ann_vol / np.sqrt(TRADING_DAYS)
    cov = np.outer(daily_vol, daily_vol) * corr

    daily_returns = rng.multivariate_normal(daily_mean, cov, size=n_days)
    dates = pd.RangeIndex(n_days)
    return pd.DataFrame(daily_returns, index=dates, columns=ASSETS)


def get_returns(use_fallback: bool = False) -> tuple:
    """Returns (returns_df, is_synthetic)."""
    if use_fallback:
        print("Using synthetic returns calibrated to sourced long-run parameters (dated 2026-07-19).")
        print("NOTE: this is NOT real historical price data — for demonstration only.\n")
        return generate_fallback_returns(), True
    try:
        returns = fetch_live_returns()
        print("Pulled live 5-year daily price history via yfinance.\n")
        return returns, False
    except Exception as e:
        print(f"Live data fetch failed ({e}).")
        print("Falling back to synthetic returns calibrated to sourced long-run parameters.")
        print("NOTE: this is NOT real historical price data — for demonstration only.\n")
        return generate_fallback_returns(), True


def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, cov: np.ndarray) -> tuple:
    """Annualized (return, volatility, Sharpe ratio) for a given weight vector.
    Assumes a 0% risk-free rate for Sharpe simplicity (can be parameterized)."""
    port_return = np.sum(mean_returns * weights) * TRADING_DAYS
    port_vol = np.sqrt(weights.T @ cov @ weights) * np.sqrt(TRADING_DAYS)
    sharpe = port_return / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe


def simulate_random_portfolios(mean_returns: np.ndarray, cov: np.ndarray, n_portfolios: int = 8000) -> pd.DataFrame:
    """Monte Carlo simulation of random long-only portfolio weights to trace
    out the feasible region and approximate the efficient frontier."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_assets = len(mean_returns)
    results = []

    for _ in range(n_portfolios):
        weights = rng.random(n_assets)
        weights /= weights.sum()
        ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov)
        results.append([ret, vol, sharpe] + list(weights))

    columns = ["return", "volatility", "sharpe"] + ASSETS
    return pd.DataFrame(results, columns=columns)


def calculate_var_cvar(portfolio_returns: pd.Series, confidence: float = 0.95) -> dict:
    """Historical and parametric (variance-covariance) daily VaR and CVaR
    for a portfolio return series, at the given confidence level."""
    alpha = 1 - confidence

    # Historical (empirical) method
    hist_var = -np.percentile(portfolio_returns, alpha * 100)
    tail_losses = portfolio_returns[portfolio_returns <= -hist_var]
    hist_cvar = -tail_losses.mean() if len(tail_losses) > 0 else hist_var

    # Parametric (variance-covariance) method, assumes normality
    from scipy.stats import norm
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z = norm.ppf(alpha)
    param_var = -(mu + z * sigma)
    param_cvar = -(mu - sigma * norm.pdf(z) / alpha)

    return {
        "confidence": confidence,
        "historical_var": hist_var,
        "historical_cvar": hist_cvar,
        "parametric_var": param_var,
        "parametric_cvar": param_cvar,
    }


def monte_carlo_simulation(mean_return: float, vol: float, initial_value: float = 100.0,
                            horizon_days: int = TRADING_DAYS, n_simulations: int = 2000) -> np.ndarray:
    """Simulate forward portfolio value paths via Geometric Brownian Motion,
    using the annualized return/vol of the selected portfolio."""
    rng = np.random.default_rng(RANDOM_SEED)
    dt = 1 / TRADING_DAYS
    daily_drift = (mean_return - 0.5 * vol ** 2) * dt
    daily_shock = vol * np.sqrt(dt)

    shocks = rng.standard_normal((n_simulations, horizon_days))
    log_returns = daily_drift + daily_shock * shocks
    cumulative = np.cumsum(log_returns, axis=1)
    paths = initial_value * np.exp(cumulative)
    paths = np.hstack([np.full((n_simulations, 1), initial_value), paths])
    return paths


def run_analysis(use_fallback: bool = False):
    returns, is_synthetic = get_returns(use_fallback=use_fallback)
    mean_returns = returns.mean().values
    cov = returns.cov().values

    # ── Efficient frontier via random portfolio simulation ──────────────
    portfolios = simulate_random_portfolios(mean_returns, cov)
    max_sharpe_port = portfolios.loc[portfolios["sharpe"].idxmax()]
    min_vol_port = portfolios.loc[portfolios["volatility"].idxmin()]

    print(f"{'='*70}")
    print("EFFICIENT FRONTIER — OPTIMAL PORTFOLIOS")
    print(f"{'='*70}\n")

    print("Maximum Sharpe Ratio Portfolio")
    print(f"  Expected Annual Return:  {max_sharpe_port['return']:.2%}")
    print(f"  Annual Volatility:       {max_sharpe_port['volatility']:.2%}")
    print(f"  Sharpe Ratio:            {max_sharpe_port['sharpe']:.2f}")
    print("  Weights:")
    for a in ASSETS:
        print(f"    {a} ({ASSET_LABELS[a]}): {max_sharpe_port[a]:.1%}")
    print()

    print("Minimum Volatility Portfolio")
    print(f"  Expected Annual Return:  {min_vol_port['return']:.2%}")
    print(f"  Annual Volatility:       {min_vol_port['volatility']:.2%}")
    print(f"  Sharpe Ratio:            {min_vol_port['sharpe']:.2f}")
    print("  Weights:")
    for a in ASSETS:
        print(f"    {a} ({ASSET_LABELS[a]}): {min_vol_port[a]:.1%}")
    print()

    # ── VaR / CVaR on the max-Sharpe portfolio ───────────────────────────
    max_sharpe_weights = max_sharpe_port[ASSETS].values.astype(float)
    port_daily_returns = returns.values @ max_sharpe_weights
    port_daily_returns = pd.Series(port_daily_returns)

    var95 = calculate_var_cvar(port_daily_returns, confidence=0.95)
    var99 = calculate_var_cvar(port_daily_returns, confidence=0.99)

    print(f"{'='*70}")
    print("VALUE-AT-RISK — Maximum Sharpe Portfolio (1-day horizon)")
    print(f"{'='*70}\n")
    for res in [var95, var99]:
        print(f"Confidence: {res['confidence']:.0%}")
        print(f"  Historical VaR:   {res['historical_var']:.2%}")
        print(f"  Historical CVaR:  {res['historical_cvar']:.2%}")
        print(f"  Parametric VaR:   {res['parametric_var']:.2%}")
        print(f"  Parametric CVaR:  {res['parametric_cvar']:.2%}\n")

    # ── Monte Carlo forward simulation ───────────────────────────────────
    mc_paths = monte_carlo_simulation(
        max_sharpe_port["return"], max_sharpe_port["volatility"]
    )
    final_values = mc_paths[:, -1]
    print(f"{'='*70}")
    print("MONTE CARLO SIMULATION — 1-Year Forward Portfolio Value (base 100)")
    print(f"{'='*70}")
    print(f"  Simulations:             {mc_paths.shape[0]:,}")
    print(f"  Median ending value:     {np.median(final_values):.2f}")
    print(f"  5th percentile:          {np.percentile(final_values, 5):.2f}")
    print(f"  95th percentile:         {np.percentile(final_values, 95):.2f}")
    print(f"{'='*70}\n")

    return {
        "returns": returns,
        "is_synthetic": is_synthetic,
        "portfolios": portfolios,
        "max_sharpe_port": max_sharpe_port,
        "min_vol_port": min_vol_port,
        "var95": var95,
        "var99": var99,
        "mc_paths": mc_paths,
    }


def plot_efficient_frontier(result: dict, output_path: str = "outputs/efficient_frontier.png"):
    portfolios = result["portfolios"]
    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        portfolios["volatility"], portfolios["return"],
        c=portfolios["sharpe"], cmap="viridis", s=6, alpha=0.5
    )
    plt.colorbar(scatter, label="Sharpe Ratio")

    ax.scatter(
        result["max_sharpe_port"]["volatility"], result["max_sharpe_port"]["return"],
        marker="*", color="#C14444", s=400, edgecolor="black", linewidth=1,
        label="Max Sharpe Ratio", zorder=5
    )
    ax.scatter(
        result["min_vol_port"]["volatility"], result["min_vol_port"]["return"],
        marker="*", color="#1672B0", s=400, edgecolor="black", linewidth=1,
        label="Min Volatility", zorder=5
    )

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    title_suffix = " (Synthetic Demo Data)" if result["is_synthetic"] else ""
    ax.set_title(f"Efficient Frontier — 5-Asset Portfolio{title_suffix}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")
    plt.close()


def plot_var_distribution(result: dict, output_path: str = "outputs/var_distribution.png"):
    weights = result["max_sharpe_port"][ASSETS].values.astype(float)
    port_returns = result["returns"].values @ weights

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(port_returns, bins=60, color="#1672B0", alpha=0.75, edgecolor="white")

    var95 = result["var95"]["historical_var"]
    var99 = result["var99"]["historical_var"]
    ax.axvline(-var95, color="#D8A03E", linestyle="--", linewidth=2, label=f"95% VaR: {var95:.2%}")
    ax.axvline(-var99, color="#C14444", linestyle="--", linewidth=2, label=f"99% VaR: {var99:.2%}")

    ax.set_xlabel("Daily Portfolio Return")
    ax.set_ylabel("Frequency")
    title_suffix = " (Synthetic Demo Data)" if result["is_synthetic"] else ""
    ax.set_title(f"Daily Return Distribution — Max Sharpe Portfolio{title_suffix}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")
    plt.close()


def plot_monte_carlo(result: dict, output_path: str = "outputs/monte_carlo_simulation.png"):
    paths = result["mc_paths"]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    days = np.arange(paths.shape[1])
    for i in range(min(150, paths.shape[0])):
        ax.plot(days, paths[i], color="#1672B0", alpha=0.04, linewidth=0.8)

    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    ax.plot(days, p50, color="black", linewidth=2, label="Median")
    ax.plot(days, p5, color="#C14444", linewidth=1.5, linestyle="--", label="5th Percentile")
    ax.plot(days, p95, color="#3EB661", linewidth=1.5, linestyle="--", label="95th Percentile")

    ax.set_xlabel("Trading Days Forward")
    ax.set_ylabel("Portfolio Value (Base = 100)")
    title_suffix = " (Synthetic Demo Data)" if result["is_synthetic"] else ""
    ax.set_title(f"Monte Carlo Simulation — 1-Year Forward Paths{title_suffix}")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio efficient frontier and VaR/CVaR analysis.")
    parser.add_argument(
        "--fallback", action="store_true",
        help="Use synthetic returns calibrated to sourced parameters instead of a live API call",
    )
    args = parser.parse_args()

    result = run_analysis(use_fallback=args.fallback)
    plot_efficient_frontier(result)
    plot_var_distribution(result)
    plot_monte_carlo(result)
