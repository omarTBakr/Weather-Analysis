"""Harmonic-regression decomposition of the global mean temperature series (shared by `eda` and `anomaly`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def harmonic_decomposition(gs: pd.DataFrame, order: int = 3) -> pd.DataFrame:
    """OLS: y = a + b*t + sum_k annual harmonics [+ mean observation-hour harmonics].

    Returns observed / fitted / trend / seasonal / residual; `attrs` hold naive vs hour-adjusted slopes (degC/yr).
    """
    g = gs.set_index("date").interpolate()
    t = (g.index - g.index[0]).days.to_numpy() / 365.25
    X = pd.DataFrame({"t": t}, index=g.index)
    doy = g.index.day_of_year.to_numpy()
    for k in range(1, order + 1):
        X[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    seas_cols = [c for c in X if c != "t"]
    y = g["temperature_celsius"]
    raw = sm.OLS(y, sm.add_constant(X)).fit()
    Xa = X.assign(hour_sin=g["hour_sin"], hour_cos=g["hour_cos"])
    adj = sm.OLS(y, sm.add_constant(Xa)).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
    out = pd.DataFrame(index=g.index)
    out["observed"] = y
    out["fitted"] = adj.fittedvalues
    out["seasonal"] = X[seas_cols] @ adj.params[seas_cols]
    # trend lines drawn at the series level: intercept + slope*t + mean of the other fitted components
    hour_cols = ["hour_sin", "hour_cos"]
    out["trend_adj"] = (
        adj.params["const"]
        + adj.params["t"] * t
        + (Xa[seas_cols + hour_cols] @ adj.params[seas_cols + hour_cols]).mean()
    )
    out["trend_raw"] = raw.params["const"] + raw.params["t"] * t + (X[seas_cols] @ raw.params[seas_cols]).mean()
    out["resid"] = adj.resid
    out.attrs.update(
        slope_raw=float(raw.params["t"]),
        slope_adj=float(adj.params["t"]),
        slope_adj_ci=[float(v) for v in adj.conf_int().loc["t"]],
    )
    return out
