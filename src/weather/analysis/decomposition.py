"""Harmonic-regression decomposition of the global mean temperature series (shared by `eda` and `anomaly`)."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from weather.enums import GlobalSeriesColumns
from weather.feature_engineering.global_series import seasonal_design_matrix


def harmonic_decomposition(gs: pd.DataFrame, order: int = 3) -> pd.DataFrame:
    """OLS: y = a + b*t + sum_k annual harmonics [+ mean observation-hour harmonics].

    Returns observed / fitted / trend / seasonal / residual; `attrs` hold naive vs hour-adjusted slopes (degC/yr).
    """
    g = gs.set_index("date").interpolate()
    X = seasonal_design_matrix(g.index, g.index[0], order)
    t = X["t"].to_numpy()
    seas_cols = [c for c in X if c != "t"]
    y = g["temperature_celsius"]
    raw = sm.OLS(y, sm.add_constant(X)).fit()
    hour_cols = GlobalSeriesColumns.HOUR
    Xa = X.join(g[hour_cols])
    adj = sm.OLS(y, sm.add_constant(Xa)).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
    out = pd.DataFrame(index=g.index)
    out["observed"] = y
    out["fitted"] = adj.fittedvalues
    out["seasonal"] = X[seas_cols] @ adj.params[seas_cols]
    # trend lines drawn at the series level: intercept + slope*t + mean of the other fitted components
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
