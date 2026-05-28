from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

from merger_simulator.demand.base import EstimationResult


def estimate_iv2sls(
    dep: pd.Series,
    exog: pd.DataFrame,
    endog: pd.DataFrame,
    instruments: pd.DataFrame,
    cov_type: str = "robust",
) -> EstimationResult:
    """Run IV-2SLS and return an EstimationResult."""
    model = IV2SLS(dep, exog, endog, instruments).fit(cov_type=cov_type)

    params = dict(model.params)
    std_errors = dict(model.std_errors)
    t_stats = dict(model.tstats)
    residuals = model.resids.values

    first_stage_f = None
    try:
        diag = model.first_stage
        if diag is not None:
            f_vals = [diag.diagnostics[col]["f.stat"] for col in diag.diagnostics]
            first_stage_f = min(f_vals) if f_vals else None
    except Exception:
        pass

    return EstimationResult(
        params=params,
        std_errors=std_errors,
        t_stats=t_stats,
        residuals=residuals,
        first_stage_f=first_stage_f,
        r_squared=model.rsquared,
        n_obs=model.nobs,
        model_object=model,
    )
