import pandas as pd

from soccercpd.core import SoccerCPD
from soccercpd.myconstants import (
    MAX_SWITCH_RATE,
    MAX_PVAL,
    MIN_PERIOD_DUR,
    MIN_FORM_DIST,
)

def run_formcpd(
    match,
    *,
    # CPD configuration
    formcpd_type: str = "gseg_avg",
    rolecpd_type: str = "gseg_avg",
    apply_cpd: bool = True,

    # Hyperparameters (EXPLICIT)
    min_pdur: int = MIN_PERIOD_DUR,
    min_fdist: int = MIN_FORM_DIST,
    max_pval: float = MAX_PVAL,
    max_sr: float = MAX_SWITCH_RATE,
    
    # Output control
    return_cpd: bool = False,
):
    cpd = SoccerCPD(
        match,
        apply_cpd=apply_cpd,
        formcpd_type=formcpd_type,
        rolecpd_type=rolecpd_type,
        min_pdur=min_pdur,
        min_fdist=min_fdist,
        max_pval=max_pval,
        max_sr=max_sr,
    )

    cpd.run()

    form_periods = cpd.form_periods.copy()
    change_points = form_periods["start_dt"].iloc[1:].tolist()

    out = {
        "n_segments": len(form_periods),
        "change_points": change_points,
        "form_periods": form_periods,
    }

    if return_cpd:
        out["cpd"] = cpd

    return out
