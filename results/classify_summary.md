# Classification sweep — comparison tables

### Bangladesh held-out (transfer) — AUC ± std (95% CI)

| Target | XGBoost (raw) | LogReg (raw) | Small frozen + MLP | Small unfrozen + MLP | Small no-pretrain + MLP | Small frozen + LinProbe | Base frozen + MLP | Base unfrozen + MLP | Base no-pretrain + MLP | Base frozen + LinProbe |
|---|---|---|---|---|---|---|---|---|---|---|
| As | 0.701±0.033 (CI 0.63–0.76) | 0.720±0.012 (CI 0.65–0.79) | 0.632±0.004 (CI 0.56–0.70) | 0.703±0.025 (CI 0.63–0.77) | 0.700±0.018 (CI 0.63–0.77) | 0.642±0.015 (CI 0.57–0.71) | 0.559±0.019 (CI 0.48–0.63) | 0.690±0.014 (CI 0.62–0.76) | 0.704±0.028 (CI 0.63–0.77) | 0.571±0.026 (CI 0.49–0.65) |
| F | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| NO3 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| U | NaN | — | NaN | NaN | NaN | NaN | — | — | — | — |

### Bangladesh held-out (transfer) — AP ± std

| Target | XGBoost (raw) | LogReg (raw) | Small frozen + MLP | Small unfrozen + MLP | Small no-pretrain + MLP | Small frozen + LinProbe | Base frozen + MLP | Base unfrozen + MLP | Base no-pretrain + MLP | Base frozen + LinProbe |
|---|---|---|---|---|---|---|---|---|---|---|
| As | 0.459±0.035 | 0.549±0.014 | 0.366±0.003 | 0.468±0.046 | 0.490±0.038 | 0.374±0.014 | 0.344±0.006 | 0.506±0.048 | 0.515±0.079 | 0.351±0.017 |
| F | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| NO3 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| U | NaN | — | NaN | NaN | NaN | NaN | — | — | — | — |

### Bangladesh held-out (transfer) — F1_best ± std

| Target | XGBoost (raw) | LogReg (raw) | Small frozen + MLP | Small unfrozen + MLP | Small no-pretrain + MLP | Small frozen + LinProbe | Base frozen + MLP | Base unfrozen + MLP | Base no-pretrain + MLP | Base frozen + LinProbe |
|---|---|---|---|---|---|---|---|---|---|---|
| As | 0.593±0.021 | 0.579±0.012 | 0.560±0.005 | 0.547±0.012 | 0.561±0.003 | 0.562±0.004 | 0.494±0.005 | 0.528±0.032 | 0.553±0.025 | 0.496±0.011 |
| F | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| NO3 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| U | NaN | — | NaN | NaN | NaN | NaN | — | — | — | — |

### Global test (in-distribution) — AUC ± std (95% CI)

| Target | XGBoost (raw) | LogReg (raw) | Small frozen + MLP | Small unfrozen + MLP | Small no-pretrain + MLP | Small frozen + LinProbe | Base frozen + MLP | Base unfrozen + MLP | Base no-pretrain + MLP | Base frozen + LinProbe |
|---|---|---|---|---|---|---|---|---|---|---|
| As | 0.974±0.007 (CI 0.95–0.99) | 0.970±0.006 (CI 0.95–0.99) | 0.962±0.017 (CI 0.94–0.98) | 0.956±0.015 (CI 0.93–0.98) | 0.968±0.017 (CI 0.94–0.99) | 0.956±0.011 (CI 0.92–0.98) | 0.965±0.012 (CI 0.94–0.99) | 0.969±0.017 (CI 0.94–0.99) | 0.966±0.022 (CI 0.94–0.99) | 0.958±0.005 (CI 0.93–0.98) |
| F | 0.928±0.014 (CI 0.87–0.98) | 0.895±0.025 (CI 0.82–0.95) | 0.877±0.062 (CI 0.81–0.93) | 0.891±0.069 (CI 0.82–0.95) | 0.895±0.052 (CI 0.82–0.95) | 0.876±0.065 (CI 0.80–0.93) | 0.871±0.053 (CI 0.80–0.93) | 0.896±0.045 (CI 0.82–0.95) | 0.897±0.048 (CI 0.83–0.95) | 0.869±0.051 (CI 0.79–0.93) |
| NO3 | 0.928±0.028 (CI 0.85–0.98) | 0.877±0.044 (CI 0.75–0.96) | 0.885±0.022 (CI 0.77–0.97) | 0.908±0.035 (CI 0.82–0.97) | 0.917±0.022 (CI 0.85–0.97) | 0.880±0.032 (CI 0.78–0.96) | 0.879±0.039 (CI 0.77–0.97) | 0.913±0.022 (CI 0.85–0.97) | 0.911±0.016 (CI 0.83–0.97) | 0.871±0.046 (CI 0.76–0.96) |
| U | NaN | — | NaN | NaN | NaN | NaN | — | — | — | — |

## Per-target verdict — does any encoder condition beat XGBoost on BD AUC?

### As
  XGBoost AUC = 0.701±0.033  (95% upper = 0.766)
  Base frozen + MLP            AUC=0.559±0.019  → LOSES to XGBoost (non-overlapping CI)
  Base unfrozen + MLP          AUC=0.690±0.014  → tie (overlapping CI)
  Base frozen + LinProbe       AUC=0.571±0.026  → LOSES to XGBoost (non-overlapping CI)
  Base no-pretrain + MLP       AUC=0.704±0.028  → tie (overlapping CI)

### F
  XGBoost AUC = nan±nan  (95% upper = nan)
  Base frozen + MLP            AUC=nan±nan  → tie (overlapping CI)
  Base unfrozen + MLP          AUC=nan±nan  → tie (overlapping CI)
  Base frozen + LinProbe       AUC=nan±nan  → tie (overlapping CI)
  Base no-pretrain + MLP       AUC=nan±nan  → tie (overlapping CI)

### NO3
  XGBoost AUC = nan±nan  (95% upper = nan)
  Base frozen + MLP            AUC=nan±nan  → tie (overlapping CI)
  Base unfrozen + MLP          AUC=nan±nan  → tie (overlapping CI)
  Base frozen + LinProbe       AUC=nan±nan  → tie (overlapping CI)
  Base no-pretrain + MLP       AUC=nan±nan  → tie (overlapping CI)

### U
  XGBoost AUC = nan±nan  (95% upper = nan)

## Best condition per target (BD AUC ranking)

### As
  1. LogReg (raw)                   AUC=0.720±0.012
  2. Base no-pretrain + MLP         AUC=0.704±0.028
  3. Small unfrozen + MLP           AUC=0.703±0.025
  4. XGBoost (raw)                  AUC=0.701±0.033
  5. Small no-pretrain + MLP        AUC=0.700±0.018
  6. Base unfrozen + MLP            AUC=0.690±0.014
  7. Small frozen + LinProbe        AUC=0.642±0.015
  8. Small frozen + MLP             AUC=0.632±0.004
  9. Base frozen + LinProbe         AUC=0.571±0.026
  10. Base frozen + MLP              AUC=0.559±0.019