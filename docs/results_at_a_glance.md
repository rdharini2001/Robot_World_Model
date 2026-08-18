# Results at a glance

## Nominal condition

| Observer | Replay position RMSE (m) | Closed-loop cross-track RMSE (m) |
|---|---:|---:|
| Dead reckoning | 1.503 | 1.045 |
| EKF | 0.140 | 0.212 |
| GRU-DR | 0.281 | 0.694 |
| SSM-DR | 0.325 | 0.909 |
| GRU-EKF | **0.094** | **0.163** |
| SSM-EKF | 0.121 | 0.199 |

## Observer selection across 24 conditions

- Replay-position-RMSE Spearman correlation with closed-loop cross-track RMSE: **0.923**.
- Replay-position-RMSE Pearson correlation with closed-loop cross-track RMSE: **0.477**.
- Different open-loop and closed-loop winner: **5/24 conditions (20.8%)**.
- Mean selection regret: **0.0051 m**.
- Maximum selection regret: **0.0347 m**.

## Matched ablation

At nominal conditions, cross-track RMSE is 0.199 m for selective SSM-EKF, 0.203 m for the same model without selectivity, and 0.195 m without explicit mask input. The ordering changes under stress. The EKF anchor is more consistently important than either architectural choice.

## Training initialization

Three SSM-EKF training runs produce nominal cross-track means from 0.189 to 0.199 m. Under long blackouts, the range widens to 0.231-0.336 m.

## Physical-log extension

On 600 seconds of UTIAS MRCLAM Robot 1 data, dead reckoning gives 3.392 m position RMSE and the best tested EKF setting gives 1.357 m.
