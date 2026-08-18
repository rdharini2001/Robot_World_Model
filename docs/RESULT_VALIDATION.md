# Result validation

The bundled result package was checked against the numerical claims used in the workshop manuscript.

## Automated checks

Run:

```bash
PYTHONPATH=src python scripts/verify_results.py
pytest -q
```

Both checks pass in the supplied package.

## Headline values confirmed

- 24 sensing conditions and 6 core observers (144 observer-condition pairs).
- Replay position RMSE vs. closed-loop cross-track RMSE: Spearman rho = 0.922581 -> 0.923 in the paper.
- 20-step blind-rollout RMSE vs. closed-loop cross-track RMSE: Spearman rho = 0.773511 -> 0.774.
- Top-1 selection failures: replay 5/24; blind rollout 18/24.
- Maximum regret: replay 0.034743 m; blind rollout 0.120761 m.
- Compound stress: GRU-EKF 1.716929 -> 1.061176 m with long-blackout training.
- Compound stress: SSM-EKF 1.936156 -> 1.419116 m with long-blackout training.
- Compound stress: SSM-DR 5.290042 -> 5.410953 m with long-blackout training.
- At blackout length 28: GRU-EKF 0.286 m vs. trained 0.183 m.
- At blackout length 34: SSM-EKF 0.336 m vs. trained 0.819 m.

These checks validate consistency between the bundled JSON outputs and the manuscript's reported headline results. They do not constitute an independent external replication of the simulator or training procedure.
