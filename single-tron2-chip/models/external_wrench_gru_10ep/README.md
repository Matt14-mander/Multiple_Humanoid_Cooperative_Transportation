# External wrench GRU checkpoint

This directory is a self-contained, Git-trackable deployment bundle for the
current 6D external-wrench estimator. It was trained on 1,020 episodes from 20
source trajectories with source-disjoint train/validation/test splits.

- Label: `external_wrench_base_at_base_origin`
- Frame/reference: `base_Link` / `base_Link_origin`
- Output: `[Fx,Fy,Fz,Mx,My,Mz]` in N/Nm
- Input: 30 x 53 causal proprioceptive history at 50 Hz
- Test R2: Fx 0.9182, Fy 0.9423, Fz 0.8055, Mx 0.9221, My 0.8693, Mz 0.9397
- Test zero false-positive rate: 4.34% at 1 N / 0.25 Nm thresholds

`deployment_contract.json` is the machine-readable interface. The ONNX model
was checked against PyTorch with maximum normalized absolute error below
`8e-7`. See `training_report.json`, `data_audit.json`, and
`onnx_verification.json` for full provenance and metrics.
