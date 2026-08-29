# CephFS EIL/MIU OLMo3 Prep

Independent prep package for an OLMo3-oriented EIL/MIU training layout on CephFS.

- Target model: `allenai/Olmo-3-7B-Instruct`
- Local model root: `/cephfs/shared/experiment_g/assets/models/Olmo-3-7B-Instruct`
- Batch ratio: `EIL:MIU = 2:1`
- Optimization target: faster convergence, starting from `learning_rate=2e-6`
- Total rollout budget: `200` to stay aligned with the existing stage/checkpoint matrix
- Checkpoints / outputs / logs / data: `/cephfs/shared/experiment_g/cephfs_eil_miu_v1`

This package now targets the bundled local OLMo3 profile and bridge patches in `loyal_agent_docker`.
Training can start once the CephFS model directory, matching `_torch_dist` reference checkpoint, and MIU/EIL dataset paths are all ready.

## Layout

```text
cephfs_eil_miu_v1/
  configs/e2m1_cephfs_rollout200.json
  scripts/download_model_cephfs.sh
  scripts/launch_cephfs_e2m1.sh
  scripts/preflight.sh
  TODO.md
```

## Usage

1. Download model assets to CephFS:
```bash
bash scripts/download_model_cephfs.sh
```

2. Validate the prep config and paths:
```bash
bash scripts/preflight.sh
```

3. Review the upstream integration TODO before launching:
```bash
cat TODO.md
```

4. Launch the package wrapper:
```bash
bash scripts/launch_cephfs_e2m1.sh
```
