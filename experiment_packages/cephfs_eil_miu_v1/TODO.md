# TODO

- Ensure `/cephfs/shared/experiment_g/assets/models/Olmo-3-7B-Instruct` exists on the training host.
- Ensure `/cephfs/shared/experiment_g/assets/models/Olmo-3-7B-Instruct_torch_dist` is converted before launch.
- Fill `.env` judge/adversary endpoints and API keys on the target training host, plus verify MIU/EIL dataset paths in the project tree.
- Keep the EIL:MIU ratio at `2:1` and the total rollout budget at `200` unless a later experiment explicitly changes the checkpoint grid.
