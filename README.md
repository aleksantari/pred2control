# motor-gpt-so101

Minimal Motor-GPT (GPT-style decoder) trained on SO-101 action trajectories.
Two-env workflow:
- LeRobot env: download + extract episode-grouped actions → `.pt`
- Torch env: load `.pt`, sample episode-respecting windows, train Motor-GPT

Data artifact (shared across envs): `data/svla_so101_actions.pt`
