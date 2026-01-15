"""
model.py (Torch env)

Motor-GPT: a NanoGPT-style causal decoder adapted for continuous actions.
Key differences from text GPT:
- Input "embedding": Linear(action_dim=6 -> d_model)
- Output head: Linear(d_model -> action_dim=6)
- Loss: regression (MSE or SmoothL1) on next-action prediction
"""
