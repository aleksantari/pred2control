"""
data.py (Torch env)

Dataset utilities for Motor-GPT:
- load episode-grouped action tensors from `*.pt`
- split episodes into train/val/test (by episode, no leakage)
- ActionSequenceDataset: samples random windows within episodes (no boundary crossing)

Yields:
  x: (B, T, 6) actions
  y: (B, T, 6) next-action targets (shifted by 1)
"""
