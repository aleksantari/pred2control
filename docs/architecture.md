### Design philosophy: minimal deltas

MotorGPT_chunk and MotorFlow_chunk share the same backbone on purpose:
- identical action embeddings (`action_in`, `action_out`)
- identical positional embedding scheme (`pos_emb`)
- identical context encoder structure (`ContextBlocks + RMSNorm`)
- identical decoder topology (SelfAttn → CrossAttn(ctx) → MLP)

The only intentional changes required to transition from unimodal regression to distributional modeling are:
1) Replace learned query tokens with *noisy chunk tokens* (the object being denoised).
2) Inject flow-time τ via a dedicated embedding (Sinusoidal + MLP).
3) Use τ-conditioned normalization (AdaRMSNorm) inside the decoder blocks.

This makes MotorFlow_chunk a natural continuation of MotorGPT_chunk rather than a separate codepath.
