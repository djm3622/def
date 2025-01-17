# Time Invariant Operator Guided Diffusion

Modify a UNet model to take in both conditional time embedding, standard time/noise embedding, and concat my conditional state along the channel dimension with the Y during training/sampling.

# TODO
1. Train an unconditional model to see quality results
    a. Determine the best way to train DDPM (current best techniques)
    b. Train
2. start working on model (concat conditional state, conditional time embeddings, time-embeddigs/noise-embeddings)
3. Figure out training process (do I consider CFG?, do I do LDM?)

# READING
1. Survey paper notes
    a. DDPM + DDIM
    b. Noise schedule optimization
    c. Variance learning

# References
1. https://arxiv.org/pdf/2101.12072
2. https://arxiv.org/pdf/2112.10752
3. Survey: https://arxiv.org/pdf/2209.00796v13
4. VLB Loss: https://arxiv.org/pdf/2102.09672
