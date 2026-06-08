# Diffusion Policy Validity Checklist

A result may call itself "Diffusion Policy-style" only if it includes:

- stochastic trajectory generation;
- iterative denoising or diffusion-like noise-to-action generation;
- conditioning on observation/state;
- action sequence generation rather than only one-step action prediction;
- evaluation under receding-horizon or trajectory-execution setting.

If an experiment uses a hand-designed sampler, label it:

`controlled diffusion-like action sampler`

not full Diffusion Policy.

If an experiment trains a learned denoising policy, label it:

`learned Diffusion Policy-lite`

The learned toy experiment in this repository includes state-conditioned and 32x32 image-conditioned variants. The image path uses a tiny CNN encoder over rendered block/goal/obstacle observations with visual OOD regimes, not a full visual robotics benchmark.

The learned toy experiment is a small CPU-feasible model, not evidence that full-scale robot Diffusion Policies benefit universally from high `N`.
