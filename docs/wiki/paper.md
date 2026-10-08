# Paper

TabFORGE was accepted to NeurIPS 2026. The paper introduces a tabular foundation model for generative modelling built on pretrained structure-aware representations and a two-stage latent modelling and decoding design.

[Read the paper](https://arxiv.org/abs/2605.09424) · [Pretrained model](https://huggingface.co/XiangjianJiang/TabFORGE) · [Source code](https://github.com/SilenceX12138/TabFORGE)

<figure class="paper-figure">
<a href="assets/paper-framework.svg" target="_blank" rel="noopener"><img src="assets/paper-framework.svg" alt="Original paper architecture: mixed-type input table, frozen Structure-aware Feature Encoder, Score-based Diffusion Transformer, and Denoising-aligned Latent Decoder" width="2400" height="592"></a>
</figure>

## Citation

Xiangjian Jiang, Mingxuan Liu, Nikola Simidjievski, Tassilo Klein, and Mateja Jamnik. “TabFORGE: A Tabular Foundation Model for Generative Modelling.” *Advances in Neural Information Processing Systems*, 2026.

```bibtex
@inproceedings{jiang2026tabforge,
  title     = {{TabFORGE}: A Tabular Foundation Model for Generative Modelling},
  author    = {Jiang, Xiangjian and Liu, Mingxuan and Simidjievski, Nikola
               and Klein, Tassilo and Jamnik, Mateja},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Method overview

TabFORGE uses three components:

1. **Structure-aware Feature Encoder** produces per-feature latent embeddings that capture useful inter-feature dependencies.
2. **Score-based Diffusion Transformer** learns a distribution over those latent embeddings.
3. **Denoising-aligned Latent Decoder** reconstructs table values from denoised latents, aligning its training inputs with inference-time latents.

The paper evaluates TabFORGE against 22 benchmark methods on 45 real-world datasets. Consult the paper for the experimental protocol, metrics, and complete results.

## Model and implementation

The estimator workflow and task APIs are described in the [user guide](user_guide.md). The implementation's feature-processing, diffusion, and Denoising-aligned Latent Decoder components are introduced in [feature processing](tabular_feature_processing.md) and [Score-based Diffusion Transformer and Denoising-aligned Latent Decoder](latent_diffusion_model.md).

The paper figure presents the generative architecture. `TabFORGEGenerator` starts from latent embeddings with random noise, denoises them with the Score-based Diffusion Transformer, and reconstructs table values with the Denoising-aligned Latent Decoder. The component names follow the camera-ready manuscript.

## Earlier workshop version

The earlier version, [A Generative Foundation Model for Heterogeneous Tabular Data](https://openreview.net/forum?id=RcsaxrdpfE), appeared at the 2nd ICML Workshop on Foundation Models for Structured Data in 2026.

```bibtex
@inproceedings{jiang2026generative,
  title     = {A Generative Foundation Model for Heterogeneous Tabular Data},
  author    = {Jiang, Xiangjian and Liu, Mingxuan and Simidjievski, Nikola
               and Klein, Tassilo and Jamnik, Mateja},
  booktitle = {2nd ICML Workshop on Foundation Models for Structured Data},
  year      = {2026},
  url       = {https://openreview.net/forum?id=RcsaxrdpfE}
}
```
