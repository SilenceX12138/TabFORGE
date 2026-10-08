# Table preprocessing and embedding

## Purpose

The table preprocessing and embedding workflow converts mixed-type tabular data into schema-aware latent embeddings for TabFORGE.

It combines two subsystems:

- **Tabular feature processing** captures the input schema, preprocesses numerical and categorical values, builds Denoising-aligned Latent Decoder reconstruction targets, manages missing-value masks, and restores generated data to its original column order and dtypes.
- **Structure-aware Feature Encoder** converts the processed table into contextual per-feature embeddings with the stable shape `(rows, tokens, embedding_dimension)`. It supports leakage-aware training embeddings, query-time extraction, and serializable fitted context.

## Architecture

<ol class="flow-cards">
<li><strong>Capture the table schema</strong><p>TabularFeatureProcessor records raw columns and dtypes, prepares numerical-first values, and constructs reconstruction targets.</p></li>
<li><strong>Extract contextual tokens</strong><p>TabPFNEmbeddingProvider uses held-out folds or full training context to produce a contiguous per-feature latent grid.</p></li>
<li><strong>Model and reconstruct</strong><p>The training workflow fits the Score-based Diffusion Transformer and Denoising-aligned Latent Decoder on the grid and reconstruction targets. The fitted schema restores decoded values to public column order.</p></li>
</ol>

### Embedding flow

```mermaid
sequenceDiagram
    participant API as Estimator API
    participant FP as TabularFeatureProcessor
    participant EP as TabPFNEmbeddingProvider
    participant PFN as encoder runtime
    participant Engine as Training and inference workflow

    API->>FP: fit raw X and optional y
    FP->>FP: capture schema, impute, encode, and scale
    FP-->>API: reconstruction targets and token metadata
    API->>EP: fit using processed values and schema
    EP->>PFN: extract held-out or full-context embeddings
    PFN-->>EP: backend embedding arrays
    EP-->>API: contiguous float32 grid
    API->>Engine: latent grid, reconstruction targets, and schema
    Engine-->>API: decoded table values
    API->>FP: inverse transform
    FP-->>API: values in the original public schema
```

The main contracts are:

- One embedding token corresponds to each raw feature, with an additional target token for supervised tasks.
- Embedding values use numerical-first token order.
- Reconstruction targets contain numerical scalars followed by categorical one-hot blocks.
- The embedding adapter returns CPU-owned, C-contiguous `float32` arrays.
- The feature processor preserves mappings between raw columns, embedding tokens, and Denoising-aligned Latent Decoder outputs.

## Core component documentation

- [Tabular feature processing](tabular_feature_processing.md) — `TabularFeatureProcessor`, `FeatureSchema`, `GaussianQuantileFeatureEncoder`, reconstruction layouts, target processing, inverse transformation, and imputation support.
- [Structure-aware Feature Encoder](tabpfn_embedding_adapter.md) — `EmbeddingProvider`, `TabPFNEmbeddingProvider`, `SampleEmbedding`, backend normalization, leakage-aware folds, query extraction, and provider persistence.
