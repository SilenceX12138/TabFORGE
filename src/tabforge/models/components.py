"""Neural decoder and denoiser components for the TabFORGE latent model."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Legacy-compatible sinusoidal conditioning for continuous EDM noise."""

    def __init__(self, dimension: int, timestep_max: int = 10_000) -> None:
        """Build discrete and continuous sinusoidal frequencies.

        Args:
            dimension: Latent embedding width.
            timestep_max: Largest nominal discrete timestep scale.
        """
        super().__init__()
        self.dimension = int(dimension)
        self.timestep_max = int(timestep_max)
        half_dimension = self.dimension // 2
        discrete_position = torch.arange(self.timestep_max).unsqueeze(1)
        discrete_divisor = torch.exp(torch.arange(half_dimension) * (-math.log(self.timestep_max) / self.dimension))
        discrete = torch.zeros(self.timestep_max, self.dimension)
        if self.dimension % 2 == 0:
            discrete[:, 0::2] = torch.sin(discrete_position * discrete_divisor)
            discrete[:, 1::2] = torch.cos(discrete_position * discrete_divisor)
        else:
            discrete[:, 0:-1:2] = torch.sin(discrete_position * discrete_divisor)
            discrete[:, 1::2] = torch.cos(discrete_position * discrete_divisor)
            discrete[:, -1] = torch.sin(discrete_position[:, 0] * discrete_divisor[0])
        continuous_divisor = (1.0 / self.timestep_max) ** (
            torch.arange(half_dimension, dtype=torch.float32) / max(half_dimension, 1)
        )
        self.register_buffer("frequency_terms_discrete", discrete)
        self.register_buffer("frequency_terms_continuous", continuous_divisor)

    def forward(self, values: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add timestep conditioning to latent values.

        Args:
            values: Latent token grid.
            timesteps: Per-row continuous noise conditioning values.
        """
        if not timesteps.dtype.is_floating_point:
            embedding = self.frequency_terms_discrete[timesteps]
        else:
            frequencies = torch.outer(timesteps, self.frequency_terms_continuous.to(timesteps))
            embedding = torch.zeros(
                (timesteps.shape[0], frequencies.shape[1] * 2),
                device=timesteps.device,
                dtype=timesteps.dtype,
            )
            embedding[:, 0::2] = frequencies.sin()
            embedding[:, 1::2] = frequencies.cos()
            if self.dimension % 2 == 1:
                embedding = torch.cat([embedding, embedding[:, :1].clone()], dim=1)
        return values + embedding.to(values)


class _Transformer(nn.Module):
    """Shared pre-normalized Transformer stack for latent token sequences."""

    def __init__(self, layers: int, dimension: int, heads: int, ffn_factor: int) -> None:
        """Build the shared pre-normalized transformer stack.

        Args:
            layers: Number of encoder layers.
            dimension: Token embedding width.
            heads: Attention heads per layer.
            ffn_factor: Feed-forward width relative to ``dimension``.
        """
        super().__init__()
        if dimension % heads != 0:
            raise ValueError(f"embedding dimension {dimension} must be divisible by {heads} attention heads")
        block = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=dimension * ffn_factor,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, norm=nn.LayerNorm(dimension))

    def forward(self, values: torch.Tensor, *, layer: int | None = None) -> torch.Tensor:
        """Transform a batch of token embeddings.

        Args:
            values: Tensor shaped ``(batch, tokens, dimension)``.
            layer: Optional zero-based layer whose output should be returned.
        """
        if layer is None:
            return self.encoder(values)
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be an integer or None")
        if layer < 0 or layer >= len(self.encoder.layers):
            raise ValueError(f"layer must be between 0 and {len(self.encoder.layers) - 1}")
        hidden = values
        for index, block in enumerate(self.encoder.layers):
            hidden = block(hidden)
            if index == layer:
                break
        if layer == len(self.encoder.layers) - 1 and self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)
        return hidden


class TabFORGEDecoder(nn.Module):
    """Denoising-aligned Latent Decoder for the original table schema.

    The input grid follows TabPFN's numerical-first feature order, followed by
    one target token for supervised tasks. Reconstruction heads select tokens
    from that embedding layout while writing values and logits into the
    processor's reconstruction space. The target head always reads the final
    token and is absent for unsupervision generation.
    """

    def __init__(
        self,
        *,
        n_feature_tokens: int,
        embedding_dimension: int,
        numerical_count: int,
        categorical_cardinalities: list[int],
        numerical_token_positions: list[int] | None = None,
        categorical_token_positions: list[int] | None = None,
        target_kind: str | None,
        target_cardinality: int = 0,
        prediction_head_hidden: int | None = None,
        layers: int = 4,
        heads: int = 2,
        ffn_factor: int = 10,
    ) -> None:
        """Build reconstruction heads aligned with the embedding-token layout.

        Args:
            n_feature_tokens: Number of raw feature tokens.
            embedding_dimension: Width of each latent token.
            numerical_count: Number of scalar numerical output heads.
            categorical_cardinalities: Class count for each categorical column.
            numerical_token_positions: Numerical token positions in latent order.
            categorical_token_positions: Categorical token positions in latent order.
            target_kind: Classification, regression, or ``None``.
            target_cardinality: Number of target classes.
            prediction_head_hidden: Optional lightweight predictor hidden width.
            layers: Decoder Transformer layer count.
            heads: Decoder attention-head count.
            ffn_factor: Decoder feed-forward width factor.
        """
        super().__init__()
        self.n_feature_tokens = int(n_feature_tokens)
        self.embedding_dimension = int(embedding_dimension)
        self.numerical_count = int(numerical_count)
        self.categorical_cardinalities = tuple(int(value) for value in categorical_cardinalities)
        self._record_token_positions(numerical_token_positions, categorical_token_positions)
        self.target_kind = target_kind
        self.target_cardinality = int(target_cardinality)
        self.transformer = _Transformer(layers, embedding_dimension, heads, ffn_factor)
        self._build_reconstruction_heads(embedding_dimension, numerical_count)
        self.target_head = self._build_target_head(embedding_dimension, target_kind, target_cardinality)
        self.prediction_head = self._build_prediction_head(
            embedding_dimension,
            target_kind,
            target_cardinality,
            prediction_head_hidden,
        )

    def _record_token_positions(self, numerical: list[int] | None, categorical: list[int] | None) -> None:
        """Resolve decoder-head token positions from the fitted schema.

        Args:
            numerical: Optional numerical token positions.
            categorical: Optional categorical token positions.
        """
        self.numerical_token_positions = tuple(range(self.numerical_count) if numerical is None else numerical)
        self.categorical_token_positions = tuple(
            (self.numerical_count + index for index in range(len(self.categorical_cardinalities)))
            if categorical is None
            else categorical
        )

    def _build_reconstruction_heads(self, embedding_dimension: int, numerical_count: int) -> None:
        """Build numerical and categorical reconstruction heads.

        Args:
            embedding_dimension: Width of each latent token.
            numerical_count: Number of numerical scalar outputs.
        """
        # Numerical tokens use feature-specific vectors without additive bias.
        self.numerical_weight = nn.Parameter(torch.empty(numerical_count, embedding_dimension))
        nn.init.xavier_uniform_(self.numerical_weight, gain=1 / math.sqrt(2))
        self.categorical_heads = nn.ModuleList()
        for cardinality in self.categorical_cardinalities:
            head = nn.Linear(embedding_dimension, cardinality)
            nn.init.xavier_uniform_(head.weight, gain=1 / math.sqrt(2))
            self.categorical_heads.append(head)

    @staticmethod
    def _build_target_head(
        embedding_dimension: int,
        target_kind: str | None,
        target_cardinality: int,
    ) -> nn.Module | None:
        """Build the optional supervised target reconstruction head.

        Args:
            embedding_dimension: Width of the target latent token.
            target_kind: Classification, regression, or ``None``.
            target_cardinality: Number of target classes.
        """
        if target_kind == "classification":
            return nn.Linear(embedding_dimension, target_cardinality)
        if target_kind == "regression":
            return nn.Linear(embedding_dimension, 1)
        return None

    @staticmethod
    def _build_prediction_head(
        embedding_dimension: int,
        target_kind: str | None,
        target_cardinality: int,
        hidden: int | None,
    ) -> nn.Module | None:
        """Build the compact supervised head used after target diffusion.

        Args:
            embedding_dimension: Width of the target latent token.
            target_kind: Classification, regression, or ``None``.
            target_cardinality: Number of target classes.
            hidden: Optional hidden-layer width.
        """
        if target_kind != "classification" or hidden is None:
            return None
        return nn.Sequential(
            nn.Linear(embedding_dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, target_cardinality),
        )

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Decode latent tokens into numerical, categorical, and target outputs.

        Args:
            latent: Grid shaped ``(batch, tokens, embedding_dimension)``.
        """
        hidden = self.transformer(latent)
        numerical_tokens = hidden[:, self.numerical_token_positions]
        numerical = (numerical_tokens * self.numerical_weight.unsqueeze(0)).sum(dim=-1)
        categorical = [
            head(hidden[:, index]) for index, head in zip(self.categorical_token_positions, self.categorical_heads)
        ]
        target = None
        if self.target_head is not None:
            target = self.target_head(hidden[:, self.n_feature_tokens])
        return {
            "hidden": hidden,
            "numerical": numerical,
            "categorical": categorical,
            "target": target,
        }

    def hidden_representation(self, latent: torch.Tensor, *, layer: int | None = None) -> torch.Tensor:
        """Return the selected decoder Transformer representation.

        Args:
            latent: Grid shaped ``(batch, tokens, embedding_dimension)``.
            layer: Optional zero-based decoder layer. ``None`` selects the final
                decoder output.
        """
        return self.transformer(latent, layer=layer)

    def predict_target(self, normalized_latent: torch.Tensor) -> torch.Tensor:
        """Decode a predictor target directly from its normalized latent token.

        Args:
            normalized_latent: Sampled normalized grid containing a target token.
        """
        head = self.prediction_head or self.target_head
        if head is None:
            raise RuntimeError("This decoder has no target head")
        return head(normalized_latent[:, self.n_feature_tokens])

    @property
    def n_tokens(self) -> int:
        """Return the decoder token count including an optional target token."""
        return self.n_feature_tokens + int(self.target_head is not None)


class VariableColumnDenoiser(nn.Module):
    """Score-based Diffusion Transformer for variable token grids.

    ``feature_identity`` is persistent model state initialized from
    ``random_state`` so a token keeps its identity across batches and samples.
    The internal observation-mask polarity is explicit: ``True`` marks an
    observed token that must be clamped by the sampler.  Its embedding starts
    at zero so unconditional legacy weights retain their original path.
    """

    def __init__(
        self,
        *,
        n_tokens: int,
        embedding_dimension: int,
        layers: int = 4,
        heads: int = 4,
        ffn_factor: int = 16,
    ) -> None:
        """Build the diffusion transformer and persistent token identities.

        Args:
            n_tokens: Fitted latent token count.
            embedding_dimension: Width of each latent token.
            layers: Denoiser Transformer layer count.
            heads: Denoiser attention-head count.
            ffn_factor: Denoiser feed-forward width factor.
        """
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.embedding_dimension = int(embedding_dimension)
        self.timestep_embed = SinusoidalTimestepEmbedding(embedding_dimension)
        # Zero initialization keeps legacy unconditional weights equivalent when
        # the observation-mask path is first introduced.
        self.observation_mask_embedding = nn.Parameter(torch.zeros(self.n_tokens, embedding_dimension))
        self.transformer = _Transformer(layers, embedding_dimension, heads, ffn_factor)
        # Feature identities are initialized on the first device forward.
        self.register_buffer("feature_identity", torch.zeros(self.n_tokens, self.embedding_dimension))
        self.register_buffer("feature_identity_initialized", torch.tensor(False))

    def forward(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict denoising updates for a variable-column latent grid.

        Args:
            noisy: Noisy latent grid.
            sigma: Per-row noise levels.
            observation_mask: Optional token mask where ``True`` means observed.
        """
        if noisy.ndim != 3 or noisy.shape[1] != self.n_tokens or noisy.shape[2] != self.embedding_dimension:
            raise ValueError(
                f"Expected noisy latents (batch, {self.n_tokens}, {self.embedding_dimension}), got {tuple(noisy.shape)}"
            )
        sigma = sigma.reshape(-1).to(noisy)
        self._initialize_feature_identity(noisy)
        time_input = torch.zeros(
            (len(noisy), self.embedding_dimension),
            device=noisy.device,
            dtype=noisy.dtype,
        )
        time = self.timestep_embed(time_input, torch.log(sigma.clamp_min(1e-8)) / 4.0).unsqueeze(1)
        result = noisy + time
        result = result + self.feature_identity.to(noisy).unsqueeze(0)
        if observation_mask is not None:
            if observation_mask.shape != noisy.shape[:2]:
                raise ValueError("observation_mask must have shape (batch, n_tokens)")
            result = result + observation_mask.to(noisy).unsqueeze(-1) * self.observation_mask_embedding.to(noisy)
        return self.transformer(result)

    def _initialize_feature_identity(self, noisy: torch.Tensor) -> None:
        """Initialize fixed unit-scale feature identities once.

        Args:
            noisy: First latent batch supplying device and dtype.
        """
        if not bool(self.feature_identity_initialized):
            self.feature_identity.copy_(torch.randn_like(self.feature_identity).to(noisy))
            self.feature_identity_initialized.fill_(True)
