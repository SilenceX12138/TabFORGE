"""Mixed-table preprocessing and schema-aware reconstruction for TabFORGE.

The processor owns the stable numerical-first layout shared by decoder losses,
generation, imputation masks, and checkpoint schemas. It preserves original
pandas column order, dtypes, and categorical domains at public boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from tabcamel.data.transform import (
    CategoryTransform,
    NumericTransform,
    SimpleImputeTransform,
)
from tabeval.plugins.core.models.factory import FEATURE_ENCODERS
from tabeval.plugins.core.models.feature_encoder import FeatureEncoder
from tabeval.plugins.core.models.tabular_encoder import TabularEncoder


def _quantile_count(rows: int) -> int:
    """Use data-scaled Gaussian quantile resolution.

    Args:
        rows: Number of fitted training rows.
    """
    return max(min(rows // 30, 1000), 10)


class GaussianQuantileFeatureEncoder(FeatureEncoder):
    """Pickle-safe equivalent of tabeval's Gaussian quantile wrapper."""

    n_dim_in = 2

    def __init__(self) -> None:
        """Create the Gaussian quantile transform used for continuous columns."""

        self.encoder = QuantileTransformer(
            n_quantiles=10,
            output_distribution="normal",
            subsample=int(1e9),
        )

    def _fit(self, x: np.ndarray, **kwargs: Any) -> "GaussianQuantileFeatureEncoder":
        """Fit a row-scaled quantile transform and return this encoder.

        Args:
            x: Two-dimensional continuous feature values.
            **kwargs: Keyword arguments forwarded to scikit-learn ``fit``.
        """

        self.encoder.n_quantiles = _quantile_count(len(x))
        self.encoder.fit(x, **kwargs)
        return self

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """Map a continuous column into Gaussian reconstruction space.

        Args:
            x: Continuous values in the fitted source scale.
        """

        return self.encoder.transform(x)

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Map Gaussian reconstruction values back to the fitted scale.

        Args:
            data: Values in Gaussian reconstruction space.
        """

        return self.encoder.inverse_transform(data)

    def get_feature_names_out(self) -> list[str]:
        """Return the output name required by the TabEval encoder interface."""

        return list(self.encoder.get_feature_names_out([self.feature_name_in]))


FEATURE_ENCODERS["tabforgequantile"] = GaussianQuantileFeatureEncoder


@dataclass(frozen=True)
class FeatureSchema:
    """Describe the fitted raw columns and categorical domains."""

    names: tuple[Any, ...]
    dtypes: tuple[str, ...]
    numerical_indices: tuple[int, ...]
    categorical_indices: tuple[int, ...]
    categorical_cardinalities: tuple[int, ...]
    categorical_domains: tuple[tuple[Any, ...], ...]

    @property
    def n_features(self) -> int:
        """Return the number of raw feature columns."""
        return len(self.names)


class TabularFeatureProcessor:
    """Own the reconstruction space and raw-table schema for a TabFORGE fit.

    The reconstruction vector uses all numerical columns first, followed by a
    dense one-hot span for each categorical column. This stable layout keeps
    decoder heads independent from the order in which a backend represents
    categorical values.
    """

    def __init__(self, categorical_features: Sequence[int | str] | None = None) -> None:
        """Configure explicit categorical columns when schema inference is unsuitable.

        Args:
            categorical_features: Raw categorical column names or integer positions.
        """
        self.categorical_features = None if categorical_features is None else tuple(categorical_features)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[Any] | pd.Series | np.ndarray | None = None,
    ) -> "TabularFeatureProcessor":
        """Fit feature reconstruction state from the training table.

        Args:
            X: Raw training features.
            y: Optional target fitted after feature preprocessing.

        Returns:
            TabularFeatureProcessor: This fitted processor.
        """
        frame = self._as_frame(X)
        if frame.ndim != 2 or frame.shape[1] == 0:
            raise ValueError("X must contain at least one feature column")
        self._record_input_schema(X, frame)
        self.input_categorical_indices_ = tuple(self._resolve_input_categorical_indices(frame))
        self._fit_input_preprocessor(frame)
        model_frame = self._preprocess_input(frame)
        self._fit_reconstruction_encoder(model_frame)
        self._build_feature_schema(frame)
        self._build_reconstruction_layout()
        if y is not None:
            self.fit_target(y)
        return self

    def _record_input_schema(self, X: Any, frame: pd.DataFrame) -> None:
        """Record the input container, column names, and dtypes.

        Args:
            X: Original public training table.
            frame: Internal pandas representation of ``X``.
        """
        self._is_pandas_ = isinstance(X, pd.DataFrame)
        self._input_dtypes_ = tuple(frame.dtypes)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object) if self._is_pandas_ else None
        self.n_features_in_ = frame.shape[1]
        self._feature_index_ = tuple(frame.columns)
        self._working_feature_index_ = tuple(f"__tabforge_feature_{index}" for index in range(frame.shape[1]))

    def _fit_input_preprocessor(self, frame: pd.DataFrame) -> None:
        """Fit tabcamel's imputation, encoding, and scaling pipeline.

        Args:
            frame: Internal raw training frame.
        """
        self.input_numerical_indices_ = tuple(
            index for index in range(self.n_features_in_) if index not in self.input_categorical_indices_
        )
        self._model_feature_raw_indices_ = self.input_numerical_indices_ + self.input_categorical_indices_
        self._model_feature_index_ = tuple(
            self._working_feature_index_[index] for index in self._model_feature_raw_indices_
        )
        frame = frame.copy()
        frame.columns = self._working_feature_index_
        categorical = [frame.columns[index] for index in self.input_categorical_indices_]
        numerical = [frame.columns[index] for index in self.input_numerical_indices_]
        self.input_transformers_ = (
            SimpleImputeTransform(
                categorical_feature_list=categorical,
                numerical_feature_list=numerical,
                strategy_categorical="most_frequent",
                strategy_numerical="mean",
            ),
            CategoryTransform(categorical_feature_list=categorical, strategy="ordinal"),
            NumericTransform(
                numerical_feature_list=numerical,
                strategy="standard",
                include_categorical=False,
                train_num_samples=len(frame),
            ),
        )
        transformed = frame.copy()
        for transformer in self.input_transformers_:
            transformer.fit(transformed)
            transformed = transformer.transform(transformed)

        imputed = self.input_transformers_[0].transform(frame)
        self.input_numeric_support_ = tuple(
            np.unique(pd.to_numeric(imputed[column], errors="coerce").to_numpy(dtype=np.float64))
            for column in numerical
        )

    def _preprocess_input(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply tabcamel preprocessing and numerical-first column order.

        Args:
            frame: Raw frame matching the fitted schema.
        """
        transformed = frame.copy()
        transformed.columns = self._working_feature_index_
        for transformer in self.input_transformers_:
            transformed = transformer.transform(transformed)
        return transformed.loc[:, list(self._model_feature_index_)]

    def _inverse_input_preprocessing(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Invert preprocessing and restore float-perturbed support values.

        Args:
            frame: Numerical-first frame in the processed model space.
        """
        restored = frame.copy()
        for position, raw_index in enumerate(self.input_categorical_indices_):
            column = self._working_feature_index_[raw_index]
            cardinality = len(self.input_transformers_[1].categories_[position])
            restored[column] = np.clip(
                np.rint(pd.to_numeric(restored[column], errors="coerce").fillna(-1)),
                0,
                cardinality - 1,
            )
        for transformer in reversed(self.input_transformers_):
            restored = transformer.inverse_transform(restored)
        for position, raw_index in enumerate(self.input_numerical_indices_):
            column = self._working_feature_index_[raw_index]
            restored[column] = self._restore_numeric_support_precision(
                restored[column].to_numpy(), self.input_numeric_support_[position]
            )
        restored.columns = self._feature_index_
        return restored.loc[:, list(self._feature_index_)]

    @staticmethod
    def _restore_numeric_support_precision(values: np.ndarray, support: np.ndarray) -> np.ndarray:
        """Restore fitted values perturbed only by float32 reconstruction error.

        Args:
            values: Reconstructed numerical values.
            support: Unique values observed during fitting.
        """
        values = np.asarray(values, dtype=np.float64).copy()
        if len(support) == 0:
            return values
        insertion = np.searchsorted(support, values)
        lower = np.clip(insertion - 1, 0, len(support) - 1)
        upper = np.clip(insertion, 0, len(support) - 1)
        lower_distance = np.abs(values - support[lower])
        upper_distance = np.abs(values - support[upper])
        nearest = np.where(lower_distance <= upper_distance, lower, upper)
        exact = support[nearest]
        tolerance = 5e-6 + 4 * np.finfo(np.float32).eps * np.abs(exact)
        restore = np.abs(values - exact) <= tolerance
        values[restore] = exact[restore]
        return values

    def _fit_reconstruction_encoder(self, frame: pd.DataFrame) -> None:
        """Fit tabeval's one-hot/quantile reconstruction encoder.

        Args:
            frame: Preprocessed numerical-first training frame.
        """
        self.reconstruction_encoder_ = TabularEncoder(
            categorical_encoder="onehot",
            cat_encoder_params={"sparse_output": False, "handle_unknown": "ignore"},
            continuous_encoder="tabforge_quantile",
            cont_encoder_params={},
        ).fit(frame)
        layout = self.reconstruction_encoder_.layout()
        self._model_numerical_indices_ = tuple(
            index for index, info in enumerate(layout) if info.feature_type == "continuous"
        )
        self._model_categorical_indices_ = tuple(
            index for index, info in enumerate(layout) if info.feature_type == "discrete"
        )
        self.numerical_indices_ = tuple(
            self._model_feature_raw_indices_[index] for index in self._model_numerical_indices_
        )
        self.categorical_indices_ = tuple(
            self._model_feature_raw_indices_[index] for index in self._model_categorical_indices_
        )
        self.numerical_columns_ = tuple(frame.columns[index] for index in self._model_numerical_indices_)
        self.categorical_columns_ = tuple(frame.columns[index] for index in self._model_categorical_indices_)
        self.categories_ = [
            np.asarray(layout[index].transform.encoder.categories_[0], dtype=object)
            for index in self._model_categorical_indices_
        ]
        self.cardinalities_ = [len(values) for values in self.categories_]

    def _build_feature_schema(self, frame: pd.DataFrame) -> None:
        """Capture the fitted schema used by generation checkpoints.

        Args:
            frame: Original raw training frame.
        """
        categorical_domains = self._raw_categorical_domains()
        self.schema_ = FeatureSchema(
            names=tuple(frame.columns),
            dtypes=tuple(str(dtype) for dtype in frame.dtypes),
            numerical_indices=self.numerical_indices_,
            categorical_indices=self.categorical_indices_,
            categorical_cardinalities=tuple(len(domain) for domain in categorical_domains),
            categorical_domains=categorical_domains,
        )

    def _raw_categorical_domains(self) -> tuple[tuple[Any, ...], ...]:
        """Expose decoder-discrete domains in the public raw feature schema."""
        domains = []
        input_category_position = {
            raw_index: position for position, raw_index in enumerate(self.input_categorical_indices_)
        }
        for raw_index in self.categorical_indices_:
            if raw_index in input_category_position:
                domain = self.input_transformers_[1].categories_[input_category_position[raw_index]]
            else:
                input_position = self.input_numerical_indices_.index(raw_index)
                domain = self.input_numeric_support_[input_position]
            domains.append(tuple(np.asarray(domain, dtype=object)))
        return tuple(domains)

    def _build_reconstruction_layout(self) -> None:
        """Map raw feature tokens onto the dense reconstruction vector."""
        self.feature_token_positions_ = np.arange(self.n_features_in_, dtype=np.int64)
        self.n_feature_tokens_ = self.n_features_in_
        # TabPFN sees tabcamel's outer numerical-first order. Decoder heads use
        # tabeval's feature types at their positions in that outer view.
        self.embedding_token_raw_indices_ = self._model_feature_raw_indices_
        self.numerical_embedding_positions_ = self._model_numerical_indices_
        self.categorical_embedding_positions_ = self._model_categorical_indices_
        self.numerical_reconstruction_positions_ = tuple(range(len(self.numerical_indices_)))
        offset = len(self.numerical_indices_)
        self.categorical_reconstruction_spans_ = tuple(
            tuple(
                range(
                    offset + sum(self.cardinalities_[:i]),
                    offset + sum(self.cardinalities_[: i + 1]),
                )
            )
            for i in range(len(self.cardinalities_))
        )
        self.categorical_reconstruction_positions_ = tuple(
            position for span in self.categorical_reconstruction_spans_ for position in span
        )
        self.reconstruction_dim_ = len(self.numerical_indices_) + sum(self.cardinalities_)

    def fit_target(
        self,
        y: Sequence[Any] | pd.Series | np.ndarray,
        *,
        task: str | None = None,
        regression_transform: str = "quantile",
    ) -> "TabularFeatureProcessor":
        """Fit target reconstruction state for classification or regression.

        Args:
            y: One-dimensional target values.
            task: Explicit target task or ``None`` for cardinality inference.
            regression_transform: ``"quantile"`` or ``"standard"`` scaling.
        """
        target = np.asarray(y)
        if target.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(target) == 0:
            raise ValueError("y must contain at least one row")
        if task not in {None, "classification", "regression"}:
            raise ValueError("task must be None, 'classification', or 'regression'")
        self._target_is_pandas_ = isinstance(y, pd.Series)
        self._target_name_ = y.name if isinstance(y, pd.Series) else None
        self._target_dtype_ = y.dtype if isinstance(y, pd.Series) else None
        self.target_is_classification_ = task == "classification" if task else self._is_classification_target(target)
        if self.target_is_classification_:
            self._fit_classification_target(target)
        else:
            self._fit_regression_target(target, regression_transform)
        self.target_token_position_ = self.n_feature_tokens_
        return self

    def _fit_classification_target(self, target: np.ndarray) -> None:
        """Fit class labels and target reconstruction width.

        Args:
            target: One-dimensional class labels.
        """
        self.target_encoder_ = LabelEncoder().fit(target)
        self.classes_ = self.target_encoder_.classes_.copy()
        self.target_cardinality_ = int(len(self.classes_))
        if self.target_cardinality_ < 2:
            raise ValueError("classification targets must contain at least two classes")
        self.target_reconstruction_dim_ = self.target_cardinality_

    def _fit_regression_target(self, target: np.ndarray, transform: str) -> None:
        """Fit the selected transform for continuous targets.

        Args:
            target: One-dimensional continuous values.
            transform: ``"quantile"`` or ``"standard"``.
        """
        numeric = target.astype(np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("regression targets must be finite")
        if transform == "standard":
            self.target_encoder_ = StandardScaler().fit(numeric.reshape(-1, 1))
        elif transform == "quantile":
            self.target_encoder_ = QuantileTransformer(
                n_quantiles=_quantile_count(len(numeric)),
                output_distribution="normal",
                subsample=int(1e9),
            ).fit(numeric.reshape(-1, 1))
        else:
            raise ValueError("regression_transform must be 'quantile' or 'standard'")
        self.target_cardinality_ = 0
        self.target_reconstruction_dim_ = 1

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform raw features into reconstruction space.

        Args:
            X: Raw features matching the fitted schema.
        """
        return self.transform_features(X)

    def transform_features(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Encode numerical and categorical features for decoder training.

        Args:
            X: Raw features matching the fitted schema.
        """
        self._check_fitted()
        frame = self._preprocess_input(self._check_frame(X))
        encoded = self.reconstruction_encoder_.transform(frame)
        layout = self.reconstruction_encoder_.layout()
        blocks = [
            encoded.loc[:, layout[index].transformed_features].to_numpy(dtype=np.float32)
            for index in self._model_numerical_indices_ + self._model_categorical_indices_
        ]
        return np.concatenate(blocks, axis=1)

    def transform_target(self, y: Sequence[Any] | pd.Series | np.ndarray) -> np.ndarray:
        """Encode a fitted target for decoder training.

        Args:
            y: Target values matching the fitted task.
        """
        self._check_fitted()
        if not hasattr(self, "target_encoder_"):
            raise RuntimeError("The processor was fitted without a target")
        target = np.asarray(y)
        if target.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if self.target_is_classification_:
            encoded = self.target_encoder_.transform(target)
            result = np.zeros((len(target), self.target_cardinality_), dtype=np.float32)
            result[np.arange(len(target)), encoded] = 1.0
            return result
        return self.target_encoder_.transform(target.astype(np.float64).reshape(-1, 1)).astype(np.float32)

    def transform_joint(self, X: pd.DataFrame | np.ndarray, y: Sequence[Any] | None = None) -> np.ndarray:
        """Encode features with an optional supervised target.

        Args:
            X: Raw features matching the fitted schema.
            y: Optional target values aligned with ``X``.
        """
        features = self.transform_features(X)
        if y is None:
            return features
        return np.concatenate([features, self.transform_target(y)], axis=1)

    def inverse_transform(
        self, values: np.ndarray, *, template: pd.DataFrame | np.ndarray | None = None
    ) -> pd.DataFrame | np.ndarray:
        """Decode feature reconstruction values into the original schema.

        Args:
            values: Dense numerical/one-hot reconstruction matrix.
            template: Optional table whose container metadata should be preserved.
        """
        self._check_fitted()
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != self.reconstruction_dim_:
            raise ValueError(
                f"Expected decoded features with shape (n_rows, {self.reconstruction_dim_}), got {values.shape}"
            )
        n_rows = len(values)
        layout = self.reconstruction_encoder_.layout()
        encoded_columns = [column for info in layout for column in info.transformed_features]
        encoded = pd.DataFrame(0.0, index=np.arange(n_rows), columns=encoded_columns)
        for position, model_index in enumerate(self._model_numerical_indices_):
            info = layout[model_index]
            encoded.loc[:, info.transformed_features] = values[:, [position]]
        for model_index, span in zip(self._model_categorical_indices_, self.categorical_reconstruction_spans_):
            info = layout[model_index]
            encoded.loc[:, info.transformed_features] = values[:, span]
        model_frame = self.reconstruction_encoder_.inverse_transform(encoded)
        frame = self._inverse_input_preprocessing(model_frame)
        frame = self._restore_dtypes(frame)
        if isinstance(template, pd.DataFrame):
            frame.index = template.index[:n_rows]
        if self._is_pandas_:
            return frame
        return frame.to_numpy()

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        """Decode target reconstruction values into fitted target values.

        Args:
            values: Classification probabilities/logits or scaled regression values.
        """
        self._check_fitted()
        if not hasattr(self, "target_encoder_"):
            raise RuntimeError("The processor was fitted without a target")
        values = np.asarray(values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if self.target_is_classification_:
            codes = np.argmax(values, axis=1)
            result = self.target_encoder_.inverse_transform(np.clip(codes, 0, self.target_cardinality_ - 1))
        else:
            result = self.target_encoder_.inverse_transform(values[:, :1]).reshape(-1)
        if getattr(self, "_target_is_pandas_", False):
            return pd.Series(result, name=self._target_name_)
        return result

    def inverse_joint(
        self,
        values: np.ndarray,
        *,
        template: pd.DataFrame | np.ndarray | None = None,
    ) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
        """Decode a joint feature-target reconstruction.

        Args:
            values: Dense feature and target reconstruction matrix.
            template: Optional raw feature table supplying container metadata.
        """
        self._check_fitted()
        expected = self.reconstruction_dim_ + self.target_reconstruction_dim_
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(f"Expected decoded joint values with {expected} columns, got {values.shape}")
        return (
            self.inverse_transform(values[:, : self.reconstruction_dim_], template=template),
            self.inverse_target(values[:, self.reconstruction_dim_ :]),
        )

    def missing_mask(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return the missing-cell mask in raw feature order.

        Args:
            X: Raw feature table.
        """
        frame = self._check_frame(X)
        return frame.isna().to_numpy(dtype=bool)

    def restore_observed(
        self,
        original: pd.DataFrame | np.ndarray,
        reconstructed: pd.DataFrame | np.ndarray,
        mask: np.ndarray | None = None,
    ) -> pd.DataFrame | np.ndarray:
        """Restore cells whose public mask is ``False`` or whose input is present.

        Args:
            original: Raw conditioning table.
            reconstructed: Candidate decoded table.
            mask: Raw-column mask where ``True`` allows regeneration.

        The public imputer convention is ``True means regenerate``. The
        processor owns raw-schema reconstruction, so restoration happens after
        decoded numerical/categorical values have been converted back to the
        input table's column order and dtypes.
        """
        source = self._check_frame(original)
        result = self._check_frame(reconstructed).copy()
        observed = ~self.missing_mask(original) if mask is None else ~np.asarray(mask, dtype=bool)
        if observed.shape != (len(source), self.n_features_in_):
            raise ValueError("mask must have the same shape as X")
        source_values = source.to_numpy(dtype=object)
        result_values = result.to_numpy(dtype=object)
        result_values[observed] = source_values[observed]
        restored = pd.DataFrame(result_values, columns=self._feature_index_, index=result.index)
        restored = self._restore_dtypes(restored)
        return restored if self._is_pandas_ else restored.to_numpy()

    def feature_token_count(self, *, include_target: bool = False) -> int:
        """Return the latent token count for the requested task layout.

        Args:
            include_target: Add the supervised target token.
        """
        self._check_fitted()
        return self.n_feature_tokens_ + int(include_target)

    def token_values(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return one deterministic scalar per raw feature token.

        Args:
            X: Raw features matching the fitted schema.

        This is used only by the explicit mock embedding backend. The real
        provider passes the raw table to the vendored TabPFN runtime.
        """
        self._check_fitted()
        frame = self._preprocess_input(self._check_frame(X))
        encoded = self.reconstruction_encoder_.transform(frame)
        values = np.zeros((len(frame), self.n_features_in_), dtype=np.float32)
        for position, info in enumerate(self.reconstruction_encoder_.layout()):
            block = encoded.loc[:, info.transformed_features].to_numpy(dtype=np.float32)
            if info.feature_type == "continuous":
                values[:, position] = block[:, 0]
            else:
                values[:, position] = np.argmax(block, axis=1)
                if info.output_dimensions > 1:
                    values[:, position] /= info.output_dimensions - 1
        return values

    def embedding_values(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Build the tabcamel-processed external TabPFN input view.

        Args:
            X: Raw features matching the fitted schema.
        """
        frame = self._preprocess_input(self._check_frame(X))
        return frame.to_numpy(dtype=np.float32, copy=True)

    def embedding_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """Reorder a raw-column mask to the fitted embedding-token order.

        Args:
            raw_mask: Boolean mask in original raw column order.
        """
        return np.asarray(raw_mask, dtype=bool)[:, self.embedding_token_raw_indices_]

    def target_token_values(self, y: Sequence[Any] | pd.Series | np.ndarray) -> np.ndarray:
        """Return deterministic scalar values for mock target tokens.

        Args:
            y: Target values matching the fitted task.
        """
        self._check_fitted()
        if not hasattr(self, "target_encoder_"):
            raise RuntimeError("The processor was fitted without a target")
        if self.target_is_classification_:
            return self.target_encoder_.transform(np.asarray(y)).astype(np.float32)
        return (
            self.target_encoder_.transform(np.asarray(y).astype(np.float64).reshape(-1, 1))
            .reshape(-1)
            .astype(np.float32)
        )

    def schema_dict(self) -> dict[str, Any]:
        """Return the fitted feature schema as serializable data."""
        self._check_fitted()
        return {
            "names": list(self.schema_.names),
            "dtypes": list(self.schema_.dtypes),
            "numerical_indices": list(self.schema_.numerical_indices),
            "categorical_indices": list(self.schema_.categorical_indices),
            "categorical_cardinalities": list(self.schema_.categorical_cardinalities),
            "categorical_domains": [list(domain) for domain in self.schema_.categorical_domains],
            "embedding_token_raw_indices": list(self.embedding_token_raw_indices_),
        }

    def _as_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Convert supported input containers to an internal DataFrame.

        Args:
            X: Public pandas or NumPy table.
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError("X must be a two-dimensional NumPy array")
            return pd.DataFrame(X, columns=range(X.shape[1]))
        raise TypeError(f"X must be a pandas DataFrame or NumPy array, got {type(X).__name__}")

    def _check_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Validate a table against the fitted raw-column schema.

        Args:
            X: Public pandas or NumPy table.
        """
        frame = self._as_frame(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} feature columns, got {frame.shape[1]}")
        if isinstance(X, pd.DataFrame) and tuple(X.columns) != self._feature_index_:
            raise ValueError("X columns do not match the fitted feature schema")
        frame.columns = self._feature_index_
        return frame

    def _resolve_input_categorical_indices(self, frame: pd.DataFrame) -> list[int]:
        """Resolve raw nominal columns from the public hint or pandas dtypes.

        Args:
            frame: Internal raw training frame.
        """
        if self.categorical_features is not None:
            indices: list[int] = []
            names = list(frame.columns)
            for feature in self.categorical_features:
                if isinstance(feature, (int, np.integer)):
                    index = int(feature)
                    if index < 0 or index >= len(names):
                        raise ValueError(f"categorical feature index {index} is out of range")
                else:
                    if feature not in names:
                        raise ValueError(f"categorical feature {feature!r} is not in X")
                    index = names.index(feature)
                if index not in indices:
                    indices.append(index)
            return indices
        return [
            index
            for index, dtype in enumerate(frame.dtypes)
            if not (pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype))
        ]

    def _restore_dtypes(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Restore generated columns to the fitted pandas dtypes.

        Args:
            frame: Reconstructed raw-schema frame.
        """
        for index, column in enumerate(self._feature_index_):
            dtype = self._input_dtypes_[index]
            try:
                if isinstance(dtype, pd.CategoricalDtype):
                    frame[column] = pd.Categorical(
                        frame[column],
                        categories=dtype.categories,
                        ordered=dtype.ordered,
                    )
                elif pd.api.types.is_integer_dtype(dtype) and index in self.categorical_indices_:
                    frame[column] = np.rint(frame[column].astype(float)).astype(dtype)
                elif pd.api.types.is_integer_dtype(dtype):
                    # Keep decoded numerical values continuous after reconstruction.
                    frame[column] = frame[column].astype(float)
                elif pd.api.types.is_float_dtype(dtype):
                    frame[column] = frame[column].astype(dtype)
                elif pd.api.types.is_bool_dtype(dtype):
                    frame[column] = frame[column].astype(bool)
                elif not pd.api.types.is_object_dtype(dtype):
                    frame[column] = frame[column].astype(dtype)
            except (TypeError, ValueError):
                # A generated value outside an exotic pandas extension dtype is
                # still returned with the correct column and usable object values.
                frame[column] = frame[column].astype(object)
        return frame

    @staticmethod
    def _is_classification_target(target: np.ndarray) -> bool:
        """Infer whether an unspecified target should be class encoded.

        Args:
            target: One-dimensional target values.
        """
        if target.dtype.kind in "OUSb":
            return True
        unique = np.unique(target)
        return len(unique) <= min(20, max(2, int(np.sqrt(max(len(target), 1))))) and np.all(
            np.equal(unique, np.floor(unique))
        )

    def _check_fitted(self) -> None:
        """Require fitted reconstruction state before transformation."""
        if not hasattr(self, "schema_"):
            raise RuntimeError("TabularFeatureProcessor has not been fitted")
