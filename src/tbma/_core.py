"""Numerical implementation of Tree-Based Moving Average (TBMA).

The public estimator in :mod:`tbma.model` validates user input and owns the
scikit-learn-style API. This module contains the forest, pooled moving-average,
prediction, and feature calculations.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from math import ceil

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Day, Week
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

_UINT32_MODULUS = 2**32


class TBMACore:
    """Internal numerical engine used by :class:`tbma.TBMA`."""

    def __init__(
        self,
        *,
        ma_order: int | Sequence[int],
        rf_params: dict,
        random_state: int,
        frequency: str,
        seasonal_period: int,
    ) -> None:
        self.ma_order = ma_order
        self.random_state = random_state
        self.random_gen = np.random.RandomState(
            seed=(random_state * 2 + 1) % _UINT32_MODULUS
        )
        self.rf_params = deepcopy(rf_params)
        self.rf_params["random_state"] = random_state
        self.frequency = frequency
        self.seasonal_period = seasonal_period

        self.tree_ensemble = None
        self.tree_ma_dict = None
        self.tree_parent_map = None
        self._train_X = None
        self._train_y = None
        self._train_dates = None
        self._pca_cache = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> None:
        """Fit the random forest and build the node-level PMA curves."""
        self._train_X = X.copy()
        self._train_y = np.asarray(y, dtype=np.float64)
        self._train_dates = pd.DatetimeIndex(dates)
        self._pca_cache = {}

        self.tree_ensemble = RandomForestRegressor(**self.rf_params)
        self.tree_ensemble.fit(X=self._train_X, y=self._train_y)

        self.tree_ma_dict = self._calculate_ma_curves()
        self.tree_parent_map = self._map_child_parent()

    def predict(
        self,
        X: pd.DataFrame,
        dates: pd.DatetimeIndex,
        *,
        horizon: int,
    ) -> pd.DataFrame:
        """Calculate the standalone TBMA readout for each forecast horizon."""
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("horizon must be positive")

        tree_values = self._tree_feature_array(X, dates, feature_window=horizon)
        prediction = np.full((len(X), horizon), np.nan, dtype=np.float64)
        for position in range(horizon):
            window_df = pd.DataFrame(tree_values[:, :, position], index=X.index)
            prediction[:, position] = window_df.mean(axis=1).to_numpy()

        columns = [f"horizon_{position}" for position in range(1, horizon + 1)]
        return pd.DataFrame(prediction, columns=columns, index=X.index)

    def _generate_tree_features(
        self,
        X: pd.DataFrame,
        dates: pd.DatetimeIndex,
        *,
        feature_window: int,
    ) -> pd.DataFrame:
        """Return the full tree-level TBMA representation.

        ``feature_1`` corresponds to the earliest reference position in the
        seasonally aligned feature window and ``feature_window`` to the latest.
        Columns are grouped by tree and stored with feature positions in
        descending order.
        """
        feature_window = int(feature_window)
        if feature_window < 1:
            raise ValueError("feature_window must be positive")

        values = self._tree_feature_array(X, dates, feature_window=feature_window)
        columns = [
            f"tbma_tree_{tree_idx}_feature_{position + 1}"
            for tree_idx in range(values.shape[1])
            for position in range(feature_window - 1, -1, -1)
        ]
        ordered = values[:, :, ::-1].reshape(len(X), -1)
        return pd.DataFrame(ordered, columns=columns, index=X.index)

    def generate_quantile_features(
        self,
        X: pd.DataFrame,
        dates: pd.DatetimeIndex,
        *,
        feature_window: int,
    ) -> pd.DataFrame:
        """Summarize tree-level features with the 25th, 50th, and 75th percentiles."""
        feature_window = int(feature_window)
        if feature_window < 1:
            raise ValueError("feature_window must be positive")

        values = self._tree_feature_array(X, dates, feature_window=feature_window)
        output = {}
        for quantile, label in ((0.25, "q25"), (0.5, "q50"), (0.75, "q75")):
            for position in range(feature_window):
                window_df = pd.DataFrame(values[:, :, position], index=X.index)
                output[f"tbma_feature_{position + 1}_{label}"] = window_df.quantile(
                    q=quantile, axis=1
                )
        return pd.DataFrame(output, index=X.index)

    def generate_pca_features(
        self,
        X: pd.DataFrame,
        dates: pd.DatetimeIndex,
        *,
        feature_window: int,
        pca_components: float,
        include_mean: bool,
    ) -> pd.DataFrame:
        """Summarize tree-level features using training-fitted PCA components."""
        feature_window = int(feature_window)
        if feature_window < 1:
            raise ValueError("feature_window must be positive")

        values = self._tree_feature_array(X, dates, feature_window=feature_window)
        (
            pca_models,
            component_counts,
            train_values,
            train_transformed,
        ) = self._pca_models(
            feature_window=feature_window,
            pca_components=pca_components,
        )

        output: dict[str, pd.Series | np.ndarray] = {}
        if include_mean:
            for position in range(feature_window):
                window_df = pd.DataFrame(values[:, :, position], index=X.index)
                output[f"tbma_feature_{position + 1}_mean"] = window_df.mean(axis=1)

        is_training_representation = (
            len(X) == len(self._train_X)
            and X.index.equals(self._train_X.index)
            and np.array_equal(values, train_values, equal_nan=True)
        )

        transformed_by_position: list[np.ndarray] = []
        for position in range(feature_window):
            if is_training_representation:
                transformed_by_position.append(train_transformed[position].copy())
                continue

            window_df = pd.DataFrame(values[:, :, position], index=X.index)
            complete_mask = ~window_df.isna().any(axis=1)
            n_components = component_counts[position]
            transformed = np.full((len(X), n_components), np.nan, dtype=np.float64)
            if complete_mask.any():
                transformed[complete_mask.to_numpy()] = pca_models[position].transform(
                    window_df.loc[complete_mask]
                )
            transformed_by_position.append(transformed)

        if isinstance(pca_components, (int, np.integer)):
            for component in range(int(pca_components)):
                for position in range(feature_window):
                    output[f"tbma_feature_{position + 1}_pc{component + 1}"] = (
                        transformed_by_position[position][:, component]
                    )
        else:
            # Variance-threshold PCA can select a different number of components
            # at each position. Keep positions in the same descending order used
            # by the original TBMA feature calculation.
            for position in range(feature_window - 1, -1, -1):
                for component in range(component_counts[position]):
                    output[f"tbma_feature_{position + 1}_pc{component + 1}"] = (
                        transformed_by_position[position][:, component]
                    )

        return pd.DataFrame(output, index=X.index)

    def _pca_models(
        self,
        *,
        feature_window: int,
        pca_components: float,
    ) -> tuple[list[PCA], list[int], np.ndarray, list[np.ndarray]]:
        """Fit and cache one PCA basis per feature position using training rows."""
        cache_key = (feature_window, float(pca_components))
        if cache_key in self._pca_cache:
            return self._pca_cache[cache_key]

        train_values = self._tree_feature_array(
            self._train_X,
            self._train_dates,
            feature_window=feature_window,
        )
        pca_seed = (self.random_state * 4 + 1) % _UINT32_MODULUS
        models: list[PCA] = []
        component_counts: list[int] = []
        training_transformed: list[np.ndarray] = []

        for position in range(feature_window):
            window_df = pd.DataFrame(
                train_values[:, :, position], index=self._train_X.index
            )
            complete = window_df.loc[~window_df.isna().any(axis=1)]
            if complete.empty:
                raise ValueError(
                    "PCA summary cannot be fitted because the training TBMA "
                    f"features for feature position {position + 1} contain no "
                    "complete rows"
                )

            max_components = min(complete.shape)
            if isinstance(pca_components, (int, np.integer)):
                n_components = int(pca_components)
                if n_components > max_components:
                    raise ValueError(
                        "pca_components cannot exceed the number of trees or "
                        "complete training rows; "
                        f"feature position {position + 1} supports at most "
                        f"{max_components} components"
                    )
            else:
                initial_pca = PCA(random_state=pca_seed)
                initial_pca.fit(complete)
                cumulative = np.cumsum(initial_pca.explained_variance_ratio_)
                if np.isnan(cumulative).all():
                    n_components = 1
                else:
                    n_components = int(
                        np.searchsorted(cumulative, pca_components) + 1
                    )

            pca = PCA(n_components=n_components, random_state=pca_seed)
            transformed_complete = pca.fit_transform(complete)
            transformed = np.full(
                (len(self._train_X), n_components), np.nan, dtype=np.float64
            )
            complete_mask = ~window_df.isna().any(axis=1)
            transformed[complete_mask.to_numpy()] = transformed_complete
            models.append(pca)
            component_counts.append(n_components)
            training_transformed.append(transformed)

        result = (models, component_counts, train_values, training_transformed)
        self._pca_cache[cache_key] = result
        return result

    def _tree_feature_array(
        self,
        X: pd.DataFrame,
        dates: pd.DatetimeIndex,
        *,
        feature_window: int,
    ) -> np.ndarray:
        """Return tree-level PMA values as rows x trees x feature positions."""
        feature_window = int(feature_window)
        if feature_window < 1:
            raise ValueError("feature_window must be positive")

        input_nodes = self.tree_ensemble.apply(X)
        n_trees = self.tree_ensemble.n_estimators
        reference_lag = self._reference_lag(feature_window)
        input_dates = pd.DatetimeIndex(dates)
        reference_dates = [
            self._shift_dates_by_periods(input_dates, position - reference_lag)
            for position in range(feature_window)
        ]

        values = np.full(
            (len(X), n_trees, feature_window), np.nan, dtype=np.float64
        )
        for tree_idx in range(n_trees):
            ma_arr, ma_dates = self._node_ma_lookup(tree_idx)
            for position in range(feature_window):
                values[:, tree_idx, position] = self._gather_node_ma(
                    ma_arr,
                    ma_dates,
                    reference_dates[position],
                    input_nodes[:, tree_idx],
                )
        return values

    def _reference_lag(self, window: int) -> int:
        """Return the smallest whole seasonal period covering ``window``."""
        return ceil(window / self.seasonal_period) * self.seasonal_period

    def _shift_dates_by_periods(
        self,
        date_arr: np.ndarray | pd.DatetimeIndex,
        n_periods: int,
    ) -> np.ndarray:
        """Shift timestamps by fitted data-frequency periods.

        Daily and weekly shifts use calendar offsets so timezone-aware timestamps
        retain their local wall-clock time across daylight-saving transitions.
        Other supported pandas offsets retain their native calendar or tick
        semantics.
        """
        offset = to_offset(self.frequency)
        date_idx = pd.DatetimeIndex(date_arr)
        periods = int(n_periods)

        if isinstance(offset, Day):
            shifted = date_idx + pd.DateOffset(days=offset.n * periods)
        elif isinstance(offset, Week):
            shifted = date_idx + pd.DateOffset(weeks=offset.n * periods)
        else:
            shifted = date_idx + periods * offset
        return shifted.to_numpy()

    def _node_ma_lookup(self, tree_idx: int):
        """Return a tree's as-of node-PMA matrix and its observation dates."""
        tree_col = f"tree_{tree_idx}"
        ma_df = self.tree_ma_dict[tree_col]
        ma_arr = ma_df.ffill().to_numpy(dtype=np.float64)
        ma_date_arr = ma_df.index.to_numpy()

        valid_row_arr = np.flatnonzero(~np.isnan(ma_arr).all(axis=1))
        first_valid_row = int(valid_row_arr[0]) if len(valid_row_arr) else len(ma_arr)

        parent_map = self.tree_parent_map[tree_col]
        for node_idx in range(1, ma_arr.shape[1]):
            parent_idx = parent_map.get(node_idx)
            if parent_idx is None:
                continue
            node_col_arr = ma_arr[first_valid_row:, node_idx]
            na_mask = np.isnan(node_col_arr)
            if na_mask.any():
                node_col_arr[na_mask] = ma_arr[first_valid_row:, parent_idx][na_mask]

        return ma_arr, ma_date_arr

    @staticmethod
    def _gather_node_ma(
        ma_arr,
        ma_date_arr,
        reference_date_arr,
        node_arr,
    ):
        """Return each row's node PMA available at its reference timestamp."""
        row_idx_arr = np.searchsorted(ma_date_arr, reference_date_arr, side="right") - 1
        out_arr = np.full(len(row_idx_arr), np.nan)
        ok_mask = row_idx_arr >= 0
        out_arr[ok_mask] = ma_arr[row_idx_arr[ok_mask], node_arr[ok_mask]]
        return out_arr

    def _rolling_sum_by_frequency(
        self,
        arr: np.ndarray,
        date_index: pd.DatetimeIndex,
        window: int,
    ) -> np.ndarray:
        """Return sums over ``(date - window * frequency, date]``."""
        boundary_arr = self._shift_dates_by_periods(date_index, -int(window))
        date_arr = date_index.to_numpy()
        left_idx_arr = np.searchsorted(date_arr, boundary_arr, side="right")

        prefix = np.empty((arr.shape[0] + 1, arr.shape[1]), dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(arr, axis=0, dtype=np.float64, out=prefix[1:])
        right_idx_arr = np.arange(1, arr.shape[0] + 1)
        return prefix[right_idx_arr] - prefix[left_idx_arr]

    def _calculate_ma_curves(self):
        """Build the pooled moving-average curve for every node in every tree."""
        sort_order = np.argsort(self._train_dates.to_numpy(), kind="stable")
        sorted_fit_positions = sort_order
        x_arr = self._train_X.iloc[sort_order].to_numpy(dtype=np.float32)
        y_arr = self._train_y[sort_order]
        date_arr = self._train_dates.to_numpy()[sort_order]
        ensemble_inbag_arr = self.tree_ensemble.estimators_samples_

        rf_node_ma_dict = {}
        for tree_idx, estimator in enumerate(self.tree_ensemble.estimators_):
            if isinstance(self.ma_order, list):
                node_window = int(self.random_gen.choice(self.ma_order))
            else:
                node_window = int(self.ma_order)

            inbag_mask = self._inbag_row_mask(
                ensemble_inbag_arr[tree_idx],
                sorted_fit_positions,
            )
            row_arr = np.flatnonzero(inbag_mask)
            if len(row_arr) == 0:
                raise RuntimeError(f"Tree {tree_idx} has no in-bag rows")

            # Only unique in-bag training rows contribute to a tree's PMA curves.
            path = estimator.decision_path(x_arr[row_arr]).astype(np.float64)

            # Pool node counts and target sums by timestamp before applying the
            # one-sided temporal window.
            unique_dates, date_codes = np.unique(date_arr[row_arr], return_inverse=True)
            date_group = sparse.csr_matrix(
                (
                    np.ones(len(row_arr)),
                    (date_codes, np.arange(len(row_arr))),
                ),
                shape=(len(unique_dates), len(row_arr)),
            )
            count_arr = (date_group @ path).toarray()
            value_arr = (
                date_group @ path.multiply(y_arr[row_arr][:, None]).tocsr()
            ).toarray()

            date_index = pd.DatetimeIndex(unique_dates)
            rolling_count_arr = self._rolling_sum_by_frequency(
                count_arr, date_index, node_window
            )
            rolling_value_arr = self._rolling_sum_by_frequency(
                value_arr, date_index, node_window
            )

            with np.errstate(invalid="ignore", divide="ignore"):
                ma_arr = rolling_value_arr / rolling_count_arr

            # A node has no newly calculated PMA on a timestamp where it receives
            # no in-bag observation. Lookup carries the last same-node PMA forward
            # and then falls back through parent nodes when needed.
            ma_arr[count_arr == 0] = np.nan

            tree_ma_df = pd.DataFrame(ma_arr, index=date_index)
            tree_ma_df.columns = [
                f"tree{tree_idx}_node{node_idx}"
                for node_idx in range(tree_ma_df.shape[1])
            ]
            rf_node_ma_dict[f"tree_{tree_idx}"] = tree_ma_df

        return rf_node_ma_dict

    @staticmethod
    def _inbag_row_mask(inbag_idx, sorted_fit_positions):
        """Map forest bootstrap positions onto rows after date sorting."""
        return np.isin(sorted_fit_positions, np.unique(inbag_idx))

    def _map_child_parent(self):
        """Map each non-root decision-tree node to its parent node."""
        tree_parent_map = {}
        for tree_idx, tree in enumerate(self.tree_ensemble.estimators_):
            tree_key = f"tree_{tree_idx}"
            tree_parent_map[tree_key] = {}
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            stack = [0]
            while stack:
                node_id = stack.pop()
                left_child_idx = children_left[node_id]
                right_child_idx = children_right[node_id]
                if left_child_idx != right_child_idx:
                    tree_parent_map[tree_key][left_child_idx] = node_id
                    tree_parent_map[tree_key][right_child_idx] = node_id
                    stack.append(left_child_idx)
                    stack.append(right_child_idx)

        return tree_parent_map
