from transformers import AutoTokenizer
import sqlite3
import numpy as np
from tiny_dashboard import OfflineFeatureCentricDashboard
from tiny_dashboard.dashboard_implementations import CrosscoderOnlineFeatureDashboard
from huggingface_hub import hf_hub_download
import torch as th

def stats_repo_id(crosscoder):
    return f"science-of-finetuning/diffing-stats-{crosscoder}"

class QuantileExamplesDB:
    """A persistent, read-only dictionary-like interface for quantile examples database."""

    def __init__(self, db_path, tokenizer, max_example_per_quantile=20):
        """Initialize the database connection.

        Args:
            db_path: Path to the SQLite database
        """
        # Use URI format with read-only mode for better concurrent access
        self.db_path = f"file:{db_path}?mode=ro"
        # Create connection with URI mode enabled
        self.conn = sqlite3.connect(self.db_path, uri=True)

        # Cache the feature indices for faster access
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT feature_idx FROM quantile_examples")
        self._feature_indices = frozenset(row[0] for row in cursor.fetchall())

        cursor.execute("SELECT COUNT(DISTINCT quantile_idx) FROM quantile_examples")
        self.num_quantiles = cursor.fetchone()[0]
        cursor.close()
        self.max_example_per_quantile = max_example_per_quantile
        self.tokenizer = tokenizer

    def __getitem__(self, feature_idx):
        """Get examples for a specific feature index.

        Returns:
            List of tuples (max_activation_value, token_ids, activation_values)
        """
        return self.get_examples(feature_idx)

    def get_examples(self, feature_idx, quantile=None):
        """Get examples for a specific feature index.

        Args:
            feature_idx: The feature index to get examples for
            quantile: The quantile to get examples for (default is all)

        Returns:
            List of tuples (max_activation_value, token_ids, activation_values)
        """
        if feature_idx not in self._feature_indices:
            raise KeyError(f"Feature index {feature_idx} not found in database")

        cursor = self.conn.cursor()
        if quantile is None:
            quantile = list(range(self.num_quantiles))[::-1]
        if isinstance(quantile, list):
            res = []
            for q in quantile:
                res.extend(self.get_examples(feature_idx, q))
            return res
        else:
            # If quantile is specified as an integer, filter by that quantile
            cursor.execute(
                """
                SELECT q.activation, q.sequence_idx, s.token_ids, a.positions, a.activation_values
                FROM quantile_examples q
                JOIN sequences s ON q.sequence_idx = s.sequence_idx
                JOIN activation_details a ON q.feature_idx = a.feature_idx AND q.sequence_idx = a.sequence_idx
                WHERE q.feature_idx = ? AND q.quantile_idx = ?
                ORDER BY q.activation DESC
                """,
                (feature_idx, int(quantile)),
            )

        results = []
        fetch = (
            cursor.fetchmany(self.max_example_per_quantile)
            if self.max_example_per_quantile is not None
            else cursor.fetchall()
        )
        for (
            activation,
            sequence_idx,
            token_ids_blob,
            positions_blob,
            values_blob,
        ) in fetch:
            token_ids = np.frombuffer(token_ids_blob, dtype=np.int32).tolist()
            positions = np.frombuffer(positions_blob, dtype=np.int32).tolist()
            values = np.frombuffer(values_blob, dtype=np.float32).tolist()

            # Initialize activation values with zeros
            activation_values = [0.0] * len(token_ids)

            # Fill in the non-zero activations
            for pos, val in zip(positions, values):
                activation_values[pos] = val

            results.append(
                (
                    activation,
                    self.tokenizer.convert_ids_to_tokens(token_ids),
                    activation_values,
                )
            )

        cursor.close()
        return results

    def keys(self):
        """Get all feature indices in the database."""
        return self._feature_indices

    def __iter__(self):
        """Iterate over feature indices."""
        return iter(self._feature_indices)

    def __len__(self):
        """Get the number of unique features."""
        return len(self._feature_indices)

    def __contains__(self, feature_idx):
        """Check if a feature index exists in the database."""
        return feature_idx in self._feature_indices


def offline_dashboard(crosscoder, max_example_per_quantile=20, tokenizer=None):
    """
    Returns an offline_dashboard showing activations from different quantile

    Args:
      crosscoder: The crosscoder to take the max activating examples
      max_example_per_quantile: the maximimum number of examples per quantile
    """
    db_path = hf_hub_download(
        repo_id=stats_repo_id(crosscoder),
        repo_type="dataset",
        filename="examples.db",
    )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    activation_examples = QuantileExamplesDB(
        db_path, tokenizer, max_example_per_quantile=max_example_per_quantile
    )
    dashboard = OfflineFeatureCentricDashboard(
        activation_examples,
        tokenizer,
        max_examples=activation_examples.num_quantiles * max_example_per_quantile,
    )
    dashboard.display()
    return dashboard

def online_dashboard(
    crosscoder=None,
    max_acts=None,
    crosscoder_device="auto",
    base_device="auto",
    chat_device="auto",
    torch_dtype=th.bfloat16,
):
    """
    Instantiate an online dashboard for crosscoder latent analysis.

    Args:
        crosscoder: the crosscoder to use
        max_acts: a dictionary of max activations for each latent. If None, will be loaded from the latent_df of the crosscoder.
    """
    coder = load_dictionary_model(crosscoder)
    if crosscoder_device == "auto":
        crosscoder_device = "cuda:0" if th.cuda.is_available() else "cpu"
    coder = coder.to(crosscoder_device)
    if max_acts is None:
        df = _latent_df(crosscoder)
        max_acts_cols = ["max_act", "lmsys_max_act"]
        for col in max_acts_cols:
            if col in df.columns:
                max_acts = df[col].dropna().to_dict()
                break
    base_model = load_model(
        "google/gemma-2-2b",
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        device_map=base_device,
    )
    chat_model = load_model(
        "google/gemma-2-2b-it",
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        device_map=chat_device,
    )
    return CrosscoderOnlineFeatureDashboard(
        base_model,
        chat_model,
        coder,
        13,
        max_acts=max_acts,
        crosscoder_device=crosscoder_device,
    )
