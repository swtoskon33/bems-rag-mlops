"""Unit tests for the deterministic hashing embedder."""
import numpy as np
import pytest

from bems_rag.retrieval.embeddings import HashingEmbedder


@pytest.mark.unit
def test_deterministic_same_text_same_vector():
    e = HashingEmbedder(dim=128)
    v = e.embed(["chiller maintenance", "chiller maintenance"])
    assert np.array_equal(v[0], v[1])


@pytest.mark.unit
def test_output_shape_and_dtype():
    e = HashingEmbedder(dim=128)
    v = e.embed(["a", "b", "c"])
    assert v.shape == (3, 128)
    assert v.dtype == np.float32


@pytest.mark.unit
def test_l2_normalised():
    e = HashingEmbedder(dim=128)
    v = e.embed(["solar wind energy"])
    assert np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-5)
