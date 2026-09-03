"""The reranker reorders candidates by domain-aware token overlap.

These tests pin what the reranker actually does, including where it fails. Its synonym
map was written against the dev paraphrases, so it helps there and hurts on held-out
wording; a test asserting a blanket improvement would be asserting the tuning artifact.
"""
import json

import pytest

from bems_rag.ingest.bdg2 import load_bdg2_facet_chunks
from bems_rag.retrieval.reranker import LexicalReranker, Reranker, get_reranker
from bems_rag.retrieval.retriever import Retriever
from bems_rag.types import Query


def _queries(group=None):
    with open("data/sample/golden.json") as f:
        qs = json.load(f)["queries"]
    return [q for q in qs if group is None or q.get("difficulty") == group]


def _retriever(reranker):
    chunks = load_bdg2_facet_chunks("data/bdg2/metadata.csv", limit=25)
    r = Retriever(reranker=reranker, fetch_k=10)
    r.index(chunks)
    return r


def _hit_at_1(retriever, queries):
    hits = 0
    for q in queries:
        res = retriever.retrieve(Query(text=q["text"], building_id=q["building_id"]), k=1)
        got = [rc.chunk.id for rc in res]
        if any(rid in got for rid in q["relevant_ids"]):
            hits += 1
    return hits / len(queries)


@pytest.mark.unit
def test_default_reranker_is_identity():
    assert isinstance(get_reranker(), Reranker)


@pytest.mark.unit
def test_reranker_helps_on_the_wording_its_synonyms_were_written_for():
    qs = _queries("paraphrased_dev")
    base = _hit_at_1(_retriever(Reranker()), qs)
    rank = _hit_at_1(_retriever(LexicalReranker()), qs)
    assert rank > base


@pytest.mark.unit
def test_reranker_does_not_generalise_to_unseen_wording():
    """Documents the finding rather than hiding it: on held-out paraphrases the synonym
    map does not apply and the reranker scores no better than the plain retriever."""
    qs = _queries("paraphrased_heldout")
    base = _hit_at_1(_retriever(Reranker()), qs)
    rank = _hit_at_1(_retriever(LexicalReranker()), qs)
    assert rank <= base


@pytest.mark.unit
def test_reranker_never_hurts_direct_queries():
    qs = _queries("direct")
    base = _hit_at_1(_retriever(Reranker()), qs)
    rank = _hit_at_1(_retriever(LexicalReranker()), qs)
    assert rank >= base


@pytest.mark.unit
def test_reranker_preserves_tenant_isolation():
    r = _retriever(LexicalReranker())
    chunks = load_bdg2_facet_chunks("data/bdg2/metadata.csv", limit=25)
    bid = chunks[0].building_id
    res = r.retrieve(Query(text="how big is this place?", building_id=bid), k=3)
    assert all(rc.chunk.building_id == bid for rc in res)
