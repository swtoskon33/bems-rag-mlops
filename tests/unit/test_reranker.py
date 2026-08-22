"""The reranker reorders candidates by domain-aware token overlap."""
from bems_rag.ingest.bdg2 import load_bdg2_facet_chunks
from bems_rag.retrieval.reranker import LexicalReranker, Reranker, get_reranker
from bems_rag.retrieval.retriever import Retriever
from bems_rag.types import Query


def _retriever(reranker):
    chunks = load_bdg2_facet_chunks("data/bdg2/metadata.csv", limit=25)
    r = Retriever(reranker=reranker, fetch_k=10)
    r.index(chunks)
    return r, chunks


def _hit_at_1(retriever, queries):
    hits = 0
    for q in queries:
        res = retriever.retrieve(Query(text=q["text"], building_id=q["building_id"]), k=1)
        got = [rc.chunk.id for rc in res]
        if any(rid in got for rid in q["relevant_ids"]):
            hits += 1
    return hits / len(queries)


def test_default_reranker_is_identity():
    assert isinstance(get_reranker(), Reranker)


def test_reranker_improves_hit_at_1():
    import json
    with open("data/sample/golden.json") as f:
        queries = json.load(f)["queries"]
    base, _ = _retriever(Reranker())
    rank, _ = _retriever(LexicalReranker())
    base_hit = _hit_at_1(base, queries)
    rank_hit = _hit_at_1(rank, queries)
    # the reranker should not hurt, and on this golden set it clearly helps
    assert rank_hit >= base_hit
    assert rank_hit > 0.80


def test_reranker_preserves_tenant_isolation():
    rank, chunks = _retriever(LexicalReranker())
    bid = chunks[0].building_id
    res = rank.retrieve(Query(text="how big is this place?", building_id=bid), k=3)
    assert all(rc.chunk.building_id == bid for rc in res)
