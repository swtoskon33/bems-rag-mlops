"""Hybrid retrieval: BM25 works, RRF fuses, tenant isolation holds."""
from bems_rag.ingest.bdg2 import load_bdg2_facet_chunks
from bems_rag.retrieval.hybrid import BM25Retriever, rrf_fuse
from bems_rag.retrieval.retriever import Retriever
from bems_rag.types import Query


def _setup():
    chunks = load_bdg2_facet_chunks("data/bdg2/metadata.csv", limit=25)
    bm25 = BM25Retriever(); bm25.index(chunks)
    dense = Retriever(fetch_k=10); dense.index(chunks)
    return chunks, bm25, dense


def test_bm25_returns_tenant_chunks_only():
    chunks, bm25, _ = _setup()
    bid = chunks[0].building_id
    res = bm25.retrieve(Query(text="floor area", building_id=bid), k=3)
    assert res and all(rc.chunk.building_id == bid for rc in res)


def test_rrf_fuse_combines_and_caps_k():
    chunks, bm25, dense = _setup()
    bid = chunks[0].building_id
    q = Query(text="how big is this place?", building_id=bid)
    fused = rrf_fuse(dense.retrieve(q, 10), bm25.retrieve(q, 10), k=3)
    assert len(fused) <= 3
    assert all(rc.chunk.building_id == bid for rc in fused)
