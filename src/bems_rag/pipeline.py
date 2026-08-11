"""End-to-end RAG pipeline: retrieve -> generate -> grounded answer.

This is the single entry point the serving layer and the eval harness both use, so
what we evaluate is exactly what we serve.
"""
from __future__ import annotations

from bems_rag.generation.generator import Generator, get_generator
from bems_rag.retrieval.retriever import Retriever
from bems_rag.types import Answer, Chunk, Query


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        k: int = 4,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.generator = generator or get_generator()
        self.k = k

    def index(self, chunks: list[Chunk]) -> None:
        self.retriever.index(chunks)

    def answer(self, query: Query) -> Answer:
        contexts = self.retriever.retrieve(query, k=self.k)
        return self.generator.generate(query, contexts)
