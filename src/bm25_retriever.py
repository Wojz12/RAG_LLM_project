from pyserini.search import SimpleSearcher


class BM25Retriever:
    def __init__(self, index_path="bm25_index"):
        self.searcher = SimpleSearcher(index_path)
        self.searcher.set_bm25(k1=0.9, b=0.4)

    def search(self, query, k=20):
        hits = self.searcher.search(query, k)
        return [h.raw for h in hits]

