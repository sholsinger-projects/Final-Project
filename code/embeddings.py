"""
embeddings.py — BAX-423 Technique #1: FAISS Food Embeddings

Encodes food names/descriptions into dense vectors using sentence-transformers,
indexes them with FAISS for fast semantic similarity search. Used by the planner
to find nutritionally similar but culinarily diverse alternatives during meal
selection and ranking.

Usage:
    from embeddings import FoodEmbeddingIndex
    index = FoodEmbeddingIndex.build(df)          # first run: encodes + indexes
    index = FoodEmbeddingIndex.load()             # subsequent runs: loads from disk
    similar = index.find_similar(fdc_id, k=10)   # returns list of (fdc_id, score)
    ranked  = index.rank_by_similarity(candidates_df, anchor_fdc_id)
"""

from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

DATA_DIR   = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "faiss.index"
ID_MAP_PATH = DATA_DIR / "faiss_id_map.npy"

# All-MiniLM is fast, small (80MB), and excellent for food similarity
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM  = 384


def _get_model():
    """Lazy-load the sentence transformer (avoids slow import at module level)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def _food_text(row: pd.Series) -> str:
    """
    Build the text representation of a food for embedding.
    Combines name + category + first 60 chars of ingredients for richer signal.
    """
    parts = [str(row.get("name", "") or "")]
    cat = str(row.get("food_category", "") or "")
    if cat:
        parts.append(cat)
    ingr = str(row.get("ingredients", "") or "")
    if ingr:
        parts.append(ingr[:60])
    return " | ".join(p for p in parts if p)


class FoodEmbeddingIndex:
    """
    FAISS flat inner-product index over all safe foods in the DB.

    Build once per DB, then load from disk for subsequent runs.
    Inner product on L2-normalised vectors == cosine similarity.
    """

    def __init__(self, index: faiss.IndexFlatIP, id_map: np.ndarray):
        self._index  = index
        self._id_map = id_map          # position i → fdc_id
        self._pos_map = {int(fdc_id): i for i, fdc_id in enumerate(id_map)}

    # ── Build ──────────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, df: pd.DataFrame, save: bool = True) -> "FoodEmbeddingIndex":
        """
        Encode all foods in df and build a FAISS index.
        df must contain at least: fdc_id, name, food_category, ingredients.
        """
        t0 = time.perf_counter()
        print(f"[Embeddings] Encoding {len(df):,} foods with {MODEL_NAME}…")

        model  = _get_model()
        texts  = [_food_text(row) for _, row in df.iterrows()]
        fdc_ids = df["fdc_id"].values.astype(np.int64)

        # Encode in batches of 512 with progress bar
        vectors = model.encode(
            texts,
            batch_size=512,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-norm → cosine sim via inner product
        )

        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(vectors.astype(np.float32))

        elapsed = time.perf_counter() - t0
        print(f"[Embeddings] Built index: {index.ntotal:,} vectors in {elapsed:.1f}s")

        if save:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(INDEX_PATH))
            np.save(str(ID_MAP_PATH), fdc_ids)
            print(f"[Embeddings] Saved → {INDEX_PATH}")

        return cls(index, fdc_ids)

    # ── Load ───────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "FoodEmbeddingIndex":
        """Load a previously saved FAISS index from disk."""
        if not INDEX_PATH.exists() or not ID_MAP_PATH.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run FoodEmbeddingIndex.build(df) first."
            )
        index  = faiss.read_index(str(INDEX_PATH))
        id_map = np.load(str(ID_MAP_PATH))
        print(f"[Embeddings] Loaded index: {index.ntotal:,} vectors")
        return cls(index, id_map)

    @classmethod
    def load_or_build(cls, df: pd.DataFrame) -> "FoodEmbeddingIndex":
        """Load from disk if available, otherwise build and save."""
        try:
            return cls.load()
        except FileNotFoundError:
            return cls.build(df)

    # ── Query ──────────────────────────────────────────────────────────────────

    def find_similar(self, fdc_id: int, k: int = 10) -> list[tuple[int, float]]:
        """
        Return the k most semantically similar foods to fdc_id.
        Returns list of (fdc_id, cosine_similarity) sorted descending.
        Excludes the query food itself.
        """
        pos = self._pos_map.get(int(fdc_id))
        if pos is None:
            return []

        query_vec = np.zeros((1, EMBED_DIM), dtype=np.float32)
        self._index.reconstruct(pos, query_vec[0])

        scores, positions = self._index.search(query_vec, k + 1)
        results = []
        for score, p in zip(scores[0], positions[0]):
            if p == pos or p < 0:
                continue
            results.append((int(self._id_map[p]), float(score)))
        return results[:k]

    def encode_and_search(self, text: str, k: int = 10) -> list[tuple[int, float]]:
        """
        Encode an arbitrary text query and return the k nearest foods.
        Useful for free-text meal suggestions ("something with lentils").
        """
        model = _get_model()
        vec = model.encode([text], normalize_embeddings=True,
                           convert_to_numpy=True).astype(np.float32)
        scores, positions = self._index.search(vec, k)
        return [
            (int(self._id_map[p]), float(s))
            for s, p in zip(scores[0], positions[0]) if p >= 0
        ]

    def rank_by_similarity(
        self,
        candidates: pd.DataFrame,
        anchor_fdc_id: int,
        invert: bool = False,
    ) -> pd.DataFrame:
        """
        Re-rank a candidate DataFrame by similarity to anchor_fdc_id.

        invert=False → most similar first (for finding substitutes)
        invert=True  → least similar first (for maximising meal diversity)
        """
        if candidates.empty:
            return candidates

        similar = dict(self.find_similar(anchor_fdc_id, k=len(candidates) + 10))
        candidates = candidates.copy()
        candidates["_sim"] = candidates["fdc_id"].map(
            lambda fid: similar.get(int(fid), 0.0)
        )
        candidates = candidates.sort_values("_sim", ascending=invert)
        return candidates.drop(columns=["_sim"])

    def diversity_score(self, fdc_ids: list[int]) -> float:
        """
        Reconstructs vectors from the index and compute the mean pairwise 
        cosine dissimilarity across all 21 meals
        Returns a score in [0, 1] where 1 = maximally diverse.
        Used to evaluate and enforce diversity in generated meal plans.
        """
        positions = [self._pos_map.get(int(fid)) for fid in fdc_ids]
        positions = [p for p in positions if p is not None]
        if len(positions) < 2:
            return 1.0

        vecs = np.zeros((len(positions), EMBED_DIM), dtype=np.float32)
        for i, pos in enumerate(positions):
            self._index.reconstruct(pos, vecs[i])

        # Cosine similarity matrix (vectors already L2-normalised)
        sim_matrix = vecs @ vecs.T
        n = len(positions)
        # Mean of upper triangle (excluding diagonal)
        upper = sim_matrix[np.triu_indices(n, k=1)]
        mean_sim = float(upper.mean()) if len(upper) > 0 else 0.0
        return round(1.0 - mean_sim, 4)
    
if __name__ == "__main__":
    from db import load_all_foods
    print("Loading foods...")
    df = load_all_foods()
    print(f"Loaded {len(df):,} foods. Building FAISS index...")
    index = FoodEmbeddingIndex.build(df)
    print(f"Done. Index saved to {INDEX_PATH}")
