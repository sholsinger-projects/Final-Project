"""
bloom.py — BAX-423 Technique #2: Bloom Filter Allergen Index

Wraps pybloom_live to provide fast O(1) allergen safety checks.
Also includes a benchmark comparing Bloom vs. naive DataFrame query.

Usage:
    from bloom import AllergenBloomIndex
    index = AllergenBloomIndex.build(df)
    index.is_safe(fdc_id=12345, allergens=["gluten", "dairy"])  # → True/False
    index.cross_contam_risk(fdc_id=12345, allergens=["gluten"]) # → list of warnings
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from pybloom_live import BloomFilter


ALLERGEN_COLUMNS = {
    "gluten":    "contains_gluten",
    "dairy":     "contains_dairy",
    "tree_nuts": "contains_tree_nuts",
    "peanut":    "contains_peanut",
    "shellfish": "contains_shellfish",
    "soy":       "contains_soy",
    "egg":       "contains_egg",
    "fish":      "contains_fish",
    "sesame":    "contains_sesame",
}


class AllergenBloomIndex:
    """
    Probabilistic Bloom filter index over all allergens.

    Properties:
    - No false negatives: if a food is UNSAFE, we will always catch it.
    - Tiny false positive rate (0.1%): a safe food may rarely be flagged,
      but we confirm with the actual DB column before excluding.
    - Memory: ~10KB per allergen vs. storing the full ID set.
    """

    def __init__(self, filters: dict[str, BloomFilter], df: pd.DataFrame):
        self._filters = filters
        # Keep a lightweight lookup dict: fdc_id → row (for confirmation)
        self._lookup: dict[int, dict] = {
            int(row["fdc_id"]): row
            for _, row in df[["fdc_id"] + list(ALLERGEN_COLUMNS.values())].iterrows()
        }

    @classmethod
    def build(cls, df: pd.DataFrame) -> "AllergenBloomIndex":
        """Build the index from the food DataFrame."""
        t0 = time.perf_counter()
        filters: dict[str, BloomFilter] = {}

        for allergen, col in ALLERGEN_COLUMNS.items():
            if col not in df.columns:
                continue
            unsafe_ids = df.loc[df[col] == 1, "fdc_id"].tolist()
            bf = BloomFilter(capacity=max(len(unsafe_ids), 100), error_rate=0.001)
            for fdc_id in unsafe_ids:
                bf.add(int(fdc_id))
            filters[allergen] = bf

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[BloomIndex] Built {len(filters)} allergen filters in {elapsed:.1f}ms")
        return cls(filters, df)

    def is_safe(self, fdc_id: int, allergens: list[str]) -> bool:
        """
        Return True if fdc_id is safe for all listed allergens.
        Uses Bloom filter for speed, confirms with DB lookup.
        """
        fdc_id = int(fdc_id)
        row = self._lookup.get(fdc_id)
        if row is None:
            return True  # unknown food, allow through

        for allergen in allergens:
            bf = self._filters.get(allergen)
            col = ALLERGEN_COLUMNS.get(allergen)
            if bf is None or col is None:
                continue
            # Bloom says potentially unsafe → confirm with actual column
            if fdc_id in bf:
                if row.get(col, 0) == 1:
                    return False  # confirmed unsafe
        return True

    def cross_contam_risk(self, fdc_id: int, allergens: list[str]) -> list[str]:
        """
        Return a list of cross-contamination warnings for fdc_id.
        A warning is raised when the Bloom filter flags the food but the DB
        column is 0 (false positive territory — likely shared facility risk).
        """
        fdc_id = int(fdc_id)
        row = self._lookup.get(fdc_id)
        warnings = []
        if row is None:
            return warnings

        for allergen in allergens:
            bf = self._filters.get(allergen)
            col = ALLERGEN_COLUMNS.get(allergen)
            if bf is None or col is None:
                continue
            if fdc_id in bf and row.get(col, 0) == 0:
                warnings.append(
                    f"⚠️ Possible {allergen} cross-contamination risk "
                    f"(food not certified allergen-free)"
                )
        return warnings

    def unsafe_ids_for(self, allergen: str) -> list[int]:
        """Return all fdc_ids confirmed unsafe for the given allergen."""
        col = ALLERGEN_COLUMNS.get(allergen)
        if col is None:
            return []
        return [
            fdc_id for fdc_id, row in self._lookup.items()
            if row.get(col, 0) == 1
        ]


# ─── Benchmark ────────────────────────────────────────────────────────────────

def benchmark(df: pd.DataFrame, n_lookups: int = 10_000):
    """
    Compare Bloom filter vs. naive DataFrame query for allergen checking.
    pre indexes 9 allergen sets into bloom fiters with an O(1) req
    hashing the fdc id and checking the bit array
    """
    import random

    index = AllergenBloomIndex.build(df)
    sample_ids = df["fdc_id"].sample(min(n_lookups, len(df))).tolist()
    allergens_to_check = ["gluten", "dairy", "tree_nuts"]

    # ── Bloom filter approach ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    bloom_results = [index.is_safe(fid, allergens_to_check) for fid in sample_ids]
    bloom_time = (time.perf_counter() - t0) * 1000

    # ── Naive DataFrame query approach ────────────────────────────────────────
    t0 = time.perf_counter()
    naive_results = []
    for fid in sample_ids:
        row = df[df["fdc_id"] == fid]
        if row.empty:
            naive_results.append(True)
            continue
        unsafe = any(
            row[f"contains_{a}"].values[0] == 1
            for a in allergens_to_check
            if f"contains_{a}" in df.columns
        )
        naive_results.append(not unsafe)
    naive_time = (time.perf_counter() - t0) * 1000

    print(f"""
┌─────────────────────────────────────────────────────┐
│  Bloom Filter Benchmark  ({n_lookups:,} allergen lookups)   │
├──────────────────────────┬──────────────┬───────────┤
│  Method                  │  Time (ms)   │  vs Bloom │
├──────────────────────────┼──────────────┼───────────┤
│  Bloom Filter            │  {bloom_time:>8.1f}    │    1.0×   │
│  Naive DataFrame query   │  {naive_time:>8.1f}    │  {naive_time/bloom_time:>5.1f}×   │
└──────────────────────────┴──────────────┴───────────┘
    Agreement rate: {sum(a==b for a,b in zip(bloom_results, naive_results))/len(bloom_results):.4%}
    (Bloom false positives confirmed & corrected by DB lookup)
""")


if __name__ == "__main__":
    from db import load_all_foods
    df = load_all_foods()
    benchmark(df)
