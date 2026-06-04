"""
filters.py — Phase 2: Filtering & Constraint Engine

Applies clinical condition rules, allergen exclusions, and dietary preference
filters to the food DataFrame. Also builds the Bloom filter index for fast
allergen checking (BAX-423 technique #2).

Usage (standalone test):
    python filters.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from pybloom_live import BloomFilter

from db import load_all_foods

# ─── User Profile ──────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """
    Complete clinical + dietary profile for one user.
    Passed into FilterEngine to get a safe food subset.
    """
    # Demographics
    name:            str   = "User"
    age:             int   = 30
    sex:             str   = "female"       # "male" | "female"
    calorie_target:  float = 2000.0

    # Clinical conditions (can be multiple)
    has_ibs:          bool = False
    has_gerd:         bool = False
    has_diabetes:     bool = False          # Type 2
    has_hypertension: bool = False

    # Allergens (list of strings matching ALLERGEN_KEYWORDS keys in ingest.py)
    allergens: list[str] = field(default_factory=list)
    # e.g. ["gluten", "dairy", "tree_nuts", "soy"]

    # Diet type
    diet: str = "non_veg"
    # "vegan" | "vegetarian" | "pescatarian" | "non_veg"

    # Optional cultural/religious constraints
    no_pork:   bool = False
    no_beef:   bool = False
    no_alcohol: bool = True

    # Glycaemic index cap (set automatically for diabetes, but overridable)
    gi_cap: Optional[float] = None


# ─── Bloom Filter Builder ──────────────────────────────────────────────────────

ALLERGEN_COLUMNS = [
    "contains_gluten", "contains_dairy", "contains_tree_nuts",
    "contains_peanut", "contains_shellfish", "contains_soy",
    "contains_egg", "contains_fish", "contains_sesame",
]

def build_allergen_bloom_filters(df: pd.DataFrame) -> dict[str, BloomFilter]:
    """
    BAX-423 Technique #2: Bloom Filters
    Build one BloomFilter per allergen over the unsafe fdc_ids.
    Lookup is O(1) average-case — much faster than a DataFrame query at scale.

    Returns:
        Dict mapping allergen name → BloomFilter of unsafe fdc_ids.
    """
    filters: dict[str, BloomFilter] = {}
    for col in ALLERGEN_COLUMNS:
        allergen = col.replace("contains_", "")
        unsafe_ids = df.loc[df[col] == 1, "fdc_id"].tolist()
        bf = BloomFilter(capacity=max(len(unsafe_ids), 100), error_rate=0.001)
        for fdc_id in unsafe_ids:
            bf.add(int(fdc_id))
        filters[allergen] = bf
    return filters


# ─── Explain Tracker ──────────────────────────────────────────────────────────

class ExclusionLog:
    """
    Tracks why each food was excluded, powering the 'Explain' feature.
    Key: fdc_id → list of human-readable reason strings.
    """
    def __init__(self):
        self._log: dict[int, list[str]] = {}

    def add(self, fdc_id: int, reason: str):
        self._log.setdefault(int(fdc_id), []).append(reason)

    def get(self, fdc_id: int) -> list[str]:
        return self._log.get(int(fdc_id), [])

    def all(self) -> dict[int, list[str]]:
        return dict(self._log)

    def summary(self) -> str:
        total = len(self._log)
        return f"{total} food(s) excluded across all filters"


# ─── Filter Engine ────────────────────────────────────────────────────────────

class FilterEngine:
    """
    Applies all clinical, allergen, and dietary filters to the full food DataFrame.

    Example:
        engine = FilterEngine(df)
        safe_foods, log = engine.apply(profile)
    """

    # Pork/lard keywords for religious/preference exclusions
    PORK_KEYWORDS   = {"pork", "bacon", "ham", "lard", "prosciutto", "salami",
                       "pepperoni", "sausage", "pancetta", "chorizo"}
    BEEF_KEYWORDS   = {"beef", "steak", "veal", "brisket", "burger", "meatball"}
    ALCOHOL_KEYWORDS = {"wine", "beer", "ale", "lager", "spirits", "vodka",
                        "rum", "whiskey", "whisky", "bourbon", "liqueur",
                        "brandy", "sake", "mead"}
    SODIUM_HIGH_KEYWORDS = {"salt", "soy sauce", "miso", "pickle", "cured",
                             "smoked", "processed", "deli"}
    FRIED_KEYWORDS  = {"fried", "deep-fried", "deep fried", "battered", "tempura"}
    SPICY_KEYWORDS  = {"chili", "chile", "hot sauce", "jalapeño", "cayenne",
                       "sriracha", "curry powder", "wasabi"}

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._bloom = build_allergen_bloom_filters(df)

    def _name_contains(self, series: pd.Series, keywords: set[str]) -> pd.Series:
        """Return boolean mask: True if the food name contains any keyword."""
        pattern = "|".join(keywords)
        return series.str.contains(pattern, case=False, na=False, regex=True)

    def apply(self, profile: UserProfile) -> tuple[pd.DataFrame, ExclusionLog]:
        """
        Run all filters for the given profile.
        Returns (safe_df, exclusion_log).
        """
        df   = self.df.copy()
        log  = ExclusionLog()
        t0   = time.perf_counter()

        original_count = len(df)

        # ── 1. Diet filter ────────────────────────────────────────────────────
        # non_veg = no dietary restrictions; skip the whitelist filter entirely.
        # vegan / vegetarian / pescatarian use their is_* column as a whitelist.
        diet_col = f"is_{profile.diet}"
        if profile.diet != "non_veg" and diet_col in df.columns:
            mask_diet = df[diet_col] == 1
            for fid in df.loc[~mask_diet, "fdc_id"]:
                log.add(fid, f"Excluded: not suitable for {profile.diet} diet")
            df = df[mask_diet]

        # ── 2. Allergen filter (Bloom filter lookup) ──────────────────────────
        for allergen in profile.allergens:
            col = f"contains_{allergen}"
            if col not in df.columns:
                continue

            # Use Bloom filter for fast pre-screening, then confirm with DB column
            bf = self._bloom.get(allergen)
            if bf is None:
                continue

            def is_unsafe(fdc_id, allergen=allergen, bf=bf, df_col=df[col]):
                # Bloom filter: if NOT in filter → definitely safe (no false negatives)
                return bool(bf.__contains__(int(fdc_id)))

            bloom_flagged = df["fdc_id"].apply(is_unsafe)
            # Confirm with actual DB column (eliminates Bloom false positives)
            actual_unsafe = bloom_flagged & (df[col] == 1)

            for fid in df.loc[actual_unsafe, "fdc_id"]:
                log.add(fid, f"Allergen excluded: contains {allergen}")

            df = df[~actual_unsafe]

        # ── 2b. Name-based allergen safety net ───────────────────────────────
        # Foods with empty ingredients may have wrong allergen flags (contains_X=0
        # even when the name clearly signals an allergen, e.g. "Waffle, buttermilk"
        # containing wheat). Sweep names for user's allergens as a fallback.
        from ingest import ALLERGENS as _ALLERGENS, _word_match as _wm
        for allergen in profile.allergens:
            col = f"contains_{allergen}"
            if col not in df.columns:
                continue
            keywords = _ALLERGENS.get(allergen, set())
            if not keywords:
                continue
            # Only apply to rows where the DB flag is 0 (already-flagged rows excluded above)
            unchecked = df[df[col] == 0]
            name_unsafe_mask = unchecked["name"].str.lower().apply(
                lambda n: _wm(n, keywords)
            )
            name_unsafe_ids = unchecked.loc[name_unsafe_mask, "fdc_id"]
            for fid in name_unsafe_ids:
                log.add(fid, f"Allergen excluded (name match): contains {allergen}")
            df = df[~df["fdc_id"].isin(name_unsafe_ids)]

        # ── 3. IBS / Low-FODMAP filter ────────────────────────────────────────
        if profile.has_ibs:
            mask_fodmap = df["is_high_fodmap"] == 1
            for fid in df.loc[mask_fodmap, "fdc_id"]:
                log.add(fid, "Excluded: high-FODMAP food, unsafe for IBS")
            df = df[~mask_fodmap]

        # ── 4. GERD filter ────────────────────────────────────────────────────
        if profile.has_gerd:
            mask_gerd = df["is_gerd_trigger"] == 1
            for fid in df.loc[mask_gerd, "fdc_id"]:
                log.add(fid, "Excluded: GERD trigger food (citrus/tomato/spicy/fried/caffeine)")
            df = df[~mask_gerd]

            # Extra keyword sweep for fried and spicy (not always caught by DB flag)
            mask_fried = self._name_contains(df["name"], self.FRIED_KEYWORDS)
            for fid in df.loc[mask_fried, "fdc_id"]:
                log.add(fid, "Excluded: fried food, GERD trigger")
            df = df[~mask_fried]

            mask_spicy = self._name_contains(df["name"], self.SPICY_KEYWORDS)
            for fid in df.loc[mask_spicy, "fdc_id"]:
                log.add(fid, "Excluded: spicy food, GERD trigger")
            df = df[~mask_spicy]

        # ── 5. Diabetes / Glycaemic Index filter ──────────────────────────────
        if profile.has_diabetes:
            gi_cap = profile.gi_cap or 55
            # Only exclude foods where GI is known AND above cap
            mask_high_gi = df["glycaemic_index"].notna() & (df["glycaemic_index"] > gi_cap)
            for fid in df.loc[mask_high_gi, "fdc_id"]:
                gi_val = df.loc[df["fdc_id"] == fid, "glycaemic_index"].values[0]
                log.add(fid, f"Excluded: GI={gi_val:.0f} > {gi_cap} (diabetes threshold)")
            df = df[~mask_high_gi]

            # Flag high-sugar foods by name (honey, syrup, candy, etc.)
            SUGAR_KEYWORDS = {"sugar", "syrup", "honey", "candy", "dessert",
                              "soda", "juice", "sweetened", "white rice", "white bread"}
            mask_sugar = self._name_contains(df["name"], SUGAR_KEYWORDS)
            for fid in df.loc[mask_sugar, "fdc_id"]:
                log.add(fid, "Excluded: high added sugar, not suitable for Type 2 diabetes")
            df = df[~mask_sugar]

        # ── 6. Hypertension / DASH filter ─────────────────────────────────────
        if profile.has_hypertension:
            # Cap sodium at 1500 mg per 100g serving (very high-sodium processed foods)
            SODIUM_CAP = 1500
            mask_sodium = df["sodium_mg"].notna() & (df["sodium_mg"] > SODIUM_CAP)
            for fid in df.loc[mask_sodium, "fdc_id"]:
                na_val = df.loc[df["fdc_id"] == fid, "sodium_mg"].values[0]
                log.add(fid, f"Excluded: sodium={na_val:.0f}mg/100g > {SODIUM_CAP}mg cap (hypertension)")
            df = df[~mask_sodium]

            # Also exclude by name: pickled, cured, heavily processed
            mask_salty = self._name_contains(df["name"], self.SODIUM_HIGH_KEYWORDS)
            for fid in df.loc[mask_salty, "fdc_id"]:
                log.add(fid, "Excluded: high-sodium processed food (DASH diet)")
            df = df[~mask_salty]

        # ── 7. Cultural/religious constraints ─────────────────────────────────
        if profile.no_pork:
            mask_pork = self._name_contains(df["name"], self.PORK_KEYWORDS)
            for fid in df.loc[mask_pork, "fdc_id"]:
                log.add(fid, "Excluded: pork/pork-derived (user preference)")
            df = df[~mask_pork]

        if profile.no_beef:
            mask_beef = self._name_contains(df["name"], self.BEEF_KEYWORDS)
            for fid in df.loc[mask_beef, "fdc_id"]:
                log.add(fid, "Excluded: beef (user preference)")
            df = df[~mask_beef]

        if profile.no_alcohol:
            mask_alc = self._name_contains(df["name"], self.ALCOHOL_KEYWORDS)
            for fid in df.loc[mask_alc, "fdc_id"]:
                log.add(fid, "Excluded: contains alcohol")
            df = df[~mask_alc]

        # ── 8. Drop foods with no calorie data (can't plan meals without it) ──
        mask_no_cal = df["calories"].isna() | (df["calories"] <= 0)
        df = df[~mask_no_cal]

        elapsed = time.perf_counter() - t0
        excluded = original_count - len(df)
        print(
            f"[FilterEngine] {original_count:,} foods → {len(df):,} safe "
            f"({excluded:,} excluded) in {elapsed*1000:.1f}ms"
        )

        return df.reset_index(drop=True), log


# ─── Test Personas ─────────────────────────────────────────────────────────────

TEST_PERSONAS = {
    "priya": UserProfile(
        name="Priya", age=28, sex="female", calorie_target=1800,
        has_ibs=True,
        allergens=["dairy"],
        diet="vegetarian",
    ),
    "ravi": UserProfile(
        name="Ravi", age=35, sex="male", calorie_target=2200,
        has_gerd=True,
        allergens=["gluten"],
        diet="non_veg",
        no_pork=True,
    ),
    "mei": UserProfile(
        name="Mei", age=52, sex="female", calorie_target=1600,
        has_diabetes=True,
        allergens=["tree_nuts"],
        diet="vegan",
        gi_cap=55,
    ),
    "james": UserProfile(
        name="James", age=45, sex="male", calorie_target=2000,
        has_hypertension=True,
        allergens=["soy"],
        diet="pescatarian",
    ),
}


# ─── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading food database…")
    df = load_all_foods()
    print(f"Loaded {len(df):,} foods.\n")

    engine = FilterEngine(df)

    for persona_name, profile in TEST_PERSONAS.items():
        print(f"── Persona: {profile.name} ──────────────────────────")
        safe_df, log = engine.apply(profile)
        print(f"   Safe foods: {len(safe_df):,}")
        print(f"   {log.summary()}")

        # Show a few exclusion examples
        sample_ids = list(log.all().keys())[:3]
        for fid in sample_ids:
            reasons = log.get(fid)
            name_matches = df.loc[df["fdc_id"] == fid, "name"]
            fname = name_matches.values[0] if len(name_matches) else f"fdc_id={fid}"
            print(f"   ❌ {fname[:50]}: {reasons[0]}")
        print()
