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
import re
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
                       "pepperoni", "sausage", "pancetta", "chorizo", "meat", "burrito"}
    BEEF_KEYWORDS   = {"beef", "steak", "veal", "brisket", "burger", "meatball", "meat", "burrito"}
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
            # Drop structural high-FODMAP database designations
            mask_fodmap = df["is_high_fodmap"] == 1
            df = df[~mask_fodmap]
            
            # Runtime name sweep for high-FODMAP food categories and aromatics
            IBS_CATCH = {"garlic", "onion", "shallot", "leek", "buttermilk", "milk", "honey", "bean", "lentil", "apple", "pear"}
            pattern = "|".join(r"\b" + re.escape(k) + r"s?\b" for k in IBS_CATCH)
            df = df[~df["name"].str.contains(pattern, case=False, na=False, regex=True)]

        # ── 4. GERD filter ────────────────────────────────────────────────────
        # ✅ PLACE THIS NEW GERD BLOCK:
        if profile.has_gerd:
            mask_gerd = df["is_gerd_trigger"] == 1
            for fid in df.loc[mask_gerd, "fdc_id"]:
                log.add(fid, "Excluded: Base database GERD trigger food classification")
            df = df[~mask_gerd]
            
            # Broaden catch for carbonation, citrus, high pastry fats, and chocolate/mint escapes
            GERD_CATCH = {
                "coffee", "tea", "cola", "soda", "citrus", "lemon", "lime", "orange", 
                "tomato", "mint", "chocolate", "cheese snack", "cream puff", "cobbler", 
                "pastry", "croissant", "fried", "spicy", "chili", "hot sauce"
            }
            pattern = "|".join(r"\b" + re.escape(k) + r"s?\b" for k in GERD_CATCH)
            mask_gerd_words = df["name"].str.contains(pattern, case=False, na=False, regex=True)
            
            for fid in df.loc[mask_gerd_words, "fdc_id"]:
                log.add(fid, "Excluded: High-fat pastry, acid, or gastric-irritant trigger (GERD protection)")
            df = df[~mask_gerd_words]

        # ── 5. Diabetes filter ────────────────────────────────────────────────
        if profile.has_diabetes:
            # Pass A: Check the database Glycaemic Index numeric column
            cap = profile.gi_cap if profile.gi_cap is not None else 55.0
            mask_gi = df["glycaemic_index"].notna() & (df["glycaemic_index"] > cap)
            for fid in df.loc[mask_gi, "fdc_id"]:
                gi_val = df.loc[df["fdc_id"] == fid, "glycaemic_index"].values[0]
                log.add(fid, f"Excluded: glycaemic_index={gi_val:.1f} > {cap} clinical cap (Type 2 Diabetes)")
            df = df[~mask_gi]

            # Pass B: Runtime High-Sugar/Bakery indicator net (catches text escapes instantly)
            HIGH_SUGAR_SWEETS = {
                "churro", "fritter", "caramel", "chocolate", "fudge", "baklava", 
                "croissant", "tart", "cobbler", "crisp", "puff", "beignet", "basbousa",
                "cookie", "cake", "candy", "pie", "doughnut", "donut", "pastry", "syrup", "sweetened"
            }
            # Modifies the expression to catch both singular and plural forms (e.g., churro and churros)
            pattern = "|".join(r"\b" + re.escape(k) + r"s?\b" for k in HIGH_SUGAR_SWEETS)
            mask_sweets = df["name"].str.contains(pattern, case=False, na=False, regex=True)
            
            for fid in df.loc[mask_sweets, "fdc_id"]:
                log.add(fid, "Excluded: High glycaemic sugar/bakery item, unsafe for Type 2 Diabetes")
            df = df[~mask_sweets]

       # ── 6. Hypertension / DASH filter ─────────────────────────────────────
        if profile.has_hypertension:
            # Drop any item that has a sodium density greater than 1.5 mg per calorie
            # This shields hypertensive users from low-calorie sodium traps like buttermilk and light dairy
            mask_density = (
                df["sodium_mg"].notna() & 
                df["calories"].notna() & 
                (df["calories"] > 0) & 
                (df["sodium_mg"] / df["calories"] > 1.5)
            )
            for fid in df.loc[mask_density, "fdc_id"]:
                log.add(fid, "Excluded: Sodium density exceeds strict 1.5 mg/kcal DASH target limit")
            df = df[~mask_density]

            # Keep the high-sodium processed keyword fallback active
            mask_salty = self._name_contains(df["name"], self.SODIUM_HIGH_KEYWORDS)
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
