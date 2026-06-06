"""
db.py — Shared database connection and query helpers.
All other modules import from here so DB_PATH is defined in one place.
"""

import sqlite3
from pathlib import Path
from functools import lru_cache

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH  = DATA_DIR / "nutriai.db"


def get_conn() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@lru_cache(maxsize=1)
def load_all_foods() -> pd.DataFrame:
    """
    Load the full food table into a pandas DataFrame (cached in memory).
    Called once at app startup; subsequent calls return the cached copy.
    """
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM foods", conn)
    conn.close()
    return df


def db_summary() -> dict:
    """Return basic stats about the food database."""
    conn = get_conn()
    stats = {}
    stats["total"]    = conn.execute("SELECT COUNT(*) FROM foods").fetchone()[0]
    stats["vegan"]    = conn.execute("SELECT COUNT(*) FROM foods WHERE is_vegan=1").fetchone()[0]
    stats["fodmap"]   = conn.execute("SELECT COUNT(*) FROM foods WHERE is_high_fodmap=1").fetchone()[0]
    stats["gerd"]     = conn.execute("SELECT COUNT(*) FROM foods WHERE is_gerd_trigger=1").fetchone()[0]
    stats["gluten"]   = conn.execute("SELECT COUNT(*) FROM foods WHERE contains_gluten=1").fetchone()[0]
    stats["has_calories"] = conn.execute(
        "SELECT COUNT(*) FROM foods WHERE calories IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return stats


# ─── RDA Reference Tables ─────────────────────────────────────────────────────
# Source: NIH Dietary Reference Intakes https://www.ncbi.nlm.nih.gov/books/NBK56068
# Values are daily targets; thresholded at 80% for flagging.

RDA_ADULT_MALE = {
    "calories":      2500,   # kcal (approximate; adjusted by user profile)
    "protein_g":       56,   # g
    "carbs_g":        130,   # g (EAR; actual target is higher, ~300g for 2500 kcal)
    "fat_g":           78,   # g (30% of 2500 kcal)
    "fibre_g":         38,   # g
    "calcium_mg":    1000,   # mg
    "iron_mg":          8,   # mg
    "potassium_mg":  3400,   # mg
    "magnesium_mg":   420,   # mg
    "zinc_mg":         11,   # mg
    "vitamin_d_iu":   600,   # IU
    "vitamin_b12_ug":   2.4, # µg
    "sodium_mg":     2300,   # mg (upper limit, not target)
}

RDA_ADULT_FEMALE = {
    "calories":      2000,
    "protein_g":       46,
    "carbs_g":        130,
    "fat_g":           65,
    "fibre_g":         25,
    "calcium_mg":    1000,
    "iron_mg":         18,
    "potassium_mg":  2600,
    "magnesium_mg":   320,
    "zinc_mg":          8,
    "vitamin_d_iu":   600,
    "vitamin_b12_ug":   2.4,
    "sodium_mg":     2300,
}


def get_rda(
    sex: str, 
    age: int, 
    calorie_target: float = None, 
    has_hypertension: bool = False, 
    has_diabetes: bool = False,
    has_gerd: bool = False,
    has_ibs: bool = False
) -> dict:
    """
    Return an RDA dict for the given sex/age, optionally overriding calorie target.
    Adjusts values dynamically to align with NHLBI DASH diet guidelines, diabetic 
    carb limits, GERD fat caps, and IBS fiber targets.
    """
    base = RDA_ADULT_MALE.copy() if sex.lower() == "male" else RDA_ADULT_FEMALE.copy()

    # Scale macros to user's calorie target if provided
    if calorie_target:
        base["calories"] = calorie_target
        
        # ─── Macro Allocation Priority Tree ───
        # 1. Diabetes: Drop carbs to a controlled 35% tier, shift to high protein
        if has_diabetes:
            carb_pct = 0.35
            protein_pct = 0.35
            fat_pct = 0.30
        # 2. GERD: Reduce fat down to a strict 22% limit to prevent gastric reflux loading
        elif has_gerd:
            carb_pct = 0.53
            protein_pct = 0.25
            fat_pct = 0.22
        # 3. Standard baseline macro configuration
        else:
            carb_pct = 0.50
            protein_pct = 0.20
            fat_pct = 0.30

        base["protein_g"] = round(calorie_target * protein_pct / 4, 1)
        base["carbs_g"]   = round(calorie_target * carb_pct / 4, 1)
        base["fat_g"]     = round(calorie_target * fat_pct / 9, 1)

    # Age adjustments
    if age >= 71:
        base["calcium_mg"]   = 1200
        base["vitamin_d_iu"] = 800
    elif age >= 51:
        base["calcium_mg"]   = 1200 if sex.lower() == "female" else 1000
        base["vitamin_d_iu"] = 600

    # ─── Clinical Condition Fine-Tuning ───
    
    # IBS Profile Adjustments (Fiber Control)
    if has_ibs:
        # Stabilize fiber to a moderate therapeutic ceiling (~20g) to prevent bowel fermentation spikes
        base["fibre_g"] = 20.0

    # DASH Diet Adjustments (Hypertension Control)
    if has_hypertension:
        base["sodium_mg"]    = 1500.0  # Strict DASH target cap
        base["potassium_mg"] = 4700.0  # Elevated target to offset sodium
        base["calcium_mg"]   = max(base["calcium_mg"], 1200.0)
        base["magnesium_mg"] = max(base["magnesium_mg"], 500.0)

    return base
