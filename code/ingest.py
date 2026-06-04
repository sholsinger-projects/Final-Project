"""
ingest.py — NutriAI Phase 1: Data Pipeline

Pulls food records from USDA FoodData Central API, stores them in SQLite
with full nutrient data and accurate clinical/allergen/dietary tags.

Usage:
    python ingest.py --api-key YOUR_KEY            # fresh ingest
    python ingest.py --api-key YOUR_KEY --reset    # wipe DB and re-ingest
    python ingest.py --verify                      # check DB stats
    python ingest.py --retag                       # re-run tags without API

Get a free API key at: https://fdc.nal.usda.gov/api-key-signup.html
"""

import argparse
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).parent.parent / "data"
DB_PATH   = DATA_DIR / "nutriai.db"
SNAP_PATH = DATA_DIR / "usda_snapshot.csv"

# ── USDA API ───────────────────────────────────────────────────────────────────
USDA_BASE  = "https://api.nal.usda.gov/fdc/v1"
BATCH_SIZE = 20    # /foods endpoint max per request
PAGE_SIZE  = 200   # /foods/list endpoint max per page
DATA_TYPES = ["SR Legacy", "Survey (FNDDS)", "Foundation", "Branded"]

# ── Nutrient ID → column ───────────────────────────────────────────────────────
NUTRIENT_MAP = {
    1008: "calories",
    1003: "protein_g",
    1005: "carbs_g",
    1004: "fat_g",
    1079: "fibre_g",
    1087: "calcium_mg",
    1089: "iron_mg",
    1092: "potassium_mg",
    1090: "magnesium_mg",
    1095: "zinc_mg",
    1114: "vitamin_d_iu",
    1178: "vitamin_b12_ug",
    1093: "sodium_mg",
    1316: "omega3_g",
}

# ── Allergen keyword sets ──────────────────────────────────────────────────────
# "butter" intentionally excluded — handled separately to avoid false-positives
# on "peanut butter", "almond butter", etc.
ALLERGENS = {
    "gluten":    {"wheat", "barley", "rye", "spelt", "kamut", "triticale", "malt",
                  "wheat flour", "enriched flour", "wheat starch"},
    "dairy":     {"milk", "cheese", "cream", "yogurt", "whey", "casein", "lactose",
                  "buttermilk", "ghee", "kefir", "paneer", "ricotta", "mozzarella",
                  "cheddar", "parmesan", "brie", "gouda", "nonfat milk", "skim milk",
                  "milkfat", "milk chocolate", "dairy", "milk solids"},
    "tree_nuts": {"almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut",
                  "macadamia", "brazil nut", "chestnut"},
    "peanut":    {"peanut", "groundnut"},
    "shellfish": {"shrimp", "crab", "lobster", "clam", "oyster", "scallop", "mussel"},
    "soy":       {"soy", "soybean", "soy bean", "soymilk", "tofu", "edamame",
                  "miso", "tempeh", "soy sauce", "tamari", "soy lecithin",
                  "soy protein", "textured soy"},
    "egg":       {"egg", "egg white", "egg yolk", "albumin", "mayonnaise"},
    "fish":      {"salmon", "tuna", "cod", "tilapia", "sardine", "anchovy", "bass",
                  "flounder", "halibut", "mackerel", "trout", "fish"},
    "sesame":    {"sesame", "tahini"},
}

FODMAP_TRIGGERS = {
    "garlic", "onion", "shallot", "leek", "wheat", "rye", "barley",
    "apple", "pear", "mango", "watermelon", "peach", "cherry",
    "honey", "fructose", "sorbitol", "mannitol", "xylitol",
    "cashew", "pistachio", "lentil", "chickpea", "kidney bean",
    "milk", "yogurt", "cream", "ice cream", "custard",
}

GERD_TRIGGERS = {
    "citrus", "lemon", "lime", "orange", "grapefruit", "tomato",
    "chocolate", "coffee", "espresso", "caffeine",
    "mint", "peppermint", "spearmint",
    "fried", "chili", "chile", "hot sauce", "vinegar",
    "onion", "garlic",
}

MEAT_KEYWORDS = {
    "beef", "pork", "chicken", "turkey", "lamb", "veal", "duck",
    "venison", "bison", "bacon", "ham", "sausage", "salami", "pepperoni",
    "prosciutto", "pancetta", "chorizo", "steak", "brisket",
}

FISH_KEYWORDS = ALLERGENS["fish"] | ALLERGENS["shellfish"]

NUT_BUTTER_RE = re.compile(
    r"(?:peanut|almond|cashew|sunflower|seed|nut|tahini|soy|coconut)\s+butter",
    re.IGNORECASE,
)


# ── Tagging ────────────────────────────────────────────────────────────────────

def _word_match(text: str, keywords: set) -> bool:
    """
    True if any keyword appears as a whole word (or phrase) in text.
    Allows a trailing 's' for plural forms (peanuts → peanut matches).
    Prevents false matches like "egg" in "vegetable", "soy" in "soybean".
    """
    for kw in keywords:
        # Allow plain plural (keyword + s) but not other suffixes (keywords + ed, ing…)
        pattern = r"(?<![a-z])" + re.escape(kw) + r"(?!(?:[a-rt-z]|s[a-z]))"
        if re.search(pattern, text):
            return True
    return False


def tag_food(name: str, ingredients: str) -> dict:
    """
    Return a dict of boolean clinical/allergen/dietary flags for a food.

    Uses ingredients string as primary source (authoritative for branded/
    processed foods); falls back to name when ingredients is empty.
    """
    ingr     = (ingredients or "").strip().lower()
    name_low = (name or "").strip().lower()
    primary  = ingr if ingr else name_low

    def hit(keywords: set) -> bool:
        return _word_match(primary, keywords) or _word_match(name_low, keywords)

    flags = {
        "is_high_fodmap":  hit(FODMAP_TRIGGERS),
        "is_gerd_trigger": hit(GERD_TRIGGERS),
    }

    for allergen, keywords in ALLERGENS.items():
        flags[f"contains_{allergen}"] = hit(keywords)

    # Bare "butter" → dairy, unless it's a nut/seed butter
    combined = name_low + " " + primary
    if re.search(r"(?<![a-z])butter(?![a-rt-z]|s[a-z])", combined):
        if not NUT_BUTTER_RE.search(combined):
            flags["contains_dairy"] = True

    has_meat  = hit(MEAT_KEYWORDS)
    has_fish  = hit(FISH_KEYWORDS)
    has_dairy = flags["contains_dairy"]
    has_egg   = flags["contains_egg"]
    has_animal = has_meat or has_fish or has_dairy or has_egg

    flags["is_vegan"]       = not has_animal
    flags["is_vegetarian"]  = not has_meat and not has_fish
    flags["is_pescatarian"] = not has_meat
    flags["is_non_veg"]     = has_meat or has_fish

    return flags


def extract_nutrients(food_nutrients: list) -> dict:
    result = {col: None for col in NUTRIENT_MAP.values()}
    for n in (food_nutrients or []):
        nid = n.get("nutrient", {}).get("id") or n.get("nutrientId")
        if nid in NUTRIENT_MAP:
            result[NUTRIENT_MAP[nid]] = n.get("amount")
    return result


# ── USDA API calls ─────────────────────────────────────────────────────────────

def list_page(api_key: str, page: int, data_type: str) -> list:
    """Fetch one page of food summaries (IDs + names, no nutrients)."""
    r = requests.get(
        f"{USDA_BASE}/foods/list",
        params={"api_key": api_key, "dataType": data_type,
                "pageSize": PAGE_SIZE, "pageNumber": page},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def fetch_details(api_key: str, fdc_ids: list) -> list:
    """
    Fetch full records (nutrients + ingredients) for up to 20 foods.
    format=full is required — 'abridged' omits the ingredients field,
    which breaks allergen/dietary tagging for branded foods.
    """
    r = requests.post(
        f"{USDA_BASE}/foods",
        params={"api_key": api_key},
        json={"fdcIds": fdc_ids, "format": "full"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── Database ───────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS foods (
    fdc_id           INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    data_type        TEXT,
    food_category    TEXT,
    ingredients      TEXT,

    calories         REAL,
    protein_g        REAL,
    carbs_g          REAL,
    fat_g            REAL,
    fibre_g          REAL,
    calcium_mg       REAL,
    iron_mg          REAL,
    potassium_mg     REAL,
    magnesium_mg     REAL,
    zinc_mg          REAL,
    vitamin_d_iu     REAL,
    vitamin_b12_ug   REAL,
    sodium_mg        REAL,
    omega3_g         REAL,

    is_high_fodmap      INTEGER DEFAULT 0,
    is_gerd_trigger     INTEGER DEFAULT 0,
    contains_gluten     INTEGER DEFAULT 0,
    contains_dairy      INTEGER DEFAULT 0,
    contains_tree_nuts  INTEGER DEFAULT 0,
    contains_peanut     INTEGER DEFAULT 0,
    contains_shellfish  INTEGER DEFAULT 0,
    contains_soy        INTEGER DEFAULT 0,
    contains_egg        INTEGER DEFAULT 0,
    contains_fish       INTEGER DEFAULT 0,
    contains_sesame     INTEGER DEFAULT 0,
    is_vegan            INTEGER DEFAULT 0,
    is_vegetarian       INTEGER DEFAULT 0,
    is_pescatarian      INTEGER DEFAULT 0,
    is_non_veg          INTEGER DEFAULT 0,
    glycaemic_index     REAL,

    ingested_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_category ON foods(food_category);
CREATE INDEX IF NOT EXISTS idx_vegan    ON foods(is_vegan);
CREATE INDEX IF NOT EXISTS idx_fodmap   ON foods(is_high_fodmap);
CREATE INDEX IF NOT EXISTS idx_calories ON foods(calories);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO foods (
    fdc_id, name, data_type, food_category, ingredients,
    calories, protein_g, carbs_g, fat_g, fibre_g,
    calcium_mg, iron_mg, potassium_mg, magnesium_mg, zinc_mg,
    vitamin_d_iu, vitamin_b12_ug, sodium_mg, omega3_g,
    is_high_fodmap, is_gerd_trigger,
    contains_gluten, contains_dairy, contains_tree_nuts, contains_peanut,
    contains_shellfish, contains_soy, contains_egg, contains_fish, contains_sesame,
    is_vegan, is_vegetarian, is_pescatarian, is_non_veg, glycaemic_index
) VALUES (
    :fdc_id, :name, :data_type, :food_category, :ingredients,
    :calories, :protein_g, :carbs_g, :fat_g, :fibre_g,
    :calcium_mg, :iron_mg, :potassium_mg, :magnesium_mg, :zinc_mg,
    :vitamin_d_iu, :vitamin_b12_ug, :sodium_mg, :omega3_g,
    :is_high_fodmap, :is_gerd_trigger,
    :contains_gluten, :contains_dairy, :contains_tree_nuts, :contains_peanut,
    :contains_shellfish, :contains_soy, :contains_egg, :contains_fish, :contains_sesame,
    :is_vegan, :is_vegetarian, :is_pescatarian, :is_non_veg, NULL
)
"""


def init_db(reset: bool = False) -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if reset:
        for path in [DB_PATH, SNAP_PATH]:
            if path.exists():
                path.unlink()
                print(f"🗑️  Deleted {path.name}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_food(conn: sqlite3.Connection, food: dict, data_type: str):
    fdc_id      = food.get("fdcId")
    name        = (food.get("description") or "").strip()
    ingredients = (food.get("ingredients") or "").strip()
    # foodCategory can be a dict {"id":..,"description":..} or a plain string
    fc = food.get("foodCategory") or food.get("foodCategoryLabel") or ""
    if isinstance(fc, dict):
        fc = fc.get("description", "")
    category = (fc or "").strip()
    dt          = food.get("dataType", data_type)

    nutrients = extract_nutrients(food.get("foodNutrients") or [])
    flags     = tag_food(name, ingredients)

    conn.execute(INSERT_SQL, {
        "fdc_id": fdc_id, "name": name, "data_type": dt,
        "food_category": category, "ingredients": ingredients,
        **nutrients,
        **{k: int(v) for k, v in flags.items()},
    })


# ── Main ingest ────────────────────────────────────────────────────────────────

def ingest(api_key: str, target: int = 10000, reset: bool = False):
    conn     = init_db(reset=reset)
    existing = conn.execute("SELECT COUNT(*) FROM foods").fetchone()[0]

    if existing >= target and not reset:
        print(f"✅ DB already has {existing:,} records. Use --reset to re-ingest.")
        conn.close()
        return

    needed = target - existing
    print(f"📥 Fetching {needed:,} foods (have {existing:,} / target {target:,})\n")
    pbar     = tqdm(total=needed, unit="foods")
    inserted = 0

    for data_type in DATA_TYPES:
        if inserted >= needed:
            break
        print(f"\n── {data_type} ──")
        page = 1

        while inserted < needed:
            # Step 1: lightweight page of IDs
            try:
                summaries = list_page(api_key, page, data_type)
            except requests.HTTPError as e:
                print(f"\n⚠️  List error (page {page}): {e}")
                break
            if not summaries:
                break

            fdc_ids = [s["fdcId"] for s in summaries if "fdcId" in s]

            # Step 2: full detail fetch in chunks of 20 — ALL ids on the page
            for i in range(0, len(fdc_ids), BATCH_SIZE):
                chunk = fdc_ids[i:i + BATCH_SIZE]
                try:
                    details = fetch_details(api_key, chunk)
                except requests.HTTPError as e:
                    print(f"\n⚠️  Detail error (chunk {i}): {e}")
                    time.sleep(2)
                    continue

                before = conn.execute("SELECT COUNT(*) FROM foods").fetchone()[0]
                for food in details:
                    if food.get("fdcId"):
                        insert_food(conn, food, data_type)
                conn.commit()

                new = conn.execute("SELECT COUNT(*) FROM foods").fetchone()[0] - before
                inserted += new
                pbar.update(new)
                time.sleep(0.1)

            page += 1
            time.sleep(0.1)

    pbar.close()
    total = conn.execute("SELECT COUNT(*) FROM foods").fetchone()[0]
    print(f"\n✅ Done. {total:,} foods in DB.")

    print(f"💾 Saving snapshot → {SNAP_PATH}")
    pd.read_sql("SELECT * FROM foods LIMIT 5000", conn).to_csv(SNAP_PATH, index=False)

    conn.close()
    verify_db()


# ── Retag ──────────────────────────────────────────────────────────────────────

def retag():
    if not DB_PATH.exists():
        print("❌ DB not found. Run ingest first.")
        return

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT fdc_id, name, ingredients FROM foods").fetchall()
    print(f"Re-tagging {len(rows):,} rows…")

    batch = []
    for fdc_id, name, ingredients in tqdm(rows):
        f = tag_food(name or "", ingredients or "")
        batch.append((
            int(f["is_high_fodmap"]),     int(f["is_gerd_trigger"]),
            int(f["contains_gluten"]),    int(f["contains_dairy"]),
            int(f["contains_tree_nuts"]), int(f["contains_peanut"]),
            int(f["contains_shellfish"]), int(f["contains_soy"]),
            int(f["contains_egg"]),       int(f["contains_fish"]),
            int(f["contains_sesame"]),
            int(f["is_vegan"]),           int(f["is_vegetarian"]),
            int(f["is_pescatarian"]),     int(f["is_non_veg"]),
            fdc_id,
        ))
        if len(batch) >= 500:
            _flush(conn, batch); batch = []
    if batch:
        _flush(conn, batch)

    conn.close()
    print("✅ Retag done.")
    verify_db()


def _flush(conn, batch):
    conn.executemany("""
        UPDATE foods SET
            is_high_fodmap=?,     is_gerd_trigger=?,
            contains_gluten=?,    contains_dairy=?,    contains_tree_nuts=?,
            contains_peanut=?,    contains_shellfish=?, contains_soy=?,
            contains_egg=?,       contains_fish=?,     contains_sesame=?,
            is_vegan=?,           is_vegetarian=?,     is_pescatarian=?,
            is_non_veg=?
        WHERE fdc_id=?
    """, batch)
    conn.commit()


# ── Verify ─────────────────────────────────────────────────────────────────────

def verify_db():
    if not DB_PATH.exists():
        print("❌ DB not found.")
        return
    conn = sqlite3.connect(DB_PATH)
    q = lambda s: conn.execute(s).fetchone()[0]
    print(f"""
────────────────────────────────────────
  NutriAI Database
────────────────────────────────────────
  Total records    : {q("SELECT COUNT(*) FROM foods"):>8,}
  Has calories     : {q("SELECT COUNT(*) FROM foods WHERE calories IS NOT NULL"):>8,}
  Has ingredients  : {q("SELECT COUNT(*) FROM foods WHERE ingredients IS NOT NULL AND ingredients != ''"):>8,}
  Vegan-safe       : {q("SELECT COUNT(*) FROM foods WHERE is_vegan=1"):>8,}
  High-FODMAP      : {q("SELECT COUNT(*) FROM foods WHERE is_high_fodmap=1"):>8,}
  GERD triggers    : {q("SELECT COUNT(*) FROM foods WHERE is_gerd_trigger=1"):>8,}
  Contains gluten  : {q("SELECT COUNT(*) FROM foods WHERE contains_gluten=1"):>8,}
  Contains dairy   : {q("SELECT COUNT(*) FROM foods WHERE contains_dairy=1"):>8,}
────────────────────────────────────────
""")
    df = pd.read_sql(
        "SELECT fdc_id, name, calories, protein_g, is_vegan, is_high_fodmap "
        "FROM foods ORDER BY RANDOM() LIMIT 10", conn
    )
    print(df.to_string(index=False))
    conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NutriAI Data Ingestion")
    parser.add_argument("--api-key", default=None, help="USDA FoodData Central API key")
    parser.add_argument("--target",  type=int, default=10000, help="Min records to ingest")
    parser.add_argument("--reset",   action="store_true", help="Wipe DB and snapshot, re-ingest from scratch")
    parser.add_argument("--verify",  action="store_true", help="Print DB stats and exit")
    parser.add_argument("--retag",   action="store_true", help="Re-run tagging on existing rows (no API needed)")
    args = parser.parse_args()

    if args.verify:
        verify_db()
    elif args.retag:
        retag()
    else:
        if not args.api_key:
            parser.error("--api-key is required")
        ingest(api_key=args.api_key, target=args.target, reset=args.reset)
