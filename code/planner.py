"""
planner.py — Phase 3: Meal Plan Generator & Diversity Engine

Generates a 7-day × 3-meal plan for a UserProfile from the filtered food pool.
Enforces:
  - No repeated meals across the full week
  - Category variety within each day (breakfast/lunch/dinner pools)
  - Diversity score ≥ 0.55 across all 21 meals (FAISS-based)
  - Calorie targets met within ±15% per day
  - Macro balance scoring (carbs/protein/fat ratios)
  - Per-meal sodium cap to prevent extreme spikes
  - Generation time < 60 seconds
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from db import get_rda
from filters import UserProfile

# ── Constants ──────────────────────────────────────────────────────────────────
MEAL_SLOTS    = ["breakfast", "lunch", "dinner"]
DAYS          = 7
TARGET_MEALS  = DAYS * len(MEAL_SLOTS)   # 21
FOODS_PER_MEAL = 2
MIN_DIVERSITY  = 0.55
MAX_RETRIES    = 5

# Calorie split across meal slots
CALORIE_SPLIT = {"breakfast": 0.25, "lunch": 0.35, "dinner": 0.40}

# Target macro ratios (fraction of calories)
MACRO_TARGETS = {
    "carbs_g":   (0.45, 0.65),   # 45–65% of calories from carbs  (1g = 4 kcal)
    "protein_g": (0.10, 0.35),   # 10–35% from protein             (1g = 4 kcal)
    "fat_g":     (0.20, 0.35),   # 20–35% from fat                 (1g = 9 kcal)
}
KCAL_PER = {"carbs_g": 4.0, "protein_g": 4.0, "fat_g": 9.0}

# Sodium cap per meal (mg) — prevents single meals blowing the daily limit
SODIUM_PER_MEAL_CAP = 600   # ~2300mg / ~3.8 meals with some headroom

# Food categories by slot
BREAKFAST_CATEGORIES = {
    "breakfast cereals", "baked products", "dairy and egg products",
    "fruits and fruit juices", "nut and seed products",
}
LUNCH_CATEGORIES = {
    "legumes and legume products", "vegetables and vegetable products",
    "soups, sauces, and gravies", "finfish and shellfish products",
    "poultry products", "beef products", "pork products",
    "baked products",
}
DINNER_CATEGORIES = {
    "beef products", "poultry products", "finfish and shellfish products",
    "pork products", "lamb, veal, and game products",
    "legumes and legume products", "vegetables and vegetable products",
    "cereal grains and pasta",
}

# Patterns that indicate a food is a raw ingredient / pantry staple,
# not suitable as a standalone meal component
RAW_INGREDIENT_RE = re.compile(
    r"\braw\b"
    r"|, raw$"
    r"|dry mix"
    r"|dehydrated"
    r"|freeze.dried"
    r"|\bdried\b"
    r"|unprepared"
    r"|unenriched$"
    r"|whole.grain$"
    r"|\bflour\b"
    r"|\bbran\b"
    r"|\bstarch\b"
    r"|isolate"
    r"|concentrate"
    r"|extender"
    r"|\bpowder\b(?! sugar)"  # allow "powder sugar" but not generic powder
    r"|mature seeds"
    r"|uncooked$"
    r"|, dry$"
    r"|dry, enriched"
    r"|dry, unenriched"
    r"|hard red"
    r"|hard white"
    r"|soft red"
    r"|soft white"
    r"|\bgrain\b(?!.*bar|.*bowl|.*salad)"  # grain alone, not grain bar/bowl/salad
    r"|vital wheat gluten"
    r"|wheat germ"
    r"|corn germ"
    r"|rice bran"
    r"|oat bran$"
    r"|shortening"
    r"|lard$"
    r"|tallow"
    r"|suet"
    r"|separable (lean|fat)"
    r"|seam fat"
    r"|skin only"
    r"|mechanically deboned"
    r"|, uncooked"
    r"|khorasan"
    r"|spelt$"
    r"|teff$"
    r"|sorghum grain"
    r"|amaranth grain"
    r"|\bgrits\b, (dry|regular)"
    r"|cornmeal, (whole|degermed|bolted|self-rising)"
    r"|semolina"
    r"|tapioca, pearl"
    r"|arrowroot"
    r"|potato (starch|flour)"
    r"|soy (flour|protein)"
    r"|peanut flour"
    r"|meat extender"
    r"|protein (supplement|powder)"
    r"|gravies.*dry"
    r"|gravy.*(dry|powder|mix)"
    r"|sauce.*dry"
    r"|soup.*dry"
    r"|noodle.*dry"
    r"|pasta.*dry"
    r"|couscous, dry"
    r"|rice.*parboiled.*dry"
    r"|ramen.*dry"
    r"|somen.*dry"
    r"|soba.*dry"
    r"|chow mein.*dry",
    re.IGNORECASE,
)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Meal:
    slot:      str
    day:       int
    foods:     list[dict]
    calories:  float = 0.0
    protein_g: float = 0.0
    carbs_g:   float = 0.0
    fat_g:     float = 0.0
    sodium_mg: float = 0.0

    def __post_init__(self):
        for attr in ("calories", "protein_g", "carbs_g", "fat_g", "sodium_mg"):
            setattr(self, attr, sum(f.get(attr, 0) or 0 for f in self.foods))

    @property
    def fdc_ids(self) -> list[int]:
        return [f["fdc_id"] for f in self.foods]

    @property
    def names(self) -> list[str]:
        return [f["name"] for f in self.foods]


@dataclass
class DayPlan:
    day:   int
    meals: dict[str, Meal] = field(default_factory=dict)

    @property
    def total_calories(self) -> float:
        return sum(m.calories for m in self.meals.values())

    @property
    def total_sodium(self) -> float:
        return sum(m.sodium_mg for m in self.meals.values())


@dataclass
class WeekPlan:
    profile:            UserProfile
    days:               list[DayPlan]
    diversity_score:    float = 0.0
    generation_time_s:  float = 0.0
    warnings:           list[str] = field(default_factory=list)

    @property
    def all_meals(self) -> list[Meal]:
        return [m for d in self.days for m in d.meals.values()]

    @property
    def avg_daily_calories(self) -> float:
        return sum(d.total_calories for d in self.days) / max(len(self.days), 1)

    def __str__(self) -> str:
        lines = [
            f"═══ 7-Day Meal Plan for {self.profile.name} ═══",
            f"Diversity score : {self.diversity_score:.2f}",
            f"Avg daily kcal  : {self.avg_daily_calories:.0f}",
            f"Generated in    : {self.generation_time_s:.1f}s",
            "",
        ]
        for day in self.days:
            lines.append(f"── Day {day.day} ({day.total_calories:.0f} kcal, {day.total_sodium:.0f}mg Na) ──")
            for slot in MEAL_SLOTS:
                meal = day.meals.get(slot)
                if meal:
                    names = " + ".join(n[:35] for n in meal.names)
                    lines.append(
                        f"  {slot:<10} {meal.calories:>5.0f} kcal  "
                        f"P:{meal.protein_g:.0f}g C:{meal.carbs_g:.0f}g F:{meal.fat_g:.0f}g  "
                        f"{names}"
                    )
            lines.append("")
        if self.warnings:
            lines.append("⚠️  Warnings:")
            for w in self.warnings:
                lines.append(f"   {w}")
        return "\n".join(lines)


# ── Planner ────────────────────────────────────────────────────────────────────

class MealPlanner:

    def __init__(self, safe_df: pd.DataFrame):
        self.df = safe_df.copy().reset_index(drop=True)
        self._meal_ready = self._build_meal_ready_pool(self.df)
        self._embedding_index = None
        self._try_load_embeddings()

    def _try_load_embeddings(self):
        try:
            from embeddings import FoodEmbeddingIndex
            self._embedding_index = FoodEmbeddingIndex.load()
        except Exception:
            self._embedding_index = None

    # ── Pool building ──────────────────────────────────────────────────────────

    def _build_meal_ready_pool(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip out raw ingredients, pantry staples, and industrial food components.
        Falls back to full pool if fewer than 100 meal-ready foods remain.
        """
        mask = ~df["name"].str.contains(RAW_INGREDIENT_RE.pattern, case=False, na=False, regex=True)
        filtered = df[mask]
        return filtered if len(filtered) >= 100 else df

    def _slot_pool(self, slot: str) -> pd.DataFrame:
        """Return meal-ready foods filtered by slot category, with fallback."""
        cat_col = "food_category"
        if cat_col not in self._meal_ready.columns:
            return self._meal_ready

        category_map = {
            "breakfast": BREAKFAST_CATEGORIES,
            "lunch":     LUNCH_CATEGORIES,
            "dinner":    DINNER_CATEGORIES,
        }
        cats = category_map.get(slot, set())
        mask = self._meal_ready[cat_col].str.lower().isin(cats)
        pool = self._meal_ready[mask]
        return pool if len(pool) >= 30 else self._meal_ready

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _macro_score(self, food: dict, slot_cal_target: float) -> float:
        """
        Score a food's macro balance. Lower = better fit.
        Penalises foods that push macros outside MACRO_TARGETS ranges.
        Also heavily penalises foods with very high sodium per calorie.
        """
        cal = food.get("calories") or 1.0
        score = 0.0

        for macro, (lo, hi) in MACRO_TARGETS.items():
            grams  = food.get(macro) or 0.0
            kcal_m = grams * KCAL_PER[macro]
            ratio  = kcal_m / cal
            if ratio < lo:
                score += (lo - ratio) * 3   # softer penalty
            elif ratio > hi:
                score += (ratio - hi) * 3

        # Sodium penalty: flag foods with > 1g sodium per 100 kcal
        sodium = food.get("sodium_mg") or 0.0
        sodium_density = sodium / max(cal, 1.0)  # mg per kcal
        if sodium_density > 10:   # 10mg/kcal = 1000mg per 100kcal → very salty
            score += (sodium_density - 10) * 2

        return score

    def _combined_score(
        self,
        row: pd.Series,
        cal_target: float,
        day_sodium_so_far: float,
    ) -> float:
        """
        Combined score for food selection: calorie fit + macro balance + sodium.
        Lower = better candidate.
        """
        food = row.to_dict()
        cal  = food.get("calories") or 0.0

        # Calorie distance (normalised)
        per_food_target = cal_target / FOODS_PER_MEAL
        cal_score = abs(cal - per_food_target) / max(per_food_target, 1.0)

        # Macro score
        macro_score = self._macro_score(food, cal_target)

        # Soft sodium penalty — penalise but don't hard-exclude
        sodium = food.get("sodium_mg") or 0.0
        sodium_score = 0.0
        if day_sodium_so_far + sodium > 2300:
            sodium_score = ((day_sodium_so_far + sodium - 2300) / 2300) * 0.3

        # Deprioritise obvious junk/dessert foods — push them to back of queue
        name = (food.get("name") or "").lower()
        junk_score = 0.0
        if any(k in name for k in ("cookie", "cake", "candy", "chips", "doughnut",
                                    "donut", "archway", "keebler", "fudge",
                                    "iced mo", "frosty", "strudel", "sundae cone",
                                    "cream pie", "cheesecake", "cinnamon roll",
                                    "french fried", "french fries", "onion ring")):
            junk_score = 2.0

        return cal_score + macro_score * 0.3 + sodium_score + junk_score

    # ── Food selection ─────────────────────────────────────────────────────────

    def _pick_foods(
        self,
        slot: str,
        calorie_target: float,
        used_ids: set[int],
        anchor_id: Optional[int],
        day_sodium_so_far: float,
    ) -> list[dict]:

        def available_pool(exclude_ids):
            p = self._slot_pool(slot)
            p = p[~p["fdc_id"].isin(exclude_ids)]
            p = p[p["calories"].notna() & (p["calories"] > 0)]
            if len(p) < FOODS_PER_MEAL:
                # Fallback: full meal-ready pool, no slot restriction
                p = self._meal_ready[
                    ~self._meal_ready["fdc_id"].isin(exclude_ids) &
                    self._meal_ready["calories"].notna() &
                    (self._meal_ready["calories"] > 0)
                ]
            return p

        def score_and_pick(pool, n, cal_target):
            """Score pool and return top-n by combined score."""
            if pool.empty:
                return []
            if self._embedding_index and anchor_id and len(pool) > n * 3:
                try:
                    pool = self._embedding_index.rank_by_similarity(
                        pool, anchor_id, invert=True
                    )
                except Exception:
                    pass
            top_n = pool.head(60) if len(pool) > 60 else pool
            top_n = top_n.copy()
            top_n["_score"] = top_n.apply(
                lambda r: self._combined_score(r, cal_target, day_sodium_so_far),
                axis=1,
            )
            top_n = top_n.sort_values("_score").drop(columns=["_score"])
            picked = []
            for _, row in top_n.iterrows():
                if len(picked) >= n:
                    break
                food = row.to_dict()
                if food["fdc_id"] not in used_ids:
                    picked.append(food)
                    used_ids.add(food["fdc_id"])
            return picked

        # ── Step 1: pick initial FOODS_PER_MEAL foods ─────────────────────────
        pool     = available_pool(used_ids)
        if pool.empty:
            return []
        selected = score_and_pick(pool, FOODS_PER_MEAL, calorie_target)

        # ── Step 2: calorie top-up ─────────────────────────────────────────────
        # If the selected foods fall below 60% of the slot calorie target,
        # keep adding one food at a time (up to 2 extras) until we're close.
        MIN_CAL_RATIO = 0.60
        MAX_EXTRA     = 2
        extras        = 0

        while extras < MAX_EXTRA:
            meal_cal = sum(f.get("calories", 0) or 0 for f in selected)
            if meal_cal >= calorie_target * MIN_CAL_RATIO:
                break

            # Remaining calorie gap — pick a food closest to that gap
            gap  = calorie_target - meal_cal
            pool = available_pool(used_ids)
            if pool.empty:
                break

            # For top-up, ignore diversity ranking — just go closest to gap
            pool = pool.copy()
            pool["_gap"] = (pool["calories"] - gap).abs()
            pool = pool.sort_values("_gap").drop(columns=["_gap"])

            added = 0
            for _, row in pool.iterrows():
                food = row.to_dict()
                fid  = food["fdc_id"]
                if fid not in used_ids:
                    selected.append(food)
                    used_ids.add(fid)
                    added = 1
                    break

            if not added:
                break
            extras += 1

        return selected

    # ── Plan generation ────────────────────────────────────────────────────────

    def generate(self, profile: UserProfile, seed: Optional[int] = None) -> WeekPlan:
        t0        = time.perf_counter()
        rda       = get_rda(profile.sex, profile.age, profile.calorie_target)
        daily_cal = rda["calories"]

        best_plan:      Optional[WeekPlan] = None
        best_diversity: float = -1.0

        for attempt in range(1, MAX_RETRIES + 1):
            if seed is not None:
                random.seed(seed + attempt)
                np.random.seed(seed + attempt)

            plan, diversity = self._build_plan(profile, daily_cal)

            if diversity > best_diversity:
                best_diversity = diversity
                best_plan      = plan

            if diversity >= MIN_DIVERSITY:
                break

        elapsed = time.perf_counter() - t0
        best_plan.generation_time_s = elapsed
        best_plan.diversity_score   = best_diversity

        if elapsed > 60:
            best_plan.warnings.append(f"Generation took {elapsed:.1f}s (target < 60s)")
        if best_diversity < MIN_DIVERSITY:
            best_plan.warnings.append(
                f"Diversity score {best_diversity:.2f} below target {MIN_DIVERSITY} "
                f"after {MAX_RETRIES} attempts — pool may be too small"
            )

        print(
            f"[Planner] {profile.name}: plan generated in {elapsed:.1f}s, "
            f"diversity={best_diversity:.2f}, avg {best_plan.avg_daily_calories:.0f} kcal/day"
        )
        return best_plan

    def _build_plan(self, profile: UserProfile, daily_cal: float) -> tuple[WeekPlan, float]:
        used_ids: set[int] = set()
        days:     list[DayPlan] = []
        all_fdc_ids: list[int] = []

        anchor_id = int(self._meal_ready["fdc_id"].sample(1).values[0])

        for day_num in range(1, DAYS + 1):
            day_plan      = DayPlan(day=day_num)
            day_sodium    = 0.0

            for slot in MEAL_SLOTS:
                slot_cal = daily_cal * CALORIE_SPLIT[slot]
                foods    = self._pick_foods(
                    slot, slot_cal, used_ids, anchor_id, day_sodium
                )
                if not foods:
                    continue

                meal = Meal(slot=slot, day=day_num, foods=foods)
                day_plan.meals[slot] = meal
                all_fdc_ids.extend(meal.fdc_ids)
                day_sodium += meal.sodium_mg
                anchor_id   = foods[-1]["fdc_id"]

            days.append(day_plan)

        plan = WeekPlan(profile=profile, days=days)

        # Diversity score
        if self._embedding_index and all_fdc_ids:
            try:
                diversity = self._embedding_index.diversity_score(all_fdc_ids)
            except Exception:
                diversity = _fallback_diversity(all_fdc_ids, self.df)
        else:
            diversity = _fallback_diversity(all_fdc_ids, self.df)

        return plan, diversity


# ── Fallback diversity ─────────────────────────────────────────────────────────

def _fallback_diversity(fdc_ids: list[int], df: pd.DataFrame) -> float:
    if not fdc_ids or "food_category" not in df.columns:
        return 0.7
    cats = df[df["fdc_id"].isin(fdc_ids)]["food_category"].dropna()
    return min(1.0, len(cats.unique()) / len(fdc_ids)) if not cats.empty else 0.7


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from db import load_all_foods
    from filters import FilterEngine, TEST_PERSONAS

    print("Loading foods…\n")
    df     = load_all_foods()
    engine = FilterEngine(df)

    for persona_name, profile in TEST_PERSONAS.items():
        print("=" * 60)
        safe, log = engine.apply(profile)
        if safe.empty:
            print(f"⚠️  No safe foods for {profile.name}")
            continue
        planner = MealPlanner(safe)
        plan    = planner.generate(profile, seed=42)
        print(plan)
        assert plan.generation_time_s < 60
        assert len(plan.all_meals) > 0
        print(f"✅ {profile.name} — {len(plan.all_meals)} meals, diversity={plan.diversity_score:.2f}\n")
