"""
nutrients.py — Phase 4: Nutrient Analysis & RDA Flagging

Computes per-meal and daily macro/micronutrient totals from a WeekPlan,
compares against NIH RDA targets (age/sex adjusted), and flags any day
that falls below 80% of target for any tracked nutrient.

Also powers the "Explain" feature: given a food that was excluded,
returns a human-readable reason from the ExclusionLog.

Usage:
    from nutrients import NutrientAnalyser
    analyser = NutrientAnalyser(plan, profile)
    report   = analyser.daily_report()       # list of DayNutrients
    flags    = analyser.rda_flags()          # list of warning strings
    summary  = analyser.weekly_summary()     # dict of weekly averages
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from db import get_rda
from filters import ExclusionLog, UserProfile
from planner import WeekPlan, Meal

# Nutrients tracked for RDA comparison
TRACKED = [
    "calories", "protein_g", "carbs_g", "fat_g", "fibre_g",
    "calcium_mg", "iron_mg", "potassium_mg", "magnesium_mg",
    "zinc_mg", "vitamin_d_iu", "vitamin_b12_ug", "sodium_mg",
]

# Friendly display names
NUTRIENT_LABELS = {
    "calories":      "Calories (kcal)",
    "protein_g":     "Protein (g)",
    "carbs_g":       "Carbohydrates (g)",
    "fat_g":         "Total Fat (g)",
    "fibre_g":       "Dietary Fibre (g)",
    "calcium_mg":    "Calcium (mg)",
    "iron_mg":       "Iron (mg)",
    "potassium_mg":  "Potassium (mg)",
    "magnesium_mg":  "Magnesium (mg)",
    "zinc_mg":       "Zinc (mg)",
    "vitamin_d_iu":  "Vitamin D (IU)",
    "vitamin_b12_ug":"Vitamin B12 (µg)",
    "sodium_mg":     "Sodium (mg)",
}

RDA_THRESHOLD = 0.80   # flag if daily total < 80% of RDA
SODIUM_CAP_RATIO = 1.0 # flag if sodium > 100% of upper limit (2300mg)


# ── Per-day nutrient totals ────────────────────────────────────────────────────

@dataclass
class DayNutrients:
    day:          int
    totals:       dict[str, float]     # nutrient → total for the day
    rda:          dict[str, float]     # nutrient → RDA target
    flags:        list[str] = field(default_factory=list)   # warning strings
    meal_breakdown: dict[str, dict] = field(default_factory=dict)

    @property
    def pct_rda(self) -> dict[str, float]:
        """Return each nutrient as % of RDA."""
        result = {}
        for k, rda_val in self.rda.items():
            if rda_val and rda_val > 0:
                result[k] = round((self.totals.get(k) or 0) / rda_val * 100, 1)
        return result

    def to_df_row(self) -> dict:
        row = {"day": self.day}
        pct = self.pct_rda
        for k in TRACKED:
            row[f"{k}_total"] = round(self.totals.get(k) or 0, 2)
            row[f"{k}_pct_rda"] = pct.get(k, 0.0)
        row["flagged"] = len(self.flags) > 0
        return row


def _meal_nutrients(meal: Meal) -> dict[str, float]:
    """
    Sum nutrient values across all foods in a meal.
    Returns None for a nutrient if NO food in the meal reported it,
    so we can distinguish "genuinely zero" from "data not available".
    """
    totals  = {n: None for n in TRACKED}
    for food in meal.foods:
        for n in TRACKED:
            val = food.get(n)
            # Ensure the value is neither Python None nor a Pandas/NumPy NaN
            if val is not None and not pd.isna(val):
                totals[n] = (totals[n] or 0.0) + float(val)
    return totals


def _day_nutrients(meals: dict[str, Meal]) -> dict[str, float]:
    """
    Sum nutrients across all meals in a day.
    A nutrient is None only if every meal returned None for it.
    """
    totals = {n: None for n in TRACKED}
    for meal in meals.values():
        for n, v in _meal_nutrients(meal).items():
            if v is not None:
                totals[n] = (totals[n] or 0.0) + v
    return totals


# ── Analyser ───────────────────────────────────────────────────────────────────

class NutrientAnalyser:
    """
    Analyses a WeekPlan against RDA targets for a given UserProfile.
    """

    def __init__(self, plan: WeekPlan, profile: UserProfile):
        self.plan    = plan
        self.profile = profile
        self.rda     = get_rda(profile.sex, profile.age, profile.calorie_target)

    def daily_report(self) -> list[DayNutrients]:
        """Return a DayNutrients object for each of the 7 days."""
        report = []
        for day in self.plan.days:
            totals = _day_nutrients(day.meals)
            meal_breakdown = {
                slot: _meal_nutrients(meal)
                for slot, meal in day.meals.items()
            }

            dn = DayNutrients(
                day=day.day,
                totals=totals,
                rda=self.rda,
                meal_breakdown=meal_breakdown,
            )

            # Flag deficiencies (< 80% RDA) and sodium excess
            for nutrient, rda_val in self.rda.items():
                if nutrient not in TRACKED or not rda_val:
                    continue
                actual = totals.get(nutrient)   # None = no data, 0.0 = genuinely zero
                if actual is None:
                    continue
                ratio  = actual / rda_val

                # Only flag if we actually have data — None means no foods
                # reported this nutrient, not that the value is zero
                if actual is None:
                    continue

                if nutrient == "sodium_mg":
                    if ratio > SODIUM_CAP_RATIO:
                        dn.flags.append(
                            f"Day {day.day}: Sodium {actual:.0f}mg exceeds "
                            f"{rda_val:.0f}mg upper limit "
                            f"({ratio*100:.0f}% of cap)"
                        )
                else:
                    if ratio < RDA_THRESHOLD:
                        label = NUTRIENT_LABELS.get(nutrient, nutrient)
                        dn.flags.append(
                            f"Day {day.day}: {label} {actual:.1f} is "
                            f"{ratio*100:.0f}% of RDA "
                            f"(target ≥ {rda_val:.1f})"
                        )

            report.append(dn)
        return report

    def rda_flags(self) -> list[str]:
        """Flat list of all RDA warning strings across the week."""
        return [flag for dn in self.daily_report() for flag in dn.flags]

    def weekly_summary(self) -> dict:
        """
        Weekly average for each nutrient + % of RDA.
        Only averages days where the nutrient was actually reported —
        days with no data (None) are excluded from the average, not
        treated as zero. This prevents genuine readings from being
        diluted by missing-data days.
        """
        report = self.daily_report()
        summary = {}
        for nutrient in TRACKED:
            # Only include days that have real data for this nutrient
            vals = [
                dn.totals.get(nutrient)
                for dn in report
                if dn.totals.get(nutrient) is not None
            ]
            if not vals:
                # Truly no data across the entire week
                label = NUTRIENT_LABELS.get(nutrient, nutrient)
                summary[label] = {"avg_daily": float("nan"), "pct_rda": float("nan")}
                continue

            avg   = sum(vals) / len(vals)
            rda_v = self.rda.get(nutrient) or 1
            pct   = round(avg / rda_v * 100, 1)
            label = NUTRIENT_LABELS.get(nutrient, nutrient)
            summary[label] = {"avg_daily": round(avg, 2), "pct_rda": pct}
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with one row per day, all nutrient totals and % RDA."""
        return pd.DataFrame([dn.to_df_row() for dn in self.daily_report()])

    def meal_df(self) -> pd.DataFrame:
        """
        Return a long-format DataFrame with one row per meal per day.
        Useful for the Streamlit analytics charts.
        """
        rows = []
        for day in self.plan.days:
            for slot, meal in day.meals.items():
                mn = _meal_nutrients(meal)
                rows.append({
                    "day":    day.day,
                    "slot":   slot,
                    "foods":  " + ".join(f["name"][:30] for f in meal.foods),
                    **{NUTRIENT_LABELS.get(k, k): round(v, 2) for k, v in mn.items()},
                })
        return pd.DataFrame(rows)


# ── Explain feature ────────────────────────────────────────────────────────────

def explain_exclusion(
    fdc_id: int,
    food_name: str,
    log: ExclusionLog,
    profile: UserProfile,
) -> str:
    """
    Return a plain-English explanation of why a food was excluded for a profile.
    Combines the ExclusionLog reasons with profile context.

    Example output:
        "Garlic bread was excluded for Priya because:
         • High-FODMAP food, unsafe for IBS
         • Contains gluten (allergen)"
    """
    reasons = log.get(fdc_id)

    if not reasons:
        return (
            f'"{food_name}" was not excluded — it should be in the safe food pool. '
            f"If it's missing from the plan, it may not have matched the calorie "
            f"target or was crowded out by diversity enforcement."
        )

    lines = [f'"{food_name}" was excluded for {profile.name} because:']
    for r in reasons:
        lines.append(f"  • {r}")

    return "\n".join(lines)


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from db import load_all_foods
    from filters import FilterEngine, TEST_PERSONAS
    from planner import MealPlanner

    df = load_all_foods()
    engine = FilterEngine(df)

    for persona_name, profile in list(TEST_PERSONAS.items())[:2]:
        print(f"\n{'='*60}")
        print(f"Persona: {profile.name}")
        safe, log = engine.apply(profile)
        planner   = MealPlanner(safe)
        plan      = planner.generate(profile, seed=42)
        analyser  = NutrientAnalyser(plan, profile)

        summary = analyser.weekly_summary()
        print("\nWeekly Averages:")
        for label, vals in summary.items():
            bar_len = min(int(vals["pct_rda"] / 5), 20)
            bar = "█" * bar_len
            flag = " ⚠️" if vals["pct_rda"] < 80 else ""
            print(f"  {label:<25} {vals['avg_daily']:>8.1f}  {vals['pct_rda']:>5.1f}% {bar}{flag}")

        flags = analyser.rda_flags()
        if flags:
            print(f"\n⚠️  {len(flags)} RDA flags:")
            for f in flags[:5]:
                print(f"   {f}")
        else:
            print("\n✅ All nutrients within RDA targets")

        # Test explain feature
        sample_ids = list(log.all().keys())[:2]
        for fid in sample_ids:
            name = df.loc[df["fdc_id"] == fid, "name"]
            fname = name.values[0] if len(name) else f"fdc_id={fid}"
            print(f"\n{explain_exclusion(fid, fname, log, profile)}")
