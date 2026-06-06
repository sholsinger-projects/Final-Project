"""
app.py — Phase 5: NutriAI Streamlit UI
Run with: streamlit run app.py
"""

import time
import streamlit as st
import pandas as pd
import re
import plotly.graph_objects as go
import plotly.express as px

from db import load_all_foods, db_summary
from filters import FilterEngine, UserProfile, ExclusionLog
from planner import MealPlanner, WeekPlan
from nutrients import NutrientAnalyser, NUTRIENT_LABELS, explain_exclusion

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriAI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f4f1eb !important;
    color: #1a1a1a !important;
}

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

/* Force light background on all major containers */
section[data-testid="stMain"] > div,
div[data-testid="stAppViewContainer"],
div[data-testid="block-container"] {
    background-color: #f4f1eb !important;
}

/* Tabs */
div[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600;
    color: #444 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2d6a4f !important;
    border-bottom-color: #2d6a4f !important;
}

/* Meal cards */
.meal-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1rem 1.25rem 0.9rem;
    margin-bottom: 0.6rem;
    border: 1px solid #e2ddd4;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.meal-slot {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #2d6a4f;
    margin-bottom: 0.3rem;
}

.meal-foods {
    font-size: 0.92rem;
    color: #1a1a1a;
    font-weight: 500;
    line-height: 1.4;
    margin-bottom: 0.35rem;
}

.meal-food-item {
    display: block;
    padding: 0.1rem 0;
    color: #2a2a2a;
}

.meal-macros {
    display: flex;
    gap: 0.75rem;
    font-size: 0.76rem;
    color: #777;
    margin-top: 0.1rem;
}

.macro-pill {
    background: #f0ede6;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-weight: 600;
    color: #555;
}

.macro-pill.kcal { background: #e8f5ee; color: #1b4332; }

.day-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    font-weight: 400;
    color: #1a1a1a;
    padding: 0.8rem 0 0.4rem 0;
    border-bottom: 2px solid #2d6a4f;
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}

.day-stats {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: #777;
    font-weight: 400;
}

.warn-box {
    background: #fff8e7;
    border: 1px solid #f0c040;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #7a5c00;
}

.flag-box {
    background: #fff1f0;
    border-left: 3px solid #e85d4a;
    border-radius: 6px;
    padding: 0.5rem 0.85rem;
    margin: 0.3rem 0;
    font-size: 0.83rem;
    color: #b03a2e;
}

.ok-box {
    background: #edfaf3;
    border-left: 3px solid #2d6a4f;
    border-radius: 6px;
    padding: 0.5rem 0.85rem;
    font-size: 0.83rem;
    color: #1a4731;
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: #1a2e22 !important;
}
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] span,
div[data-testid="stSidebar"] div {
    color: #d4ead9 !important;
}

.sidebar-section {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7aab8a !important;
    margin: 1.3rem 0 0.5rem 0;
}

/* Generate button */
.stButton > button {
    background: #2d6a4f !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #1b4332 !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    border: 1px solid #e2ddd4;
}
div[data-testid="stMetricLabel"] { color: #555555 !important; font-size: 0.82rem !important; }
div[data-testid="stMetricLabel"] p { color: #555555 !important; }
div[data-testid="stMetricValue"] { color: #1a1a1a !important; }
div[data-testid="stMetricValue"] > div { color: #1a1a1a !important; }
div[data-testid="stMetricDelta"] { color: #2d6a4f !important; }

/* Force all main content text to dark — prevents sidebar dark theme bleeding */
section[data-testid="stMain"] p,
section[data-testid="stMain"] span,
section[data-testid="stMain"] label,
section[data-testid="stMain"] div:not([data-testid="stSidebar"]) {
    color: #1a1a1a;
}

/* Landing page stat cards specifically */
div[data-testid="stMetricLabel"] * { color: #666666 !important; }
div[data-testid="stMetricValue"] * { color: #1a1a1a !important; }

/* Tab labels */
button[data-baseweb="tab"] p { color: #444444 !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #2d6a4f !important; }

/* Headings */
section[data-testid="stMain"] h1,
section[data-testid="stMain"] h2,
section[data-testid="stMain"] h3 { color: #1a1a1a !important; }

/* Plotly chart container background */
div[data-testid="stPlotlyChart"] { background: white; border-radius: 10px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
for key, default in [
    ("plan", None), ("log", None), ("profile", None),
    ("analyser", None), ("generated", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading food database…")
def get_engine():
    df = load_all_foods()
    return FilterEngine(df), df

@st.cache_resource(show_spinner=False)
def get_db_stats():
    return db_summary()


# ── Sidebar — Profile Builder ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥗 NutriAI")
    st.markdown("*Personalised 7-day meal plans*")
    st.divider()

    st.markdown('<div class="sidebar-section">Personal Info</div>', unsafe_allow_html=True)
    name   = st.text_input("Your name", value="Alex")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=90, value=30)
    with col2:
        sex = st.selectbox("Sex", ["female", "male"])
    calorie_target = st.slider("Daily calorie target", 1200, 3500, 2000, step=50)

    st.markdown('<div class="sidebar-section">Diet Type</div>', unsafe_allow_html=True)
    diet = st.selectbox("Diet preference", ["non_veg", "vegetarian", "vegan", "pescatarian"],
                        format_func=lambda x: x.replace("_", " ").title())

    st.markdown('<div class="sidebar-section">Clinical Conditions</div>', unsafe_allow_html=True)
    has_ibs          = st.checkbox("IBS / Low-FODMAP")
    has_gerd         = st.checkbox("GERD / Acid Reflux")
    has_diabetes     = st.checkbox("Type 2 Diabetes")
    has_hypertension = st.checkbox("Hypertension / DASH")

    st.markdown('<div class="sidebar-section">Allergens</div>', unsafe_allow_html=True)
    allergen_options = ["gluten", "dairy", "egg", "soy", "peanut",
                        "tree_nuts", "fish", "shellfish", "sesame"]
    allergens = st.multiselect("Select allergens to exclude", allergen_options)

    st.markdown('<div class="sidebar-section">Cultural / Religious</div>', unsafe_allow_html=True)
    no_pork = st.checkbox("No pork")
    no_beef = st.checkbox("No beef")

    st.divider()
    generate_btn = st.button("✦ Generate My Plan", use_container_width=True)


# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("# NutriAI")
st.markdown("##### AI-powered personalised nutrition planning")
st.divider()

if generate_btn:
    engine, df = get_engine()

    profile = UserProfile(
        name=name, age=int(age), sex=sex,
        calorie_target=float(calorie_target),
        diet=diet,
        has_ibs=has_ibs, has_gerd=has_gerd,
        has_diabetes=has_diabetes, has_hypertension=has_hypertension,
        allergens=allergens,
        no_pork=no_pork, no_beef=no_beef,
    )

    with st.spinner("Filtering safe foods…"):
        t0 = time.perf_counter()
        safe_df, log = engine.apply(profile)

    if safe_df.empty:
        st.error("No safe foods found for this profile. Try relaxing some constraints.")
        st.stop()

    with st.spinner(f"Generating 7-day plan from {len(safe_df):,} safe foods…"):
        planner = MealPlanner(safe_df)
        plan    = planner.generate(profile)

    analyser = NutrientAnalyser(plan, profile)

    st.session_state.plan      = plan
    st.session_state.log       = log
    st.session_state.profile   = profile
    st.session_state.analyser  = analyser
    st.session_state.generated = True
    elapsed = time.perf_counter() - t0
    st.success(f"Plan generated in {elapsed:.1f}s — {len(safe_df):,} safe foods, diversity score {plan.diversity_score:.2f}")


# ── Display plan ───────────────────────────────────────────────────────────────
if st.session_state.generated and st.session_state.plan:
    plan:     WeekPlan        = st.session_state.plan
    profile:  UserProfile     = st.session_state.profile
    analyser: NutrientAnalyser = st.session_state.analyser
    log:      ExclusionLog    = st.session_state.log

    tab1, tab2, tab3 = st.tabs(["📅  7-Day Plan", "📊  Nutrition Analysis", "🔍  Explain"])

    # ── Tab 1: Plan ────────────────────────────────────────────────────────────
    with tab1:
        col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
        with col_meta1:
            st.metric("Avg Daily Calories", f"{plan.avg_daily_calories:.0f} kcal",
                      delta=f"{plan.avg_daily_calories - profile.calorie_target:+.0f} vs target")
        with col_meta2:
            st.metric("Diversity Score", f"{plan.diversity_score:.2f}",
                      delta="✓ Good" if plan.diversity_score >= 0.55 else "⚠ Low")
        with col_meta3:
            st.metric("Generation Time", f"{plan.generation_time_s:.1f}s",
                      delta="✓ < 60s" if plan.generation_time_s < 60 else "⚠ Slow")
        with col_meta4:
            flags = analyser.rda_flags()
            st.metric("RDA Flags", len(flags),
                      delta="✓ On target" if len(flags) == 0 else f"{len(flags)} nutrients flagged")

        if plan.warnings:
            for w in plan.warnings:
                st.markdown(f'<div class="warn-box">⚠️ {w}</div>', unsafe_allow_html=True)

        st.divider()

        SLOT_ICONS = {"breakfast": "☀️", "lunch": "🌿", "dinner": "🌙"}

        # One day per row, full width — much more readable
        for day in plan.days:
            st.markdown(
                f'<div class="day-header">'
                f'Day {day.day}'
                f'<span class="day-stats">{day.total_calories:.0f} kcal &nbsp;·&nbsp; {day.total_sodium:.0f} mg sodium</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(3)
            for col_idx, slot in enumerate(["breakfast", "lunch", "dinner"]):
                meal = day.meals.get(slot)
                with cols[col_idx]:
                    if not meal:
                        st.markdown(
                            f'<div class="meal-card"><div class="meal-slot">{SLOT_ICONS[slot]} {slot}</div>'
                            f'<div class="meal-foods" style="color:#aaa;font-style:italic">No meal planned</div></div>',
                            unsafe_allow_html=True
                        )
                        continue

                    def clean_food_name(raw: str) -> str:
                        """
                        Convert USDA-style names to readable form.
                        'Bread, white, commercially prepared, low sodium' → 'White Bread'
                        'Chicken, broiler, rotisserie, BBQ, breast' → 'Rotisserie BBQ Chicken Breast'
                        'From Fast Food / Restaurant Ns As To Type Beans' → 'Beans'
                        
                        Strategy: Strip administrative placeholders, drop trailing qualifiers 
                        after the 2nd comma, reverse the leading category, and title-case.
                        """
                        if not raw:
                            return ""

                        # 1. Strip annoying USDA administrative restaurant prefixes
                        cleaned_raw = re.sub(r"from fast food\s*/\s*restaurant\s*,?\s*", "", raw, flags=re.IGNORECASE)
                        cleaned_raw = re.sub(r"restaurant\s*,?\s*", "", cleaned_raw, flags=re.IGNORECASE)
                        
                        # Clear out bureaucratic "Not Further Specified" (NFS / NS) placeholders
                        cleaned_raw = re.sub(r"\bns\s+as\s+to\s+type\b", "", raw, flags=re.IGNORECASE)
                        cleaned_raw = re.sub(r"\bnfs\b", "", cleaned_raw, flags=re.IGNORECASE)
                        cleaned_raw = re.sub(r"\bns\b", "", cleaned_raw, flags=re.IGNORECASE)
                        cleaned_raw = re.sub(r"\bnot\s+specified\b", "", cleaned_raw, flags=re.IGNORECASE)

                        # 3. Clean up formatting artifacts left behind by the removals
                        cleaned_raw = re.sub(r"\s+", " ", cleaned_raw)
                        cleaned_raw = re.sub(r",\s*,", ",", cleaned_raw)
                        cleaned_raw = cleaned_raw.strip().strip(",").strip()

                        # 4. Run original comma parsing split logic
                        parts = [p.strip() for p in cleaned_raw.split(",")]
                        if len(parts) == 1:
                            return parts[0].title()[:90]
                            
                        # Drop generic trailing qualifiers
                        drop = {
                            "commercially prepared", "home prepared", "cooked", "raw",
                            "unenriched", "enriched", "frozen", "canned", "ready-to-heat",
                            "prepared from recipe", "low sodium", "reduced fat", "whole grain"
                        }
                        kept = [p for p in parts[1:] if p.lower() not in drop]
                        
                        # Rebuild: meaningful descriptors + base noun
                        base = parts[0]
                        desc = " ".join(kept[:2])   # max 2 descriptors
                        name = f"{desc} {base}".strip() if desc else base
                        
                        # Expanded slice length ensures long consumer product descriptions aren't abruptly truncated
                        return name.title()[:90]

                    food_items_html = "".join(
                        f'<span class="meal-food-item">• {clean_food_name(f["name"])}</span>'
                        for f in meal.foods
                    )

                    st.markdown(f"""
                    <div class="meal-card">
                        <div class="meal-slot">{SLOT_ICONS[slot]}&nbsp; {slot}</div>
                        <div class="meal-foods">{food_items_html}</div>
                        <div class="meal-macros">
                            <span class="macro-pill kcal">{meal.calories:.0f} kcal</span>
                            <span class="macro-pill">Protein {meal.protein_g:.0f}g</span>
                            <span class="macro-pill">Carbs {meal.carbs_g:.0f}g</span>
                            <span class="macro-pill">Fat {meal.fat_g:.0f}g</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

    # ── Tab 2: Nutrition Analysis ──────────────────────────────────────────────
    with tab2:
        summary = analyser.weekly_summary()
        flags   = analyser.rda_flags()

        # Summary bar chart — only show nutrients with known data
        labels, pct_vals, colours, bar_text = [], [], [], []
        for label, vals in summary.items():
            pct = vals["pct_rda"]
            is_nan = (pct != pct)
            short_label = label.split(" (")[0]
            labels.append(short_label)
            if is_nan:
                pct_vals.append(4)   # tiny sliver so bar is visible
                colours.append("#e8e4dd")
                bar_text.append("—")
            else:
                pct_vals.append(min(pct, 200))
                colours.append("#e85d4a" if pct < 80 else "#2d6a4f" if pct <= 120 else "#f0a500")
                bar_text.append(f"{pct:.0f}%")

        fig = go.Figure(go.Bar(
            x=labels, y=pct_vals,
            marker_color=colours,
            text=bar_text,
            textposition="outside",
        ))
        fig.add_hline(y=80,  line_dash="dot",  line_color="#e85d4a", annotation_text="80% RDA")
        fig.update_layout(
            title="Weekly Average — % of RDA",
            yaxis_title="% of RDA", xaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="DM Sans",
            font=dict(color="#1a1a1a", size=11),
            title_font=dict(color="#1a1a1a", size=13),
            height=380, margin=dict(t=50, b=80),
            yaxis=dict(range=[0, 220], tickfont=dict(color="#333333")),
            xaxis=dict(tickfont=dict(color="#333333"), tickangle=-30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Daily calorie trend
        daily_cals = [d.total_calories for d in plan.days]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[f"Day {i+1}" for i in range(7)],
            y=daily_cals,
            mode="lines+markers",
            line=dict(color="#2d6a4f", width=2.5),
            marker=dict(size=8, color="#2d6a4f"),
            name="Actual",
        ))
        fig2.add_hline(y=profile.calorie_target, line_dash="dash",
                       line_color="#f0a500", annotation_text="Target")
        fig2.update_layout(
            title="Daily Calorie Distribution",
            yaxis_title="kcal", xaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="DM Sans", height=280,
            font=dict(color="#1a1a1a", size=11),
            title_font=dict(color="#1a1a1a", size=13),
            yaxis=dict(tickfont=dict(color="#333333")),
            xaxis=dict(tickfont=dict(color="#333333")),
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # RDA flags grouped by day
        st.subheader("RDA Flags")
        if flags:
            # Group by day number parsed from flag string "Day N: ..."
            from collections import defaultdict
            by_day = defaultdict(list)
            for f in flags:
                import re as _re
                m = _re.match(r"Day (\d+):", f)
                day_key = int(m.group(1)) if m else 0
                by_day[day_key].append(f)

            for day_num in sorted(by_day.keys()):
                day_flags = by_day[day_num]
                label = f"Day {day_num}" if day_num else "General"
                with st.expander(f"⚠ {label} — {len(day_flags)} flag{'s' if len(day_flags)>1 else ''}", expanded=False):
                    for f in day_flags:
                        # Strip "Day N: " prefix since it's already in the header
                        text = _re.sub(r"^Day \d+: ", "", f)
                        st.markdown(f'<div class="flag-box">⚠ {text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-box">✓ All nutrients within RDA targets</div>',
                        unsafe_allow_html=True)

        # Full weekly averages table
        with st.expander("Full nutrient breakdown"):
            rows = []
            for label, vals in summary.items():
                avg = vals["avg_daily"]
                pct = vals["pct_rda"]
                is_nan = (avg != avg) or (pct != pct)  # NaN check
                rows.append({
                    "Nutrient":  label,
                    "Daily Avg": "Unknown" if is_nan else f"{avg:.1f}",
                    "% of RDA":  "Unknown" if is_nan else f"{pct:.1f}%",
                    "Status":    "—" if is_nan else (
                        "✓" if 80 <= pct <= 150 else ("⚠ Low" if pct < 80 else "⚠ High")
                    ),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Tab 3: Explain ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Why was a food excluded?")
        st.markdown("Enter a food name to see why it was removed from your safe pool.")

        query = st.text_input("Food name", placeholder="e.g. garlic bread, cheddar cheese…")

        if query:
            engine_obj, df = get_engine()
            matches = df[df["name"].str.contains(query, case=False, na=False)].head(5)

            if matches.empty:
                st.info("No foods found matching that name.")
            else:
                for _, row in matches.iterrows():
                    fid   = row["fdc_id"]
                    fname = row["name"]
                    explanation = explain_exclusion(fid, fname, log, profile)
                    if "not excluded" in explanation:
                        st.markdown(f'<div class="ok-box">✓ <b>{fname}</b> — in your safe food pool</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="flag-box">❌ <b>{fname}</b></div>',
                                    unsafe_allow_html=True)
                        for line in explanation.split("\n")[1:]:
                            if line.strip():
                                st.markdown(f"&nbsp;&nbsp;{line.strip()}")

        st.divider()
        st.subheader("Profile Summary")
        p = st.session_state.profile
        conditions = [c for c, v in [
            ("IBS/FODMAP", p.has_ibs), ("GERD", p.has_gerd),
            ("Diabetes", p.has_diabetes), ("Hypertension", p.has_hypertension),
        ] if v]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {p.name}")
            st.markdown(f"**Age/Sex:** {p.age} / {p.sex}")
            st.markdown(f"**Diet:** {p.diet.replace('_',' ').title()}")
            st.markdown(f"**Calorie target:** {p.calorie_target:.0f} kcal")
        with col2:
            st.markdown(f"**Conditions:** {', '.join(conditions) if conditions else 'None'}")
            st.markdown(f"**Allergens:** {', '.join(p.allergens) if p.allergens else 'None'}")
            st.markdown(f"**No pork:** {'Yes' if p.no_pork else 'No'}")
            st.markdown(f"**No beef:** {'Yes' if p.no_beef else 'No'}")

else:
    # Landing state — show DB stats and instructions
    st.markdown("### Get started")
    st.markdown("Fill in your profile in the sidebar and click **✦ Generate My Plan**.")
    st.divider()

    try:
        stats = get_db_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Foods in database", f"{stats['total']:,}")
        c2.metric("Vegan-safe", f"{stats['vegan']:,}")
        c3.metric("FODMAP-tagged", f"{stats['fodmap']:,}")
        c4.metric("Allergen-indexed", "9 types")
    except Exception:
        st.info("Database loading… run `python ingest.py --api-key YOUR_KEY` first.")
