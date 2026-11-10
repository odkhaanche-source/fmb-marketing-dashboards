"""FMB Residential Sales Dashboard (Streamlit). Run: streamlit run app_residential.py"""
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def find(name):
    """Find a data file anywhere inside the repo, so folder layout does not matter."""
    hits = sorted(ROOT.rglob(name))
    if not hits:
        found = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*.csv"))[:25]
        st.error(f"Data file '{name}' was not found in the repository. CSV files found: {found or 'none'}. "
                 "Upload the 'data' folder (with data/clean inside) to GitHub and reboot the app.")
        st.stop()
    return hits[0]
sns.set_theme(style="whitegrid"); NAVY, ORANGE, RED, GREEN, GREY = "#1F3A5F", "#E07B39", "#C0392B", "#2E8B57", "#9AA5B1"

def bar(ax, s, color=NAVY, fmt="{:.1f}%", ref=None):
    s.plot.barh(ax=ax, color=color); ax.set_ylabel("")
    if ref is not None: ax.axvline(ref, color=RED, ls="--")
    for i, v in enumerate(s.values): ax.text(v, i, " " + fmt.format(v), va="center", fontsize=9)

st.set_page_config(page_title="FMB Residential Sales", layout="wide")

@st.cache_data
def load():
    r = pd.read_csv(find("residential_leads_clean.csv"), parse_dates=["lead_date"])
    c = pd.read_csv(find("campaign_performance_clean.csv"), parse_dates=["month_dt"])
    return r, c
r_all, c_all = load()

st.title("Residential Flat Sales - Marketing Analytics")
st.caption("Objective: raise lead-to-booking conversion and cut marketing cost per booking. Synthetic data, Jan 2024 to Dec 2025.")

st.sidebar.header("Filters")
d0, d1 = r_all.lead_date.min().date(), r_all.lead_date.max().date()
dr = st.sidebar.date_input("Lead date range", (d0, d1), min_value=d0, max_value=d1)
start, end = (pd.to_datetime(dr[0]), pd.to_datetime(dr[1])) if len(dr) == 2 else (pd.to_datetime(d0), pd.to_datetime(d1))
chs = st.sidebar.multiselect("Channels", sorted(r_all.channel.unique()), default=sorted(r_all.channel.unique()))
prj = st.sidebar.multiselect("Projects", sorted(r_all.project_name.unique()), default=sorted(r_all.project_name.unique()))
r = r_all[r_all.lead_date.between(start, end) & r_all.channel.isin(chs) & r_all.project_name.isin(prj)]
if r.empty: st.warning("Filters return no data - widen the selection."); st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Channels & ROI", "Why do leads convert?", "Correlation explorer", "Insights"])

with tab1:
    k = st.columns(5)
    k[0].metric("Leads", f"{len(r):,}"); k[1].metric("Bookings", f"{int(r.converted.sum()):,}")
    k[2].metric("Lead-to-sale", f"{r.converted.mean() * 100:.1f}%")
    k[3].metric("Marketing cost / booking", f"{r.marketing_cost.sum() / max(r.converted.sum(), 1):,.0f}")
    k[4].metric("Avg days to close", f"{r.days_to_close.mean():.0f}")
    m = r.groupby("lead_month").agg(leads=("lead_id", "count"), conv=("converted", "mean")).reset_index(); m["conv"] *= 100
    fig, ax = plt.subplots(figsize=(12, 3.8)); ax.bar(m.lead_month, m.leads, color=NAVY); ax.tick_params(axis="x", rotation=70, labelsize=8)
    ax2 = ax.twinx(); ax2.plot(m.lead_month, m.conv, color=ORANGE, marker="o"); ax2.grid(False); ax.set_ylabel("Leads"); ax2.set_ylabel("Conversion %")
    ax.set_title("Monthly leads and conversion"); st.pyplot(fig)
    st.markdown("**Reading the chart:** volume and conversion both rise in Oct-Nov (festive season), the best window to concentrate spend and sales staff.")
    pj = r.groupby("project_name").agg(leads=("lead_id", "count"), conv=("converted", "mean"), price=("price_per_sqft", "first")); pj["conv"] *= 100
    st.subheader("Project performance"); st.dataframe(pj.round(1).sort_values("conv", ascending=False), use_container_width=True)

with tab2:
    ch = r.groupby("channel").agg(leads=("lead_id", "count"), sales=("converted", "sum"), conv=("converted", "mean"), cost=("marketing_cost", "sum"), rev=("booking_value", "sum"))
    ch["conv"] *= 100; ch["CAC"] = ch.cost / ch.sales.replace(0, np.nan); ch["revenue_per_cost"] = ch.rev / ch.cost
    ch["share_of_leads_%"] = ch.leads / ch.leads.sum() * 100; ch["share_of_sales_%"] = ch.sales / ch.sales.sum() * 100
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4)); bar(ax, ch.conv.sort_values(), ORANGE, ref=r.converted.mean() * 100); ax.set_title("Conversion % by channel"); st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 4)); bar(ax, ch.CAC.dropna().sort_values(ascending=False), NAVY, fmt="{:,.0f}"); ax.set_title("Marketing cost per booking (CAC)"); st.pyplot(fig)
    st.dataframe(ch.round(1).sort_values("conv", ascending=False), use_container_width=True)
    st.markdown("**Insight:** where *share of leads* far exceeds *share of sales*, the channel brings volume, not buyers.")
    st.subheader("Budget efficiency (campaign table)")
    cs = c_all[c_all.segment == "Residential"].groupby("channel").agg(spend=("spend", "sum"), conv=("conversions_reported", "sum"))
    cs["efficiency_index"] = (cs.conv / cs.conv.sum()) / (cs.spend / cs.spend.sum())
    fig, ax = plt.subplots(figsize=(8, 3.8)); s = cs.efficiency_index.sort_values()
    s.plot.barh(ax=ax, color=[GREEN if v >= 1 else RED for v in s]); ax.axvline(1, color="black"); ax.set_xlabel("share of sales / share of spend (>1 = efficient)"); ax.set_ylabel(""); st.pyplot(fig)

with tab3:
    st.markdown("Pick a factor: conversion by level (descriptive) plus a chi-square test of whether the difference is real (diagnostic).")
    opts = {"Response speed": "response_bucket", "Site visits": "site_visits", "Budget vs price": "affordability_band", "Loan pre-approved": "loan_preapproved",
            "Festive season": "festive_season", "Local lead (<=5 km)": "local_lead", "Unit type": "unit_type", "Age group": "age_group", "Income quartile": "income_band", "Occupation": "occupation", "Project": "project_name"}
    pick = st.selectbox("Factor", list(opts)); col = opts[pick]
    g = r.groupby(col, observed=True).agg(leads=("lead_id", "count"), conv=("converted", "mean")); g["conv"] *= 100
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(g.index.astype(str), g.conv, color=NAVY); ax.axhline(r.converted.mean() * 100, color=RED, ls="--")
    for i, v in enumerate(g.conv): ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Conversion %"); ax.tick_params(axis="x", rotation=20); st.pyplot(fig)
    ct = pd.crosstab(r[col], r.converted)
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, _, _ = stats.chi2_contingency(ct); v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        st.info(f"Chi-square p-value = {p:.4g} | Cramer's V = {v:.2f} -> " + ("statistically significant relationship" if p < .05 else "no significant relationship"))
    st.dataframe(g.round(1), use_container_width=True)
    st.subheader("Meta Ads drill-down (largest lead source, weak conversion)")
    r2 = r.assign(group=np.where(r.channel == "Meta Ads", "Meta Ads", "Other channels"))
    st.dataframe(r2.groupby("group").agg(leads=("lead_id", "count"), conv_pct=("converted", lambda x: x.mean() * 100), median_response_hrs=("response_time_hrs", "median"),
                 avg_site_visits=("site_visits", "mean"), median_budget_ratio=("budget_to_price_ratio", "median"), loan_share=("loan_preapproved", "mean")).round(2))

with tab4:
    cols = ["age", "annual_income", "budget_to_price_ratio", "response_time_hrs", "distance_from_site_km", "follow_up_calls", "site_visits", "loan_preapproved", "festive_season", "home_loan_rate_pct", "marketing_cost", "converted"]
    method = st.selectbox("Method", ["spearman", "pearson"]); cm = r[cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(9, 7)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot_kws={"size": 8}, ax=ax); st.pyplot(fig)
    x = st.selectbox("Scatter X", cols[:-1]); y = st.selectbox("Scatter Y", cols[:-1], index=3)
    fig, ax = plt.subplots(figsize=(7, 4)); sns.scatterplot(data=r.sample(min(len(r), 1500), random_state=1), x=x, y=y, hue="converted", alpha=.55, palette=[GREY, ORANGE], ax=ax); st.pyplot(fig)

with tab5:
    st.subheader("Data-quality log")
    st.dataframe(pd.read_csv(find("cleaning_log.csv")), use_container_width=True)
    st.subheader("Recommended actions")
    st.markdown("""
1. **Respond within 2 hours**: conversion falls from about 18% (under 2h) to about 2% (over 24h).
2. **Push every serious lead to repeat site visits**: about 4% with no visit, about 23% with three.
3. **Shift budget** from Property Portals, Hoardings and Meta to Referral, Walk-in drives, WhatsApp nurture and Brokers.
4. **Qualify budget and loan status early**: surplus-budget leads convert about 3x under-budget leads.
5. **Concentrate launches and offers in Oct-Nov**: about 17% festive vs 12% otherwise.
""")
