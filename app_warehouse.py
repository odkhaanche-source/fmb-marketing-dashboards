"""FMB Warehouse Leasing Dashboard (Streamlit). Run: streamlit run app_warehouse.py"""
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from pathlib import Path
DATA = Path(__file__).parent / "data" / "clean"
TABLES = Path(__file__).parent / "tables"
sns.set_theme(style="whitegrid"); NAVY, ORANGE, RED, GREEN, GREY = "#1F3A5F", "#E07B39", "#C0392B", "#2E8B57", "#9AA5B1"

def bar(ax, s, color=NAVY, fmt="{:.1f}%", ref=None):
    s.plot.barh(ax=ax, color=color); ax.set_ylabel("")
    if ref is not None: ax.axvline(ref, color=RED, ls="--")
    for i, v in enumerate(s.values): ax.text(v, i, " " + fmt.format(v), va="center", fontsize=9)

st.set_page_config(page_title="FMB Warehouse Leasing", layout="wide")

@st.cache_data
def load():
    w = pd.read_csv(DATA / "warehouse_leads_clean.csv", parse_dates=["lead_date"])
    c = pd.read_csv(DATA / "campaign_performance_clean.csv", parse_dates=["month_dt"])
    return w, c
w_all, c_all = load()

st.title("Warehouse Leasing - Marketing Analytics")
st.caption("Objective: raise lead-to-lease conversion and cut marketing cost per lease signed. Synthetic data, Jan 2024 to Dec 2025.")

st.sidebar.header("Filters")
d0, d1 = w_all.lead_date.min().date(), w_all.lead_date.max().date()
dr = st.sidebar.date_input("Lead date range", (d0, d1), min_value=d0, max_value=d1)
start, end = (pd.to_datetime(dr[0]), pd.to_datetime(dr[1])) if len(dr) == 2 else (pd.to_datetime(d0), pd.to_datetime(d1))
chs = st.sidebar.multiselect("Channels", sorted(w_all.channel.unique()), default=sorted(w_all.channel.unique()))
prj = st.sidebar.multiselect("Warehouse parks", sorted(w_all.project_name.unique()), default=sorted(w_all.project_name.unique()))
cts = st.sidebar.multiselect("Client types", sorted(w_all.client_type.unique()), default=sorted(w_all.client_type.unique()))
w = w_all[w_all.lead_date.between(start, end) & w_all.channel.isin(chs) & w_all.project_name.isin(prj) & w_all.client_type.isin(cts)]
if w.empty: st.warning("Filters return no data - widen the selection."); st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Channels & clients", "Why do leads convert?", "Correlation explorer", "Insights"])

with tab1:
    k = st.columns(5)
    k[0].metric("Leads", f"{len(w):,}"); k[1].metric("Leases signed", f"{int(w.converted.sum()):,}")
    k[2].metric("Lead-to-lease", f"{w.converted.mean() * 100:.1f}%")
    k[3].metric("Marketing cost / lease", f"{w.marketing_cost.sum() / max(w.converted.sum(), 1):,.0f}")
    k[4].metric("Avg lease value", f"{w.total_lease_value.mean():,.0f}")
    m = w.groupby("lead_month").agg(leads=("lead_id", "count"), conv=("converted", "mean")).reset_index(); m["conv"] *= 100
    fig, ax = plt.subplots(figsize=(12, 3.8)); ax.bar(m.lead_month, m.leads, color=NAVY); ax.tick_params(axis="x", rotation=70, labelsize=8)
    ax2 = ax.twinx(); ax2.plot(m.lead_month, m.conv, color=ORANGE, marker="o"); ax2.grid(False); ax.set_ylabel("Leads"); ax2.set_ylabel("Conversion %")
    ax.set_title("Monthly warehouse leads and conversion"); st.pyplot(fig)
    pk = w.groupby("project_name").agg(leads=("lead_id", "count"), leases=("converted", "sum"), conv=("converted", "mean"), rent=("rent_per_sqft_month", "first")); pk["conv"] *= 100
    st.subheader("Park performance"); st.dataframe(pk.round(1).sort_values("conv", ascending=False), use_container_width=True)

with tab2:
    wc = w.groupby("channel").agg(leads=("lead_id", "count"), leases=("converted", "sum"), conv=("converted", "mean"), cost=("marketing_cost", "sum"), value=("total_lease_value", "sum"))
    wc["conv"] *= 100; wc["CAC"] = wc.cost / wc.leases.replace(0, np.nan); wc["lease_value_per_cost"] = wc.value / wc.cost
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4)); bar(ax, wc.conv.sort_values(), NAVY, ref=w.converted.mean() * 100); ax.set_title("Lease conversion % by channel"); st.pyplot(fig)
    with b:
        t = w.groupby("client_type").converted.mean().mul(100).sort_values(); fig, ax = plt.subplots(figsize=(6, 4)); bar(ax, t, ORANGE, ref=w.converted.mean() * 100); ax.set_title("Conversion % by client type"); st.pyplot(fig)
    st.dataframe(wc.round(1).sort_values("conv", ascending=False), use_container_width=True)
    st.subheader("Budget efficiency (campaign table)")
    cs = c_all[c_all.segment == "Warehouse"].groupby("channel").agg(spend=("spend", "sum"), conv=("conversions_reported", "sum"))
    cs["efficiency_index"] = (cs.conv / cs.conv.sum()) / (cs.spend / cs.spend.sum())
    fig, ax = plt.subplots(figsize=(8, 3.8)); s = cs.efficiency_index.sort_values()
    s.plot.barh(ax=ax, color=[GREEN if v >= 1 else RED for v in s]); ax.axvline(1, color="black"); ax.set_xlabel("share of sales / share of spend (>1 = efficient)"); ax.set_ylabel(""); st.pyplot(fig)

with tab3:
    st.markdown("Pick a factor: conversion by level plus a chi-square test of whether the difference is real.")
    wf = st.selectbox("Factor", ["proposal_sent", "decision_maker_contacted", "response_bucket", "size_band", "lease_term_years", "client_type", "project_name"])
    g = w.groupby(wf, observed=True).converted.mean().mul(100)
    fig, ax = plt.subplots(figsize=(8, 3.6)); ax.bar(g.index.astype(str), g.values, color=NAVY); ax.axhline(w.converted.mean() * 100, color=RED, ls="--"); ax.set_ylabel("Conversion %")
    for i, v in enumerate(g.values): ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig)
    ct = pd.crosstab(w[wf], w.converted)
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, _, _ = stats.chi2_contingency(ct); v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        st.info(f"Chi-square p-value = {p:.4g} | Cramer's V = {v:.2f} -> " + ("statistically significant relationship" if p < .05 else "no significant relationship"))
    w2 = w.assign(rent_gap_band=pd.cut(w.rent_gap_pct, [-100, -5, 0, 5, 100], labels=["Below mkt >5%", "0 to -5%", "0 to +5%", "Above mkt >5%"]))
    gb = w2.groupby("rent_gap_band", observed=True).converted.mean().mul(100)
    st.subheader("Quoted rent versus market rent"); fig, ax = plt.subplots(figsize=(7, 3.4)); ax.bar(gb.index.astype(str), gb.values, color=ORANGE); ax.axhline(w.converted.mean() * 100, color=RED, ls="--"); ax.set_ylabel("Conversion %")
    for i, v in enumerate(gb.values): ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig)

with tab4:
    cols = ["area_required_sqft", "lease_term_years", "quoted_rent_per_sqft", "response_time_hrs", "rent_gap_pct", "site_visits", "negotiation_rounds", "decision_maker_contacted", "proposal_sent", "marketing_cost", "converted"]
    method = st.selectbox("Method", ["spearman", "pearson"]); cm = w[cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(9, 7)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot_kws={"size": 8}, ax=ax); st.pyplot(fig)
    x = st.selectbox("Scatter X", cols[:-1]); y = st.selectbox("Scatter Y", cols[:-1], index=3)
    fig, ax = plt.subplots(figsize=(7, 4)); sns.scatterplot(data=w, x=x, y=y, hue="converted", alpha=.55, palette=[GREY, ORANGE], ax=ax); st.pyplot(fig)

with tab5:
    st.subheader("Recommended actions")
    st.markdown("""
1. **Reach the decision-maker**: about 22% conversion when reached vs about 8% when not; roughly half of leads never reach one.
2. **Send a proposal quickly**: about 21% with a proposal vs about 8% without.
3. **Respond within 4 hours**: about 19% vs about 7% after 48 hours.
4. **Keep quoted rent within +/-5% of market**: conversion falls as rent rises above market.
5. **Lean on Referral and Broker channels**; use Industry Expos selectively (highest cost per lease).
6. **Prioritise 3PL clients**: about 19% conversion vs 11-13% for other client types.
""")
