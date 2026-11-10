"""
FMB MARKETING ANALYTICS DASHBOARD  (Streamlit)
Run:  pip install streamlit pandas numpy matplotlib seaborn scipy
      streamlit run 04_streamlit_dashboard.py
Reads the cleaned files produced by 02_data_cleaning_transformation.py
"""
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats

st.set_page_config(page_title="FMB Marketing Analytics", layout="wide")
sns.set_theme(style="whitegrid"); NAVY, ORANGE, RED, GREEN = "#1F3A5F", "#E07B39", "#C0392B", "#2E8B57"

@st.cache_data
def load():
    r = pd.read_csv("data/clean/residential_leads_clean.csv", parse_dates=["lead_date"])
    w = pd.read_csv("data/clean/warehouse_leads_clean.csv", parse_dates=["lead_date"])
    c = pd.read_csv("data/clean/campaign_performance_clean.csv", parse_dates=["month_dt"])
    return r, w, c
r_all, w_all, c_all = load()

st.title("Family-Managed Construction Business - Marketing Analytics")
st.caption("Objective: lift lead-to-sale / lead-to-lease conversion and cut cost per acquisition across residential flats and warehouse leasing. (Synthetic data.)")

# ---------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
d0, d1 = r_all.lead_date.min().date(), r_all.lead_date.max().date()
dr = st.sidebar.date_input("Lead date range", (d0, d1), min_value=d0, max_value=d1)
if len(dr) == 2: start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
else: start, end = pd.to_datetime(d0), pd.to_datetime(d1)
res_ch = st.sidebar.multiselect("Residential channels", sorted(r_all.channel.unique()), default=sorted(r_all.channel.unique()))
projects = st.sidebar.multiselect("Residential projects", sorted(r_all.project_name.unique()), default=sorted(r_all.project_name.unique()))
wh_ch = st.sidebar.multiselect("Warehouse channels", sorted(w_all.channel.unique()), default=sorted(w_all.channel.unique()))
r = r_all[(r_all.lead_date.between(start, end)) & r_all.channel.isin(res_ch) & r_all.project_name.isin(projects)]
w = w_all[(w_all.lead_date.between(start, end)) & w_all.channel.isin(wh_ch)]
if r.empty or w.empty: st.warning("Filters return no data - widen the selection."); st.stop()

def bar(ax, s, color=NAVY, fmt="{:.1f}%", ref=None):
    s.plot.barh(ax=ax, color=color); ax.set_ylabel("")
    if ref is not None: ax.axvline(ref, color=RED, ls="--")
    for i, v in enumerate(s.values): ax.text(v, i, " " + fmt.format(v), va="center", fontsize=9)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Channels & ROI", "Why do leads convert? (Diagnostics)", "Warehouse leasing", "Correlation explorer", "Data quality & insights"])

# ---------------------------------------------------------------- TAB 1
with tab1:
    st.subheader("Residential flats")
    k = st.columns(5)
    k[0].metric("Leads", f"{len(r):,}"); k[1].metric("Bookings", f"{int(r.converted.sum()):,}")
    k[2].metric("Lead-to-sale", f"{r.converted.mean() * 100:.1f}%")
    k[3].metric("Marketing cost / booking", f"{r.marketing_cost.sum() / max(r.converted.sum(), 1):,.0f}")
    k[4].metric("Avg days to close", f"{r.days_to_close.mean():.0f}")
    st.subheader("Warehouse leasing")
    k = st.columns(5)
    k[0].metric("Leads", f"{len(w):,}"); k[1].metric("Leases signed", f"{int(w.converted.sum()):,}")
    k[2].metric("Lead-to-lease", f"{w.converted.mean() * 100:.1f}%")
    k[3].metric("Marketing cost / lease", f"{w.marketing_cost.sum() / max(w.converted.sum(), 1):,.0f}")
    k[4].metric("Avg lease value", f"{w.total_lease_value.mean():,.0f}")
    m = r.groupby("lead_month").agg(leads=("lead_id", "count"), conv=("converted", "mean")).reset_index(); m["conv"] *= 100
    fig, ax = plt.subplots(figsize=(12, 3.8)); ax.bar(m.lead_month, m.leads, color=NAVY); ax.tick_params(axis="x", rotation=70, labelsize=8)
    ax2 = ax.twinx(); ax2.plot(m.lead_month, m.conv, color=ORANGE, marker="o"); ax2.grid(False); ax.set_ylabel("Leads"); ax2.set_ylabel("Conversion %")
    ax.set_title("Monthly residential leads and conversion"); st.pyplot(fig)
    st.markdown("**Reading the chart:** lead volume rises in Oct-Nov (festive season) and so does conversion - the best time to concentrate spend and sales staff.")

# ---------------------------------------------------------------- TAB 2
with tab2:
    ch = r.groupby("channel").agg(leads=("lead_id", "count"), sales=("converted", "sum"), conv=("converted", "mean"), cost=("marketing_cost", "sum"), rev=("booking_value", "sum"))
    ch["conv"] *= 100; ch["CAC"] = ch.cost / ch.sales.replace(0, np.nan); ch["revenue_per_cost"] = ch.rev / ch.cost
    ch["share_of_leads_%"] = ch.leads / ch.leads.sum() * 100; ch["share_of_sales_%"] = ch.sales / ch.sales.sum() * 100
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4)); bar(ax, ch.conv.sort_values(), ORANGE, ref=r.converted.mean() * 100); ax.set_title("Conversion % by channel"); st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 4)); bar(ax, ch.CAC.sort_values(ascending=False), NAVY, fmt="{:,.0f}"); ax.set_title("Marketing cost per booking (CAC)"); st.pyplot(fig)
    st.dataframe(ch.round(1).sort_values("conv", ascending=False), use_container_width=True)
    st.markdown("**Insight:** compare *share of leads* with *share of sales* - channels where leads share >> sales share are burning budget on low-intent traffic.")
    st.subheader("Budget efficiency from campaign table")
    seg = st.radio("Segment", ["Residential", "Warehouse"], horizontal=True)
    cs = c_all[c_all.segment == seg].groupby("channel").agg(spend=("spend", "sum"), conv=("conversions_reported", "sum"))
    cs["efficiency_index"] = (cs.conv / cs.conv.sum()) / (cs.spend / cs.spend.sum())
    fig, ax = plt.subplots(figsize=(8, 3.8)); s = cs.efficiency_index.sort_values()
    s.plot.barh(ax=ax, color=[GREEN if v >= 1 else RED for v in s]); ax.axvline(1, color="black"); ax.set_xlabel("share of sales / share of spend (>1 = efficient)"); ax.set_ylabel(""); st.pyplot(fig)

# ---------------------------------------------------------------- TAB 3
with tab3:
    st.markdown("Pick a factor to see how conversion changes across its levels (descriptive) and whether the difference is statistically significant (diagnostic).")
    opts = {"Response speed": "response_bucket", "Site visits": "site_visits", "Budget vs price": "affordability_band", "Loan pre-approved": "loan_preapproved",
            "Festive season": "festive_season", "Local lead (<=5 km)": "local_lead", "Unit type": "unit_type", "Age group": "age_group", "Income quartile": "income_band", "Occupation": "occupation", "Project": "project_name"}
    pick = st.selectbox("Factor", list(opts)); col = opts[pick]
    g = r.groupby(col, observed=True).agg(leads=("lead_id", "count"), conv=("converted", "mean")); g["conv"] *= 100
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(g.index.astype(str), g.conv, color=NAVY); ax.axhline(r.converted.mean() * 100, color=RED, ls="--")
    for i, v in enumerate(g.conv): ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Conversion %"); ax.tick_params(axis="x", rotation=20); st.pyplot(fig)
    ct = pd.crosstab(r[col], r.converted)
    if ct.shape[0] > 1:
        chi2, p, _, _ = stats.chi2_contingency(ct); v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        st.info(f"Chi-square p-value = {p:.4g}  |  Cramer's V = {v:.2f}  ->  " + ("statistically significant relationship" if p < .05 else "no significant relationship"))
    st.dataframe(g.round(1), use_container_width=True)
    st.subheader("Meta Ads drill-down (largest lead source, weak conversion)")
    r2 = r.assign(group=np.where(r.channel == "Meta Ads", "Meta Ads", "Other channels"))
    st.dataframe(r2.groupby("group").agg(leads=("lead_id", "count"), conv_pct=("converted", lambda x: x.mean() * 100), median_response_hrs=("response_time_hrs", "median"),
                 avg_site_visits=("site_visits", "mean"), median_budget_ratio=("budget_to_price_ratio", "median"), loan_share=("loan_preapproved", "mean")).round(2))

# ---------------------------------------------------------------- TAB 4
with tab4:
    wc = w.groupby("channel").agg(leads=("lead_id", "count"), leases=("converted", "sum"), conv=("converted", "mean"), cost=("marketing_cost", "sum"), value=("total_lease_value", "sum"))
    wc["conv"] *= 100; wc["CAC"] = wc.cost / wc.leases.replace(0, np.nan); wc["lease_value_per_cost"] = wc.value / wc.cost
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4)); bar(ax, wc.conv.sort_values(), NAVY, ref=w.converted.mean() * 100); ax.set_title("Lease conversion % by channel"); st.pyplot(fig)
    with b:
        t = w.groupby("client_type").converted.mean().mul(100).sort_values(); fig, ax = plt.subplots(figsize=(6, 4)); bar(ax, t, ORANGE, ref=w.converted.mean() * 100); ax.set_title("Conversion % by client type"); st.pyplot(fig)
    wf = st.selectbox("Warehouse factor", ["proposal_sent", "decision_maker_contacted", "response_bucket", "size_band", "lease_term_years", "project_name"])
    g = w.groupby(wf, observed=True).converted.mean().mul(100)
    fig, ax = plt.subplots(figsize=(8, 3.6)); ax.bar(g.index.astype(str), g.values, color=NAVY); ax.axhline(w.converted.mean() * 100, color=RED, ls="--"); ax.set_ylabel("Conversion %")
    for i, v in enumerate(g.values): ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig); st.dataframe(wc.round(1).sort_values("conv", ascending=False), use_container_width=True)

# ---------------------------------------------------------------- TAB 5
with tab5:
    seg = st.radio("Dataset", ["Residential", "Warehouse"], horizontal=True, key="corr")
    if seg == "Residential":
        cols = ["age", "annual_income", "budget_to_price_ratio", "response_time_hrs", "distance_from_site_km", "follow_up_calls", "site_visits", "loan_preapproved", "festive_season", "home_loan_rate_pct", "marketing_cost", "converted"]; d = r
    else:
        cols = ["area_required_sqft", "lease_term_years", "quoted_rent_per_sqft", "response_time_hrs", "rent_gap_pct", "site_visits", "negotiation_rounds", "decision_maker_contacted", "proposal_sent", "marketing_cost", "converted"]; d = w
    method = st.selectbox("Method", ["spearman", "pearson"]); cm = d[cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(9, 7)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot_kws={"size": 8}, ax=ax); st.pyplot(fig)
    x = st.selectbox("Scatter X", cols[:-1]); y = st.selectbox("Scatter Y", cols[:-1], index=min(3, len(cols) - 2))
    fig, ax = plt.subplots(figsize=(7, 4)); sns.scatterplot(data=d.sample(min(len(d), 1500), random_state=1), x=x, y=y, hue="converted", alpha=.55, palette=[GREY if False else "#9AA5B1", ORANGE], ax=ax); st.pyplot(fig)

# ---------------------------------------------------------------- TAB 6
with tab6:
    st.subheader("Data-quality log (Step 2)")
    try: st.dataframe(pd.read_csv("tables/cleaning_log.csv"), use_container_width=True)
    except Exception: st.write("Run 02_data_cleaning_transformation.py first.")
    st.subheader("Key insights and recommended actions")
    st.markdown("""
1. **Respond within 2 hours** - conversion falls from ~18% (<2h) to ~2% (>24h).
2. **Push every serious lead to a site visit** - conversion rises from ~4% (0 visits) to ~23% (3 visits).
3. **Shift budget** from Property Portals / Hoardings / Meta to Referral, Walk-in conversion drives, WhatsApp nurture and Brokers.
4. **Qualify budget and loan status early** - budget surplus leads convert ~3x those under budget; loan pre-approval adds ~6.5 points.
5. **Concentrate launches and offers in Oct-Nov** - festive conversion is ~17% vs ~12% otherwise.
6. **Warehouses:** reach the decision-maker, send a proposal quickly and keep quoted rent within +/-5% of market.
""")
