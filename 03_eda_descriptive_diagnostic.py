"""
STEP 3 - DESCRIPTIVE + DIAGNOSTIC ANALYTICS (EDA)
Descriptive : WHAT happened?  (KPIs, distributions, trends, channel/project performance)
Diagnostic  : WHY did it happen? (correlation, hypothesis tests, drill-downs, driver model)
Outputs     : figures/*.png, tables/*.csv, tables/results.json
"""
import json, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.95)
NAVY, ORANGE, GREY, GREEN, RED = "#1F3A5F", "#E07B39", "#8A94A6", "#2E8B57", "#C0392B"
C = "data/clean/"; F = "figures/"; T = "tables/"
R = {}   # results store

r = pd.read_csv(C + "residential_leads_clean.csv", parse_dates=["lead_date"])
w = pd.read_csv(C + "warehouse_leads_clean.csv", parse_dates=["lead_date"])
c = pd.read_csv(C + "campaign_performance_clean.csv", parse_dates=["month_dt"])
def save(name): plt.tight_layout(); plt.savefig(F + name, dpi=130, bbox_inches="tight"); plt.close()
def pct(x): return round(float(x) * 100, 2)

# ============================================================ A. DESCRIPTIVE - KPIs
R["res_leads"] = len(r); R["res_sales"] = int(r.converted.sum()); R["res_conv"] = pct(r.converted.mean())
R["res_mkt_cost"] = float(r.marketing_cost.sum()); R["res_revenue"] = float(r.booking_value.sum())
R["res_cac"] = round(r.marketing_cost.sum() / r.converted.sum(), 0)
R["res_avg_days_close"] = round(r.days_to_close.mean(), 1); R["res_avg_ticket"] = round(r.booking_value.mean(), 0)
R["res_rev_per_cost"] = round(r.booking_value.sum() / r.marketing_cost.sum(), 1)
R["wh_leads"] = len(w); R["wh_leases"] = int(w.converted.sum()); R["wh_conv"] = pct(w.converted.mean())
R["wh_mkt_cost"] = float(w.marketing_cost.sum()); R["wh_lease_value"] = float(w.total_lease_value.sum())
R["wh_cac"] = round(w.marketing_cost.sum() / w.converted.sum(), 0)
R["wh_rev_per_cost"] = round(w.total_lease_value.sum() / w.marketing_cost.sum(), 1)
R["wh_avg_lease_value"] = round(w.total_lease_value.mean(), 0)
desc_cols = ["age", "annual_income", "budget", "quoted_price", "budget_to_price_ratio", "response_time_hrs",
             "distance_from_site_km", "follow_up_calls", "site_visits", "marketing_cost"]
r[desc_cols].describe().T.round(2).to_csv(T + "descriptive_stats_residential.csv")
w[["area_required_sqft", "lease_term_years", "quoted_rent_per_sqft", "response_time_hrs", "rent_gap_pct",
   "site_visits", "negotiation_rounds", "marketing_cost"]].describe().T.round(2).to_csv(T + "descriptive_stats_warehouse.csv")

# Fig 1 distributions
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for a, (col, t, col_c) in zip(ax.ravel(), [("response_time_hrs", "Response time (hrs)", NAVY), ("budget_to_price_ratio", "Budget ÷ flat price", ORANGE),
        ("annual_income", "Annual income", NAVY), ("distance_from_site_km", "Distance from site (km)", ORANGE),
        ("site_visits", "Site visits per lead", NAVY), ("age", "Age", ORANGE)]):
    sns.histplot(r[col], bins=30 if col != "site_visits" else range(0, 8), ax=a, color=col_c, edgecolor="white")
    a.axvline(r[col].median(), color=RED, ls="--", lw=1.3); a.set_title(f"{t}\nmedian = {r[col].median():,.1f}", fontsize=10); a.set_xlabel("")
fig.suptitle("Fig 1 - Distribution of key buyer-lead variables (residential)", fontweight="bold"); save("fig01_distributions.png")

# Fig 2 channel volume + conversion
ch = r.groupby("channel").agg(leads=("lead_id", "count"), sales=("converted", "sum"), conv=("converted", "mean"),
                              cost=("marketing_cost", "sum"), revenue=("booking_value", "sum")).reset_index()
ch["cac"] = ch.cost / ch.sales; ch["cpl"] = ch.cost / ch.leads; ch["rev_per_cost"] = ch.revenue / ch.cost
ch["share_leads_pct"] = ch.leads / ch.leads.sum() * 100; ch["share_sales_pct"] = ch.sales / ch.sales.sum() * 100
ch["conv"] = ch.conv * 100; ch = ch.sort_values("conv", ascending=False); ch.round(2).to_csv(T + "channel_performance_residential.csv", index=False)
R["channel_res"] = ch.round(2).to_dict("records")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=ch.sort_values("leads", ascending=False), y="channel", x="leads", color=NAVY, ax=ax[0]); ax[0].set_title("Lead volume by channel"); ax[0].set_ylabel("")
for i, v in enumerate(ch.sort_values("leads", ascending=False).leads): ax[0].text(v + 5, i, int(v), va="center", fontsize=9)
sns.barplot(data=ch, y="channel", x="conv", color=ORANGE, ax=ax[1]); ax[1].axvline(r.converted.mean() * 100, color=RED, ls="--"); ax[1].set_title("Lead-to-sale conversion % (dashed = overall)"); ax[1].set_ylabel("")
for i, v in enumerate(ch.conv): ax[1].text(v + .2, i, f"{v:.1f}%", va="center", fontsize=9)
fig.suptitle("Fig 2 - Volume is not quality: which channels bring leads vs which bring buyers", fontweight="bold"); save("fig02_channel_volume_conversion.png")

# Fig 3 channel economics (CAC vs conv, bubble = leads)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(ch.conv, ch.cac / 1000, s=ch.leads * 1.6, color=NAVY, alpha=.65, edgecolor="white")
for _, x in ch.iterrows(): ax.annotate(x.channel, (x.conv, x.cac / 1000), xytext=(6, 6), textcoords="offset points", fontsize=9)
ax.set_xlabel("Lead-to-sale conversion (%)"); ax.set_ylabel("Marketing cost per sale (CAC, '000 currency units)")
ax.set_title("Fig 3 - Channel economics: conversion vs cost per sale (bubble = lead volume)", fontweight="bold"); save("fig03_channel_cac_bubble.png")

# Fig 4 monthly trend
m = r.groupby("lead_month").agg(leads=("lead_id", "count"), conv=("converted", "mean"), rate=("home_loan_rate_pct", "first")).reset_index()
m["conv"] *= 100; m.to_csv(T + "monthly_trend_residential.csv", index=False)
fig, ax = plt.subplots(figsize=(13, 4.8)); ax.bar(m.lead_month, m.leads, color=NAVY, alpha=.85); ax.set_ylabel("Leads")
ax2 = ax.twinx(); ax2.plot(m.lead_month, m.conv, color=ORANGE, marker="o", lw=2); ax2.set_ylabel("Conversion %"); ax2.grid(False)
ax.set_xticks(range(len(m))); ax.set_xticklabels(m.lead_month, rotation=60, ha="right", fontsize=8)
for i, mm in enumerate(m.lead_month):
    if mm[-2:] in ("10", "11"): ax.axvspan(i - .5, i + .5, color=GREEN, alpha=.15)
ax.set_title("Fig 4 - Monthly leads (bars) and conversion % (line); green = festive months", fontweight="bold"); save("fig04_monthly_trend.png")
R["festive_conv"] = pct(r[r.festive_season == 1].converted.mean()); R["nonfestive_conv"] = pct(r[r.festive_season == 0].converted.mean())
R["festive_lead_share"] = pct((r.festive_season == 1).mean())

# Fig 5 funnel
f_leads = len(r); f_vis = int((r.site_visits > 0).sum()); f_rep = int((r.site_visits >= 2).sum()); f_sale = int(r.converted.sum())
R["funnel"] = {"Leads": f_leads, "Visited site (>=1)": f_vis, "Repeat visit (>=2)": f_rep, "Booked": f_sale}
fig, ax = plt.subplots(figsize=(9, 4.5)); labs = list(R["funnel"]); vals = list(R["funnel"].values())
ax.barh(labs[::-1], vals[::-1], color=[GREEN, ORANGE, NAVY, NAVY][::-1]); 
for i, v in enumerate(vals[::-1]): ax.text(v + 30, i, f"{v:,} ({v / f_leads * 100:.1f}%)", va="center")
ax.set_xlim(0, f_leads * 1.25); ax.set_title("Fig 5 - Sales funnel (residential)", fontweight="bold"); save("fig05_funnel.png")

# ============================================================ B. DIAGNOSTIC - correlation
num_cols = ["age", "annual_income", "budget_to_price_ratio", "response_time_hrs", "distance_from_site_km", "follow_up_calls",
            "site_visits", "loan_preapproved", "first_time_buyer", "home_loan_rate_pct", "festive_season", "price_per_sqft",
            "amenities_score", "marketing_cost", "converted"]
corr = r[num_cols].corr(method="spearman"); corr.round(3).to_csv(T + "correlation_residential.csv")
fig, ax = plt.subplots(figsize=(11, 8.5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot_kws={"size": 7}, ax=ax)
ax.set_title("Fig 6 - Spearman correlation matrix (residential)", fontweight="bold"); save("fig06_corr_residential.png")
tc = corr["converted"].drop("converted").sort_values(); R["corr_target_res"] = tc.round(3).to_dict()
fig, ax = plt.subplots(figsize=(8, 5.5)); ax.barh(tc.index, tc.values, color=[RED if v < 0 else GREEN for v in tc.values])
ax.set_title("Fig 7 - Correlation of each factor with conversion", fontweight="bold"); ax.set_xlabel("Spearman ρ"); save("fig07_corr_with_target.png")

# Fig 8 response time / visits / affordability / loan
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
def rate_bar(a, col, title, order=None, color=NAVY):
    g = r.groupby(col, observed=True).converted.mean().mul(100)
    if order is not None: g = g.reindex(order)
    a.bar(g.index.astype(str), g.values, color=color); a.axhline(r.converted.mean() * 100, color=RED, ls="--", lw=1)
    for i, v in enumerate(g.values): a.text(i, v + .3, f"{v:.1f}%", ha="center", fontsize=9)
    a.set_title(title, fontsize=10.5); a.set_ylabel("Conversion %"); return g
g_resp = rate_bar(ax[0, 0], "response_bucket", "A. Speed of first response", ["<2h", "2-6h", "6-24h", ">24h"])
g_vis = rate_bar(ax[0, 1], "site_visits", "B. Number of site visits", None, ORANGE)
g_aff = rate_bar(ax[1, 0], "affordability_band", "C. Buyer budget vs flat price", ["Under-budget (<0.8)", "Stretch (0.8-0.95)", "Comfortable (0.95-1.1)", "Surplus (>1.1)"]); ax[1, 0].tick_params(axis="x", labelsize=8)
g_loan = rate_bar(ax[1, 1], "loan_preapproved", "D. Home-loan pre-approved? (0 = no, 1 = yes)", None, ORANGE)
fig.suptitle("Fig 8 - Diagnostic drill-down: what separates buyers from browsers (dashed = overall rate)", fontweight="bold"); save("fig08_drivers_res.png")
R["resp_bucket"] = g_resp.round(2).to_dict(); R["visits"] = g_vis.round(2).to_dict(); R["afford"] = g_aff.round(2).to_dict(); R["loan"] = g_loan.round(2).to_dict()

# Fig 9 project + zone
pj = r.groupby(["project_name", "zone"]).agg(leads=("lead_id", "count"), conv=("converted", "mean"), pps=("price_per_sqft", "first"),
                                             cac=("marketing_cost", "sum"), sales=("converted", "sum")).reset_index()
pj["cac"] = pj.cac / pj.sales; pj["conv"] *= 100; pj = pj.sort_values("conv", ascending=False); pj.round(2).to_csv(T + "project_performance.csv", index=False)
R["projects"] = pj.round(2).to_dict("records")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
sns.barplot(data=pj, y="project_name", x="conv", color=NAVY, ax=ax[0]); ax[0].set_title("Conversion % by project"); ax[0].set_ylabel("")
ax[1].scatter(pj.pps, pj.conv, s=pj.leads / 2.5, color=ORANGE, alpha=.8, edgecolor="white")
for _, x in pj.iterrows(): ax[1].annotate(x.project_name, (x.pps, x.conv), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax[1].set_xlabel("Price per sqft"); ax[1].set_ylabel("Conversion %"); ax[1].set_title("Price positioning vs conversion")
fig.suptitle("Fig 9 - Project-level performance", fontweight="bold"); save("fig09_projects.png")

# Fig 10 interest rate vs monthly conversion
rho, p = stats.spearmanr(m.rate, m.conv); R["rate_monthly_rho"] = round(rho, 3); R["rate_monthly_p"] = round(p, 4)
fig, ax = plt.subplots(figsize=(7, 5)); sns.regplot(data=m, x="rate", y="conv", color=NAVY, line_kws={"color": RED}, ax=ax)
ax.set_xlabel("Home-loan interest rate (%)"); ax.set_ylabel("Monthly conversion %"); ax.set_title(f"Fig 10 - Interest rate vs conversion (ρ={rho:.2f}, p={p:.3f})", fontweight="bold"); save("fig10_rate_vs_conv.png")

# Fig 11 Meta Ads drill-down (why is the biggest source weak?)
r["is_meta"] = np.where(r.channel == "Meta Ads", "Meta Ads", "All other channels")
dd = r.groupby("is_meta").agg(response_hrs=("response_time_hrs", "median"), visits=("site_visits", "mean"), budget_ratio=("budget_to_price_ratio", "median"),
                              loan=("loan_preapproved", "mean"), dist=("distance_from_site_km", "median"), conv=("converted", "mean")).round(3)
dd.to_csv(T + "meta_drilldown.csv"); R["meta_drill"] = dd.reset_index().to_dict("records")
fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
for a, (col, t) in zip(ax, [("response_hrs", "Median response (hrs)"), ("visits", "Avg site visits"), ("budget_ratio", "Median budget ÷ price"), ("loan", "Loan pre-approved share")]):
    a.bar(dd.index, dd[col], color=[ORANGE, GREY]); a.set_title(t, fontsize=10); a.tick_params(axis="x", labelsize=8)
    for i, v in enumerate(dd[col]): a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("Fig 11 - Root-cause drill-down: why does Meta Ads (largest lead source) convert poorly?", fontweight="bold"); save("fig11_meta_drilldown.png")

# ============================================================ C. HYPOTHESIS TESTS (residential)
tests = []
def chi(df, col, y, label, H):
    ct = pd.crosstab(df[col], df[y]); chi2, p, dof, _ = stats.chi2_contingency(ct); n = ct.values.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))); tests.append([H, label, "Chi-square", round(chi2, 2), p, f"Cramer's V={v:.2f}", "Reject H0" if p < .05 else "Fail to reject"])
def mw(df, col, y, label, H):
    a, b = df[df[y] == 1][col], df[df[y] == 0][col]; u, p = stats.mannwhitneyu(a, b); d = a.median() - b.median()
    tests.append([H, label, "Mann-Whitney U", round(u, 0), p, f"median buyers={a.median():.2f} vs non={b.median():.2f}", "Reject H0" if p < .05 else "Fail to reject"])
chi(r, "channel", "converted", "Conversion differs by channel", "H1"); mw(r, "response_time_hrs", "converted", "Buyers got faster first response", "H2")
chi(r.assign(v=(r.site_visits > 0).astype(int)), "v", "converted", "Site visit vs conversion", "H3"); chi(r, "loan_preapproved", "converted", "Loan pre-approval vs conversion", "H4")
mw(r, "budget_to_price_ratio", "converted", "Higher budget-to-price ratio among buyers", "H5"); chi(r, "festive_season", "converted", "Festive season vs conversion", "H6")
chi(r, "local_lead", "converted", "Local lead (<=5km) vs conversion", "H7"); chi(r, "first_time_buyer", "converted", "First-time buyer vs conversion", "H8")
chi(r, "unit_type", "converted", "Unit type (BHK) vs conversion", "H9"); mw(r, "age", "converted", "Age difference buyers vs non-buyers", "H10")
mw(r, "annual_income", "converted", "Income difference buyers vs non-buyers", "H11")
# ANOVA-type for days_to_close across channels
g = [x.days_to_close.dropna() for _, x in r[r.converted == 1].groupby("channel")]; h, p = stats.kruskal(*g)
tests.append(["H12", "Days-to-close differs by channel", "Kruskal-Wallis", round(h, 2), p, "", "Reject H0" if p < .05 else "Fail to reject"])
tdf = pd.DataFrame(tests, columns=["id", "hypothesis (H1 = effect exists)", "test", "statistic", "p_value", "effect_size / note", "decision (alpha=0.05)"])
tdf["p_value"] = tdf.p_value.map(lambda x: f"{x:.2e}" if x < .001 else f"{x:.3f}"); tdf.to_csv(T + "hypothesis_tests_residential.csv", index=False); R["tests_res"] = tdf.to_dict("records")

# ============================================================ D. DRIVER MODEL (diagnostic) + accuracy validation
feat = ["response_time_hrs", "site_visits", "budget_to_price_ratio", "loan_preapproved", "follow_up_calls", "festive_season",
        "home_loan_rate_pct", "distance_from_site_km", "first_time_buyer", "amenities_score", "age", "annual_income"]
X = r[feat].copy(); X["log_response"] = np.log1p(X.response_time_hrs); X = X.drop(columns="response_time_hrs"); X["annual_income"] /= 1e6
X = pd.concat([X, pd.get_dummies(r.channel, drop_first=False, dtype=int).drop(columns="Google Ads")], axis=1); y = r.converted
Xs = (X - X.mean()) / X.std(); Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=.25, random_state=42, stratify=y)
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr); pr = lr.predict_proba(Xte)[:, 1]; yhat = (pr >= .5).astype(int)
auc = roc_auc_score(yte, pr); R["model"] = {"auc": round(auc, 3), "accuracy": round(accuracy_score(yte, yhat), 3), "precision": round(precision_score(yte, yhat), 3),
                                              "recall": round(recall_score(yte, yhat), 3), "confusion": confusion_matrix(yte, yhat).tolist(), "n_train": len(Xtr), "n_test": len(Xte)}
coef = pd.DataFrame({"factor": Xs.columns, "std_coef": lr.coef_[0]}); coef["odds_ratio_per_1SD"] = np.exp(coef.std_coef); coef = coef.sort_values("std_coef")
coef.round(3).to_csv(T + "driver_model_coefficients.csv", index=False); R["drivers"] = coef.round(3).to_dict("records")
fig, ax = plt.subplots(figsize=(9, 7.5)); ax.barh(coef.factor, coef.std_coef, color=[RED if v < 0 else GREEN for v in coef.std_coef])
ax.set_title(f"Fig 12 - Driver analysis (logistic regression, standardised coefficients)\nHold-out AUC = {auc:.2f}", fontweight="bold"); ax.set_xlabel("Effect on log-odds of booking (per 1 SD); channels vs Google Ads"); save("fig12_driver_model.png")

# ============================================================ E. WAREHOUSE
wc = w.groupby("channel").agg(leads=("lead_id", "count"), leases=("converted", "sum"), conv=("converted", "mean"), cost=("marketing_cost", "sum"),
                              value=("total_lease_value", "sum"), resp=("response_time_hrs", "median")).reset_index()
wc["cac"] = wc.cost / wc.leases; wc["value_per_cost"] = wc.value / wc.cost; wc["conv"] *= 100; wc = wc.sort_values("conv", ascending=False)
wc.round(2).to_csv(T + "channel_performance_warehouse.csv", index=False); R["channel_wh"] = wc.round(2).to_dict("records")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=wc, y="channel", x="conv", color=NAVY, ax=ax[0]); ax[0].axvline(w.converted.mean() * 100, color=RED, ls="--"); ax[0].set_title("Lease conversion % by channel"); ax[0].set_ylabel("")
for i, v in enumerate(wc.conv): ax[0].text(v + .2, i, f"{v:.1f}%", va="center", fontsize=9)
sns.barplot(data=wc.sort_values("cac"), y="channel", x="cac", color=ORANGE, ax=ax[1]); ax[1].set_title("Marketing cost per lease signed (CAC)"); ax[1].set_ylabel("")
fig.suptitle("Fig 13 - Warehouse leasing: channel conversion and cost per lease", fontweight="bold"); save("fig13_wh_channels.png")

wt = w.groupby("client_type").agg(leads=("lead_id", "count"), conv=("converted", "mean"), avg_value=("total_lease_value", "mean"), avg_area=("area_required_sqft", "mean")).reset_index()
wt["conv"] *= 100; wt = wt.sort_values("conv", ascending=False); wt.round(1).to_csv(T + "client_type_warehouse.csv", index=False); R["client_type"] = wt.round(1).to_dict("records")
w["rent_gap_band"] = pd.cut(w.rent_gap_pct, [-100, -5, 0, 5, 100], labels=["Below mkt >5%", "0 to -5%", "0 to +5%", "Above mkt >5%"])
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
def wbar(a, col, title, order=None, color=NAVY):
    g = w.groupby(col, observed=True).converted.mean().mul(100)
    if order is not None: g = g.reindex(order)
    a.bar(g.index.astype(str), g.values, color=color); a.axhline(w.converted.mean() * 100, color=RED, ls="--", lw=1)
    for i, v in enumerate(g.values): a.text(i, v + .3, f"{v:.1f}%", ha="center", fontsize=9)
    a.set_title(title, fontsize=10.5); a.set_ylabel("Conversion %"); a.tick_params(axis="x", labelsize=8); return g
w_gap = wbar(ax[0, 0], "rent_gap_band", "A. Quoted rent vs market rent", ["Below mkt >5%", "0 to -5%", "0 to +5%", "Above mkt >5%"])
w_prop = wbar(ax[0, 1], "proposal_sent", "B. Formal proposal sent? (0/1)", None, ORANGE)
w_dm = wbar(ax[1, 0], "decision_maker_contacted", "C. Decision-maker reached? (0/1)", None, ORANGE)
w_resp = wbar(ax[1, 1], "response_bucket", "D. Response speed", ["<4h", "4-12h", "12-48h", ">48h"])
fig.suptitle("Fig 14 - Warehouse diagnostic drill-down (dashed = overall conversion)", fontweight="bold"); save("fig14_wh_drivers.png")
R["w_gap"] = w_gap.round(2).to_dict(); R["w_prop"] = w_prop.round(2).to_dict(); R["w_dm"] = w_dm.round(2).to_dict(); R["w_resp"] = w_resp.round(2).to_dict()
wnum = ["area_required_sqft", "lease_term_years", "quoted_rent_per_sqft", "response_time_hrs", "rent_gap_pct", "site_visits", "negotiation_rounds",
        "decision_maker_contacted", "proposal_sent", "area_share_of_park_pct", "dist_to_highway_km", "marketing_cost", "converted"]
wcorr = w[wnum].corr(method="spearman"); wcorr.round(3).to_csv(T + "correlation_warehouse.csv"); R["corr_target_wh"] = wcorr["converted"].drop("converted").sort_values().round(3).to_dict()
fig, ax = plt.subplots(figsize=(10.5, 8)); sns.heatmap(wcorr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot_kws={"size": 7}, ax=ax)
ax.set_title("Fig 15 - Spearman correlation matrix (warehouse leads)", fontweight="bold"); save("fig15_corr_warehouse.png")
wt_ = []
def wchi(col, label, H):
    ct = pd.crosstab(w[col], w.converted); chi2, p, _, _ = stats.chi2_contingency(ct); v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    wt_.append([H, label, "Chi-square", round(chi2, 2), p, f"Cramer's V={v:.2f}", "Reject H0" if p < .05 else "Fail to reject"])
def wmw(col, label, H):
    a, b = w[w.converted == 1][col], w[w.converted == 0][col]; u, p = stats.mannwhitneyu(a, b)
    wt_.append([H, label, "Mann-Whitney U", round(u, 0), p, f"median leased={a.median():.2f} vs not={b.median():.2f}", "Reject H0" if p < .05 else "Fail to reject"])
wchi("channel", "Conversion differs by channel", "W1"); wchi("client_type", "Conversion differs by client type", "W2"); wchi("proposal_sent", "Proposal sent vs conversion", "W3")
wchi("decision_maker_contacted", "Decision-maker contact vs conversion", "W4"); wmw("rent_gap_pct", "Lower rent gap among leased", "W5"); wmw("response_time_hrs", "Faster response among leased", "W6")
wchi("long_lease", "Lease term >=5y vs conversion", "W7"); wmw("negotiation_rounds", "Negotiation rounds difference", "W8")
wdf = pd.DataFrame(wt_, columns=tdf.columns); wdf["p_value"] = wdf.p_value.map(lambda x: f"{x:.2e}" if x < .001 else f"{x:.3f}"); wdf.to_csv(T + "hypothesis_tests_warehouse.csv", index=False); R["tests_wh"] = wdf.to_dict("records")

# ============================================================ F. CAMPAIGN ROI (cross-table)
cs = c.groupby(["segment", "channel"]).agg(spend=("spend", "sum"), leads=("leads_reported", "sum"), conv=("conversions_reported", "sum"), clicks=("clicks", "sum"), imp=("impressions", "sum")).reset_index()
cs["cpl"] = cs.spend / cs.leads; cs["cost_per_conv"] = cs.spend / cs.conv; cs["lead_to_sale_pct"] = cs.conv / cs.leads * 100
cs["spend_share_pct"] = cs.spend / cs.groupby("segment").spend.transform("sum") * 100; cs["conv_share_pct"] = cs.conv / cs.groupby("segment").conv.transform("sum") * 100
cs["efficiency_index"] = cs.conv_share_pct / cs.spend_share_pct; cs.round(2).to_csv(T + "campaign_roi.csv", index=False); R["campaign"] = cs.round(2).to_dict("records")
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for a, seg in zip(ax, ["Residential", "Warehouse"]):
    d = cs[cs.segment == seg].sort_values("efficiency_index"); a.barh(d.channel, d.efficiency_index, color=[GREEN if v >= 1 else RED for v in d.efficiency_index])
    a.axvline(1, color="black", lw=1); a.set_title(f"{seg}: share of sales ÷ share of spend"); a.set_xlabel("Efficiency index (>1 = earns more than it costs)")
    for i, v in enumerate(d.efficiency_index): a.text(v + .02, i, f"{v:.2f}", va="center", fontsize=9)
fig.suptitle("Fig 16 - Marketing budget efficiency by channel (from campaign_performance table)", fontweight="bold"); save("fig16_budget_efficiency.png")

# Fig 17 income x unit type & age groups (customer profile)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, (col, t) in zip(ax, [("unit_type", "Unit type"), ("age_group", "Age group"), ("occupation", "Occupation")]):
    g = r.groupby(col, observed=True).converted.mean().mul(100).sort_values(); a.barh(g.index.astype(str), g.values, color=NAVY); a.set_title(f"Conversion % by {t}")
    for i, v in enumerate(g.values): a.text(v + .1, i, f"{v:.1f}%", va="center", fontsize=9)
fig.suptitle("Fig 17 - Customer profile: who buys? (little separation -> behaviour matters more than demographics)", fontweight="bold"); save("fig17_customer_profile.png")
R["profile"] = {col: r.groupby(col, observed=True).converted.mean().mul(100).round(2).to_dict() for col in ["unit_type", "age_group", "occupation", "income_band"]}
# Fig 18 revenue vs. cost days to close
dc = r[r.converted == 1].groupby("channel").days_to_close.median().sort_values(); R["days_to_close_by_channel"] = dc.round(1).to_dict()
R["days_visit0_vs_visit>0"] = {"no_visit": round(r[(r.converted == 1) & (r.site_visits == 0)].days_to_close.median(), 1), "visited": round(r[(r.converted == 1) & (r.site_visits > 0)].days_to_close.median(), 1)}

json.dump(R, open(T + "results.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["res_leads", "res_sales", "res_conv", "res_cac", "res_rev_per_cost", "wh_leads", "wh_leases", "wh_conv", "wh_cac", "wh_rev_per_cost",
      "festive_conv", "nonfestive_conv", "rate_monthly_rho", "rate_monthly_p", "model", "resp_bucket", "visits", "afford", "loan", "w_gap", "w_prop", "w_dm", "w_resp"]}, indent=1))
print(ch[["channel", "leads", "conv", "cac", "rev_per_cost", "share_leads_pct", "share_sales_pct"]].round(1).to_string())
print(wc[["channel", "leads", "conv", "cac", "value_per_cost"]].round(1).to_string())
print(tdf.to_string()); print(wdf.to_string()); print(coef.round(2).to_string()); print(R["meta_drill"]); print(wt)
print(cs[["segment", "channel", "efficiency_index", "cost_per_conv"]].round(2).to_string())
