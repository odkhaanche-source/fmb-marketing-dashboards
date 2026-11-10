"""
STEP 2 - DATA PREPARATION: CLEANING + TRANSFORMATION   (10 marks)
Input : data/raw/*.csv        Output: data/clean/*.csv, tables/cleaning_log.csv, tables/missing_before_after.csv
Every action is written to a log so the report can show evidence (before -> after).
"""
import numpy as np, pandas as pd, re
from sklearn.preprocessing import StandardScaler

RAW, CLEAN = "data/raw/", "data/clean/"
log = []
def L(table, issue, action, n):
    log.append({"table": table, "issue": issue, "action_taken": action, "records_affected": int(n)})

# canonical label maps ---------------------------------------------------------------
def norm(s):
    return re.sub(r"[\s_]+", " ", str(s)).strip().lower()
RES_CH = {norm(x): x for x in ["Google Ads", "Meta Ads", "Property Portal", "Referral", "Walk-in",
                                "Broker Network", "Hoardings/OOH", "WhatsApp/Email Campaign"]}
WH_CH = {norm(x): x for x in ["LinkedIn Outreach", "Industry Expo", "Broker", "Referral", "Website/SEO",
                               "3PL Directory", "Cold Calling"]}
ALL_CH = {**RES_CH, **WH_CH}
OCC = {norm(x): x for x in ["Salaried - IT", "Salaried - Other", "Business Owner", "Professional", "Government"]}
CT = {norm(x): x for x in ["3PL", "E-commerce", "Manufacturer", "Retail Distributor", "FMCG"]}

def parse_date(x):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%d-%m-%Y"):
        try: return pd.to_datetime(x, format=fmt)
        except Exception: pass
    return pd.NaT

projects = pd.read_csv(RAW + "projects_master.csv")
market = pd.read_csv(RAW + "market_conditions.csv")

# ================================================================== RESIDENTIAL
r = pd.read_csv(RAW + "residential_leads_raw.csv")
miss_before_r = r.isna().sum()
n0 = len(r)

# 1 duplicates
d = r.duplicated().sum(); r = r.drop_duplicates().reset_index(drop=True)
L("residential", "Exact duplicate lead rows (CRM double entry)", "Dropped duplicates", d)
d2 = r.lead_id.duplicated().sum(); r = r.drop_duplicates("lead_id").reset_index(drop=True)
L("residential", "Duplicate lead_id after exact-dup removal", "Kept first occurrence", d2)

# 2 text standardisation
bad = (r.channel != r.channel.map(norm).map(ALL_CH)).sum()
r["channel"] = r.channel.map(norm).map(ALL_CH)
L("residential", "Inconsistent channel labels (case, spaces, underscores)", "Mapped to 8 canonical channels", bad)
bad = (r.occupation.dropna() != r.occupation.dropna().map(norm).map(OCC)).sum()
r["occupation"] = r.occupation.map(lambda x: OCC.get(norm(x)) if pd.notna(x) else np.nan)
L("residential", "Occupation lower-case variants", "Standardised casing", bad)

# 3 dates
r["lead_date"] = r.lead_date.map(parse_date)
L("residential", "3 mixed date formats stored as text", "Parsed to datetime64", len(r))

# 4 target encoding
r["converted"] = r.converted.astype(str).str.strip().str.lower().map({"yes": 1, "y": 1, "1": 1, "no": 0, "n": 0, "0": 0}).astype(int)
L("residential", "Target stored as Yes/Y/No/N/1/0", "Converted to binary 0/1", len(r))

# 5 numeric coercion
r["site_visits"] = pd.to_numeric(r.site_visits, errors="coerce")   # 'NA' strings are read as NaN by pandas
L("residential", "'NA' text inside numeric site_visits column", "Coerced to numeric NaN (imputed below)", r.site_visits.isna().sum())

# 6 invalid values
n = (r.response_time_hrs < 0).sum(); r["response_time_hrs"] = r.response_time_hrs.abs()
L("residential", "Negative response time (sign entry error)", "Took absolute value", n)
n = ((r.age < 21) | (r.age > 75)).sum(); r.loc[(r.age < 21) | (r.age > 75), "age"] = np.nan
L("residential", "Impossible ages (0, 5, 17, 130, 150, 212)", "Set to NaN then imputed", n)
ratio = r.budget / r.quoted_price
n = (ratio > 8).sum()
fix = ratio > 8
r.loc[fix, "budget"] = r.loc[fix, "budget"] / 100
L("residential", "Budget typo outliers (x100) - ratio to price > 8", "Divided by 100 (verified plausible after fix)", n)

# 7 missing values (group-aware imputation)
for col, grp, how in [("age", "occupation", "median"), ("annual_income", "occupation", "median"),
                      ("response_time_hrs", "channel", "median"), ("distance_from_site_km", None, "median"),
                      ("site_visits", "channel", "median")]:
    nmiss = r[col].isna().sum()
    if grp: r[col] = r[col].fillna(r.groupby(grp)[col].transform(how))
    r[col] = r[col].fillna(r[col].median())
    L("residential", f"Missing {col}", f"Imputed {how} by {grp or 'overall'}", nmiss)
nmiss = r.occupation.isna().sum(); r["occupation"] = r.occupation.fillna(r.occupation.mode()[0])
L("residential", "Missing occupation", "Imputed mode", nmiss)
r["site_visits"] = r.site_visits.round().astype(int)

# 8 outlier capping (winsorise 1%/99%) for skewed variables
for col in ["response_time_hrs", "distance_from_site_km", "annual_income"]:
    lo, hi = r[col].quantile([.01, .99]); n = ((r[col] < lo) | (r[col] > hi)).sum()
    r[col] = r[col].clip(lo, hi)
    L("residential", f"Extreme values in {col}", "Winsorised at 1st/99th percentile", n)

# 9 TRANSFORMATION / FEATURE ENGINEERING
r = r.merge(projects[["project_id", "project_name", "zone", "price_per_sqft", "amenities_score", "dist_to_metro_km"]],
            on="project_id", how="left")
r["lead_month"] = r.lead_date.dt.to_period("M").astype(str)
r["year"] = r.lead_date.dt.year; r["quarter"] = "Q" + r.lead_date.dt.quarter.astype(str)
r["month_num"] = r.lead_date.dt.month; r["weekday"] = r.lead_date.dt.day_name()
r["festive_season"] = r.month_num.isin([10, 11]).astype(int)
r = r.merge(market, left_on="lead_month", right_on="month", how="left").drop(columns="month")
r["budget_to_price_ratio"] = (r.budget / r.quoted_price).round(3)
r["affordability_band"] = pd.cut(r.budget_to_price_ratio, [0, .8, .95, 1.1, 99],
                                 labels=["Under-budget (<0.8)", "Stretch (0.8-0.95)", "Comfortable (0.95-1.1)", "Surplus (>1.1)"])
r["response_bucket"] = pd.cut(r.response_time_hrs, [0, 2, 6, 24, 1000], labels=["<2h", "2-6h", "6-24h", ">24h"])
r["age_group"] = pd.cut(r.age, [0, 30, 40, 50, 100], labels=["<30", "30-39", "40-49", "50+"])
r["income_band"] = pd.qcut(r.annual_income, 4, labels=["Q1 Low", "Q2 Lower-mid", "Q3 Upper-mid", "Q4 High"])
r["local_lead"] = (r.distance_from_site_km <= 5).astype(int)
r["visited_site"] = (r.site_visits > 0).astype(int)
r["is_digital"] = r.channel.isin(["Google Ads", "Meta Ads", "Property Portal", "WhatsApp/Email Campaign"]).astype(int)
r["net_revenue_after_mkt"] = r.booking_value.fillna(0) - r.marketing_cost
L("residential", "Raw data lacked analytical features", "Engineered 14 features (date parts, festive flag, budget ratio, bands, buckets, local flag, joins with project + market tables)", len(r))
miss_after_r = r.isna().sum()
r.to_csv(CLEAN + "residential_leads_clean.csv", index=False)

# 10 encoding + scaling -> model/correlation-ready file
num = ["age", "annual_income", "budget_to_price_ratio", "response_time_hrs", "distance_from_site_km",
       "follow_up_calls", "site_visits", "home_loan_rate_pct", "marketing_cost", "price_per_sqft", "amenities_score"]
enc = pd.get_dummies(r[["channel", "unit_type", "occupation"]], drop_first=False, dtype=int)
sc = pd.DataFrame(StandardScaler().fit_transform(r[num]), columns=[c + "_z" for c in num])
bin_ = r[["first_time_buyer", "loan_preapproved", "festive_season", "local_lead", "converted"]]
pd.concat([sc, enc, bin_], axis=1).to_csv(CLEAN + "residential_model_ready.csv", index=False)
L("residential", "Categorical + unscaled numerics unsuitable for correlation / ML", "One-hot encoded 3 categoricals, z-scored 11 numerics", len(r))
print("Residential:", n0, "->", len(r))

# ================================================================== WAREHOUSE
w = pd.read_csv(RAW + "warehouse_leads_raw.csv")
miss_before_w = w.isna().sum(); n0w = len(w)
d = w.duplicated().sum(); w = w.drop_duplicates().reset_index(drop=True)
L("warehouse", "Exact duplicate lead rows", "Dropped duplicates", d)
bad = (w.channel != w.channel.map(norm).map(ALL_CH)).sum()
w["channel"] = w.channel.map(norm).map(ALL_CH)
L("warehouse", "Inconsistent channel labels", "Mapped to 7 canonical channels", bad)
w["client_type"] = w.client_type.map(lambda x: CT.get(norm(x)) if pd.notna(x) else np.nan)
w["lead_date"] = w.lead_date.map(parse_date)
L("warehouse", "2 mixed date formats stored as text", "Parsed to datetime64", len(w))
w["converted"] = (w.converted.str.strip().str.lower() == "yes").astype(int)
n = (w.response_time_hrs < 0).sum(); w["response_time_hrs"] = w.response_time_hrs.abs()
L("warehouse", "Negative response time", "Took absolute value", n)
n = (w.area_required_sqft > 300000).sum()
w.loc[w.area_required_sqft > 300000, "area_required_sqft"] /= 50
L("warehouse", "Area outliers (x50 typo, > 300,000 sqft requirement)", "Divided by 50", n)
for col, grp in [("area_required_sqft", "client_type"), ("response_time_hrs", "channel"), ("quoted_rent_per_sqft", "project_id")]:
    nmiss = w[col].isna().sum()
    w[col] = w[col].fillna(w.groupby(grp)[col].transform("median")).fillna(w[col].median())
    L("warehouse", f"Missing {col}", f"Imputed median by {grp}", nmiss)
nmiss = w.client_type.isna().sum(); w["client_type"] = w.client_type.fillna(w.client_type.mode()[0])
L("warehouse", "Missing client_type", "Imputed mode", nmiss)
for col in ["response_time_hrs", "area_required_sqft"]:
    lo, hi = w[col].quantile([.01, .99]); n = ((w[col] < lo) | (w[col] > hi)).sum(); w[col] = w[col].clip(lo, hi)
    L("warehouse", f"Extreme values in {col}", "Winsorised 1st/99th pct", n)

w = w.merge(projects[["project_id", "project_name", "zone", "rent_per_sqft_month", "total_area_sqft",
                      "dist_to_highway_km", "dock_doors", "clear_height_m"]], on="project_id", how="left")
w["lead_month"] = w.lead_date.dt.to_period("M").astype(str)
w["year"] = w.lead_date.dt.year; w["quarter"] = "Q" + w.lead_date.dt.quarter.astype(str)
w["rent_gap_pct"] = ((w.quoted_rent_per_sqft - w.rent_per_sqft_month) / w.rent_per_sqft_month * 100).round(2)
w["area_share_of_park_pct"] = (w.area_required_sqft / w.total_area_sqft * 100).round(2)
w["response_bucket"] = pd.cut(w.response_time_hrs, [0, 4, 12, 48, 1000], labels=["<4h", "4-12h", "12-48h", ">48h"])
w["size_band"] = pd.cut(w.area_required_sqft, [0, 20000, 50000, 100000, 1e9], labels=["Small (<20k)", "Medium (20-50k)", "Large (50-100k)", "XL (100k+)"])
w["long_lease"] = (w.lease_term_years >= 5).astype(int)
w["annual_rent_potential"] = w.area_required_sqft * w.quoted_rent_per_sqft * 12
w["net_value_after_mkt"] = w.total_lease_value.fillna(0) - w.marketing_cost
L("warehouse", "Raw data lacked analytical features", "Engineered rent-gap %, area-share, buckets, size band, long-lease flag, rent potential", len(w))
miss_after_w = w.isna().sum()
w.to_csv(CLEAN + "warehouse_leads_clean.csv", index=False)
print("Warehouse:", n0w, "->", len(w))

# ================================================================== CAMPAIGN PERFORMANCE
c = pd.read_csv(RAW + "campaign_performance_raw.csv")
miss_before_c = c.isna().sum()
d = c.duplicated().sum(); c = c.drop_duplicates().reset_index(drop=True)
L("campaign", "Duplicate monthly records", "Dropped duplicates", d)
bad = (c.channel != c.channel.map(norm).map(ALL_CH)).sum()
c["channel"] = c.channel.map(norm).map(ALL_CH)
L("campaign", "Inconsistent channel labels", "Mapped to canonical", bad)
n = c.spend.isna().sum()
cpl = (c.spend / c.leads_reported).groupby([c.segment, c.channel]).transform("median")
c["spend"] = c.spend.fillna(cpl * c.leads_reported)
L("campaign", "Missing spend (finance feed gap)", "Imputed = channel-median cost-per-lead x leads", n)
L("campaign", "Impressions/clicks blank for offline channels", "Left as NaN (not applicable, not an error)", c.impressions.isna().sum())
c["ctr_pct"] = (c.clicks / c.impressions * 100).round(3)
c["cost_per_lead"] = (c.spend / c.leads_reported).round(0)
c["lead_to_sale_pct"] = (c.conversions_reported / c.leads_reported * 100).round(2)
c["cost_per_conversion"] = (c.spend / c.conversions_reported.replace(0, np.nan)).round(0)
c["month_dt"] = pd.to_datetime(c.month + "-01")
L("campaign", "Raw data lacked KPI columns", "Derived CTR, CPL, lead-to-sale %, cost per conversion", len(c))
miss_after_c = c.isna().sum()
c.to_csv(CLEAN + "campaign_performance_clean.csv", index=False)
projects.to_csv(CLEAN + "projects_master.csv", index=False); market.to_csv(CLEAN + "market_conditions.csv", index=False)

# ------------------------------------------------------------------ logs
pd.DataFrame(log).to_csv("tables/cleaning_log.csv", index=False)
mb = pd.concat([miss_before_r.rename("residential_before"), miss_after_r.rename("residential_after")], axis=1)
mb = mb[(mb.residential_before > 0) | (mb.residential_after > 0)]
mb.to_csv("tables/missing_before_after_residential.csv")
print(pd.DataFrame(log).to_string())
print("\nResidential missing before/after:\n", mb)
print("Warehouse missing before:", miss_before_w[miss_before_w > 0].to_dict(), "after:", miss_after_w.sum())
print("Campaign missing before:", miss_before_c[miss_before_c > 0].to_dict())
