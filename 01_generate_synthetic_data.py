"""
STEP 1 - SYNTHETIC DATA GENERATION
Family-Managed Business (FMB): residential flats (for sale) + warehouses (for lease)
Business objective: improve marketing ROI (lower CAC, higher lead->sale / lead->lease conversion)

Four linked tables are generated and then deliberately "dirtied" so that Step 2 (cleaning)
has real work to do:
  1. projects_master.csv          - 8 residential projects + 4 warehouse parks
  2. residential_leads_raw.csv    - flat buyer leads (Jan-2024 .. Dec-2025)
  3. warehouse_leads_raw.csv      - B2B warehouse leasing leads
  4. campaign_performance_raw.csv - monthly spend/impressions/clicks per channel
  5. market_conditions.csv        - monthly home-loan rate & festive index (external factor)

Ground-truth relationships are embedded (e.g. slow response time hurts conversion) so that
the diagnostic analysis in Step 3 can be validated against known truth.
"""
import numpy as np, pandas as pd
from scipy.optimize import brentq

rng = np.random.default_rng(42)
RAW = "data/raw/"

# ------------------------------------------------------------------ market conditions
months = pd.period_range("2024-01", "2025-12", freq="M")
n_m = len(months)
loan_rate = np.round(8.95 - np.linspace(0, 0.75, n_m) + rng.normal(0, 0.05, n_m), 2)
festive_idx = np.array([1.0, 1.0, 1.15, 1.15, 1.0, 0.85, 0.85, 0.85, 1.0, 1.4, 1.4, 1.1] * 2)
market = pd.DataFrame({"month": months.astype(str), "home_loan_rate_pct": loan_rate,
                       "festive_index": festive_idx})
market.to_csv(RAW + "market_conditions.csv", index=False)
rate_map = dict(zip(market.month, market.home_loan_rate_pct))
fest_map = dict(zip(market.month, market.festive_index))

# ------------------------------------------------------------------ projects master
res_projects = pd.DataFrame([
    ["P01", "Riverside Heights",  "Residential", "North",    7200, 180, 8.5, 2.0, 2023],
    ["P02", "Green Meadows",      "Residential", "South",    6400, 140, 7.0, 3.5, 2023],
    ["P03", "Sunrise Residency",  "Residential", "East",     5600, 220, 6.5, 5.0, 2024],
    ["P04", "Skyline Towers",     "Residential", "Central",  9800, 120, 9.0, 1.0, 2024],
    ["P05", "Lakeview Enclave",   "Residential", "West",     6800, 160, 8.0, 4.0, 2023],
    ["P06", "Palm Court",         "Residential", "Suburban", 4900, 260, 6.0, 9.0, 2024],
    ["P07", "Heritage Square",    "Residential", "Central",  8800, 100, 8.8, 1.5, 2024],
    ["P08", "Orchid Park",        "Residential", "North",    7600, 150, 7.5, 3.0, 2024],
], columns=["project_id", "project_name", "segment", "zone", "price_per_sqft", "total_units",
            "amenities_score", "dist_to_metro_km", "launch_year"])
wh_projects = pd.DataFrame([
    ["W01", "Logistics Park A",   "Warehouse", "North",  24, 250000, 2.0, 12, 11.0],
    ["W02", "Logistics Park B",   "Warehouse", "East",   21, 180000, 5.5, 10, 10.0],
    ["W03", "Warehousing Hub C",  "Warehouse", "South",  19, 120000, 8.0,  8,  9.5],
    ["W04", "Industrial Estate D","Warehouse", "West",   17,  90000, 12.0, 6,  9.0],
], columns=["project_id", "project_name", "segment", "zone", "rent_per_sqft_month", "total_area_sqft",
            "dist_to_highway_km", "dock_doors", "clear_height_m"])
projects = pd.concat([res_projects, wh_projects], ignore_index=True)
projects.to_csv(RAW + "projects_master.csv", index=False)

# ------------------------------------------------------------------ helpers
def sample_dates(n):
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    w = np.array([fest_map[str(d.to_period("M"))] for d in days]); w = w / w.sum()
    return pd.to_datetime(rng.choice(days, size=n, p=w))

def calibrate(z_wo_intercept, target):
    f = lambda b: (1 / (1 + np.exp(-(z_wo_intercept + b)))).mean() - target
    return brentq(f, -15, 5)

# ================================================================== RESIDENTIAL LEADS
N_R = 3200
res_channels = {  #  weight, base logit effect, cost-per-lead (currency units), mean response hrs
    "Google Ads":            (0.18,  0.00, 9000, 5),
    "Meta Ads":              (0.22, -0.55, 5500, 7),
    "Property Portal":       (0.18, -0.10, 8000, 9),
    "Referral":              (0.08,  1.00, 2500, 3),
    "Walk-in":               (0.08,  0.60, 1000, 1),
    "Broker Network":        (0.12,  0.20, 7000, 6),
    "Hoardings/OOH":         (0.06, -0.30, 10000, 8),
    "WhatsApp/Email Campaign":(0.08,-0.25, 1800, 4),
}
ch_names = list(res_channels)
ch = rng.choice(ch_names, N_R, p=[res_channels[c][0] for c in ch_names])
dates = sample_dates(N_R)
mkey = np.array([str(d.to_period("M")) for d in dates])
proj_ids = rng.choice(res_projects.project_id, N_R,
                      p=np.array([.14, .11, .15, .09, .12, .17, .08, .14]))
pp = res_projects.set_index("project_id")
pps = pp.loc[proj_ids, "price_per_sqft"].values
unit_type = rng.choice(["1BHK", "2BHK", "3BHK", "4BHK"], N_R, p=[.18, .44, .30, .08])
area = np.select([unit_type == "1BHK", unit_type == "2BHK", unit_type == "3BHK"],
                 [rng.normal(620, 50, N_R), rng.normal(1020, 80, N_R), rng.normal(1450, 100, N_R)],
                 rng.normal(2000, 150, N_R)).round(0)
ticket = (pps * area).round(-3)

occupation = rng.choice(["Salaried - IT", "Salaried - Other", "Business Owner", "Professional", "Government"],
                        N_R, p=[.24, .26, .22, .16, .12])
income = (rng.lognormal(np.log(1_550_000), 0.45, N_R) *
          np.where(occupation == "Business Owner", 1.25, 1) *
          np.where(occupation == "Government", 0.85, 1)).round(-3)
budget = (income * rng.uniform(3.6, 6.8, N_R)).round(-3)
budget_ratio = budget / ticket
age = np.clip(rng.normal(38, 8, N_R), 23, 68).round(0)
first_time = rng.binomial(1, 0.55, N_R)
dist = np.where(rng.random(N_R) < .45, rng.exponential(4, N_R), rng.exponential(14, N_R)).round(1)
resp = np.array([rng.exponential(res_channels[c][3]) for c in ch]).round(1)
resp = np.clip(resp, 0.2, 96)
loan = rng.binomial(1, np.where(np.isin(occupation, ["Salaried - IT", "Salaried - Other", "Government"]), .45, .25))
calls = rng.poisson(3, N_R)
visits = rng.poisson(np.clip(1.0 + 0.6 * (budget_ratio > 0.95) + 0.5 * (resp < 4) - 0.4 * (dist > 20)
                             + 0.3 * (ch == "Referral"), 0.1, None))
visits = np.clip(visits, 0, 6)
festive = np.array([fest_map[m] for m in mkey])
rate = np.array([rate_map[m] for m in mkey])
base = (np.array([res_channels[c][1] for c in ch]) + 0.65 * np.minimum(visits, 3)
        - 0.30 * np.log1p(resp) + 0.75 * loan + 1.4 * np.clip(budget_ratio - 1, -0.6, 0.5)
        + 0.12 * np.minimum(calls, 6) + 0.9 * (festive - 1) - 0.25 * (rate - 8.5) - 0.025 * dist
        + 0.10 * pp.loc[proj_ids, "amenities_score"].values / 2 - 0.15 * first_time)
b0 = calibrate(base, 0.125)
p_conv = 1 / (1 + np.exp(-(base + b0)))
conv = rng.binomial(1, p_conv)
cpl = np.array([res_channels[c][2] for c in ch])
mkt_cost = (cpl * rng.lognormal(0, 0.25, N_R)).round(0)
disc = rng.uniform(0.90, 0.99, N_R)
booking_value = np.where(conv == 1, (ticket * disc).round(-3), np.nan)
days_to_close = np.where(conv == 1, np.clip(rng.gamma(4, 12, N_R) + 20 * (visits == 0), 7, 240).round(0), np.nan)

res = pd.DataFrame({
    "lead_id": [f"RL{100000+i}" for i in range(N_R)],
    "lead_date": dates, "channel": ch, "project_id": proj_ids, "unit_type": unit_type,
    "carpet_area_sqft": area, "quoted_price": ticket, "age": age, "occupation": occupation,
    "annual_income": income, "budget": budget, "first_time_buyer": first_time,
    "distance_from_site_km": dist, "response_time_hrs": resp, "follow_up_calls": calls,
    "site_visits": visits, "loan_preapproved": loan, "marketing_cost": mkt_cost,
    "converted": conv, "booking_value": booking_value, "days_to_close": days_to_close})

# ---- dirty the raw residential data (realistic CRM problems)
r = res.copy()
r["lead_date"] = [d.strftime(rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"], p=[.7, .2, .1])) for d in r.lead_date]
def messy(label):
    v = rng.random()
    if v < .06: return label.lower()
    if v < .10: return label.upper()
    if v < .13: return " " + label + " "
    if v < .15: return label.replace(" ", "_")
    return label
r["channel"] = r.channel.map(messy)
r["occupation"] = r.occupation.map(lambda x: x.lower() if rng.random() < .05 else x)
r["converted"] = [rng.choice(["Yes", "Y"]) if c == 1 else rng.choice(["No", "N"]) for c in r.converted]
r["converted"] = [v if rng.random() > .1 else ({"Yes": "1", "Y": "1", "No": "0", "N": "0"}[v]) for v in r.converted]
for col, p in [("age", .04), ("annual_income", .06), ("occupation", .03), ("response_time_hrs", .03),
               ("distance_from_site_km", .02)]:
    r.loc[rng.random(N_R) < p, col] = np.nan
r["site_visits"] = r["site_visits"].astype(object)
r.loc[rng.random(N_R) < .015, "site_visits"] = "NA"
idx = rng.choice(N_R, 12, replace=False); r.loc[idx, "budget"] = r.loc[idx, "budget"] * 100   # typo outliers
idx = rng.choice(N_R, 15, replace=False); r.loc[idx, "response_time_hrs"] = -r.loc[idx, "response_time_hrs"]  # negative
idx = rng.choice(N_R, 6, replace=False); r.loc[idx, "age"] = [150, 5, 0, 212, 17, 130]
r = pd.concat([r, r.sample(48, random_state=1)], ignore_index=True)  # duplicate rows
r = r.sample(frac=1, random_state=3).reset_index(drop=True)
r.to_csv(RAW + "residential_leads_raw.csv", index=False)

# ================================================================== WAREHOUSE LEADS
N_W = 1100
wh_channels = {  # weight, logit effect, cost per lead, mean response hrs
    "LinkedIn Outreach":  (0.20, -0.20, 11000, 10),
    "Industry Expo":      (0.10,  0.50, 22000, 12),
    "Broker":             (0.22,  0.30, 14000, 8),
    "Referral":           (0.10,  1.00,  4000, 4),
    "Website/SEO":        (0.16, -0.30,  6000, 14),
    "3PL Directory":      (0.12,  0.15, 12000, 9),
    "Cold Calling":       (0.10, -0.60,  3500, 20),
}
wch_names = list(wh_channels)
wch = rng.choice(wch_names, N_W, p=[wh_channels[c][0] for c in wch_names])
wdates = sample_dates(N_W)
wproj = rng.choice(wh_projects.project_id, N_W, p=[.34, .28, .22, .16])
wp = wh_projects.set_index("project_id")
ctype = rng.choice(["3PL", "E-commerce", "Manufacturer", "Retail Distributor", "FMCG"], N_W,
                   p=[.38, .22, .20, .12, .08])
area_req = np.clip(rng.lognormal(np.log(38000), 0.6, N_W), 5000, 220000).round(-2)
lease_term = rng.choice([3, 5, 7, 9], N_W, p=[.25, .40, .22, .13])
market_rent = wp.loc[wproj, "rent_per_sqft_month"].values
quoted_rent = (market_rent * rng.normal(1.0, 0.07, N_W)).round(1)
rent_gap = (quoted_rent - market_rent) / market_rent
wresp = np.clip(np.array([rng.exponential(wh_channels[c][3]) for c in wch]), 0.5, 120).round(1)
decision_maker = rng.binomial(1, np.where(np.isin(wch, ["Referral", "Industry Expo", "Broker"]), .65, .35))
wvisits = np.clip(rng.poisson(0.8 + 0.7 * decision_maker + 0.4 * (wresp < 8)), 0, 5)
proposal = rng.binomial(1, np.clip(0.25 + 0.15 * wvisits + 0.2 * decision_maker, 0, .95))
nego_rounds = rng.poisson(2.2, N_W)
wfest = np.array([fest_map[str(pd.Timestamp(d).to_period('M'))] for d in wdates])
fit = 1 - np.clip(np.abs(np.log(area_req / (wp.loc[wproj, "total_area_sqft"].values * 0.18))), 0, 2) / 2
wbase = (np.array([wh_channels[c][1] for c in wch]) + 0.9 * proposal + 0.45 * np.minimum(wvisits, 3)
         + 0.6 * decision_maker - 0.25 * np.log1p(wresp) - 3.5 * rent_gap + 0.5 * (ctype == "3PL")
         + 0.4 * fit + 0.15 * (lease_term >= 5) - 0.10 * nego_rounds + 0.4 * (wfest - 1))
wb0 = calibrate(wbase, 0.15)
wconv = rng.binomial(1, 1 / (1 + np.exp(-(wbase + wb0))))
wcpl = np.array([wh_channels[c][2] for c in wch])
wcost = (wcpl * rng.lognormal(0, 0.25, N_W)).round(0)
lease_value = np.where(wconv == 1, (area_req * quoted_rent * 12 * lease_term).round(-3), np.nan)

wh = pd.DataFrame({
    "lead_id": [f"WL{200000+i}" for i in range(N_W)], "lead_date": wdates, "channel": wch,
    "project_id": wproj, "client_type": ctype, "area_required_sqft": area_req,
    "lease_term_years": lease_term, "quoted_rent_per_sqft": quoted_rent, "response_time_hrs": wresp,
    "decision_maker_contacted": decision_maker, "site_visits": wvisits, "proposal_sent": proposal,
    "negotiation_rounds": nego_rounds, "marketing_cost": wcost, "converted": wconv,
    "total_lease_value": lease_value})
w = wh.copy()
w["lead_date"] = [d.strftime(rng.choice(["%Y-%m-%d", "%d-%m-%Y"], p=[.8, .2])) for d in w.lead_date]
w["channel"] = w.channel.map(messy)
w["client_type"] = w.client_type.map(lambda x: x.lower() if rng.random() < .05 else x)
w["converted"] = np.where(w.converted == 1, "Yes", "No")
for col, p in [("area_required_sqft", .03), ("response_time_hrs", .04), ("quoted_rent_per_sqft", .02),
               ("client_type", .03)]:
    w.loc[rng.random(N_W) < p, col] = np.nan
idx = rng.choice(N_W, 5, replace=False); w.loc[idx, "area_required_sqft"] = w.loc[idx, "area_required_sqft"] * 50
idx = rng.choice(N_W, 8, replace=False); w.loc[idx, "response_time_hrs"] = -w.loc[idx, "response_time_hrs"]
w = pd.concat([w, w.sample(20, random_state=2)], ignore_index=True).sample(frac=1, random_state=4).reset_index(drop=True)
w.to_csv(RAW + "warehouse_leads_raw.csv", index=False)

# ================================================================== CAMPAIGN PERFORMANCE (monthly x channel)
rows = []
ctr_map = {"Google Ads": .035, "Meta Ads": .012, "Property Portal": .02, "WhatsApp/Email Campaign": .06,
           "LinkedIn Outreach": .008, "Website/SEO": .03}
offline = {"Referral", "Walk-in", "Broker Network", "Hoardings/OOH", "Broker", "Industry Expo",
           "3PL Directory", "Cold Calling"}
for seg, df in [("Residential", res), ("Warehouse", wh)]:
    df = df.assign(month=df.lead_date.dt.to_period("M").astype(str))
    g = df.groupby(["month", "channel"]).agg(leads=("lead_id", "count"), spend_var=("marketing_cost", "sum"),
                                               conversions=("converted", "sum")).reset_index()
    for _, x in g.iterrows():
        fixed = rng.uniform(0.04, 0.12) * x.spend_var   # agency/retainer overhead
        spend = round(x.spend_var + fixed, 0)
        if x.channel in offline:
            imp = clicks = np.nan
        else:
            ctr = ctr_map.get(x.channel, .02)
            imp = int(x.leads / (ctr * rng.uniform(.03, .08)))
            clicks = int(imp * ctr * rng.uniform(.9, 1.1))
        rows.append([x.month, seg, x.channel, spend, imp, clicks, x.leads, x.conversions])
camp = pd.DataFrame(rows, columns=["month", "segment", "channel", "spend", "impressions", "clicks",
                                   "leads_reported", "conversions_reported"])
camp["channel"] = camp.channel.map(messy)
camp.loc[rng.random(len(camp)) < .03, "spend"] = np.nan
camp = pd.concat([camp, camp.sample(6, random_state=5)], ignore_index=True)
camp.to_csv(RAW + "campaign_performance_raw.csv", index=False)

print("Residential raw:", r.shape, "| Warehouse raw:", w.shape, "| Campaign raw:", camp.shape,
      "| Projects:", projects.shape, "| Market:", market.shape)
print("True residential conv rate: %.3f | True warehouse conv rate: %.3f" % (conv.mean(), wconv.mean()))
