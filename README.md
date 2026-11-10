# FMB Marketing Analytics

Individual assessment: data analytics for the marketing strategy of a family-managed construction business (residential flats for sale, warehouses for lease). Author: Raj Gorani. All data is synthetic.

## Live dashboards
- Residential flat sales: _add Streamlit link here_
- Warehouse leasing: _add Streamlit link here_

## Contents
| Path | Purpose |
|---|---|
| `01_generate_synthetic_data.py` | Generates the synthetic datasets (seed 42) |
| `02_data_cleaning_transformation.py` | Cleaning, transformation, feature engineering, cleaning log |
| `03_eda_descriptive_diagnostic.py` | Descriptive and diagnostic analytics, hypothesis tests, driver model |
| `04_streamlit_dashboard.py` | Single combined dashboard |
| `app_residential.py` | Residential dashboard |
| `app_warehouse.py` | Warehouse dashboard |
| `FMB_Marketing_Analytics_Report.docx` | Written report |
| `data/raw`, `data/clean` | Raw and cleaned datasets |
| `figures/`, `tables/` | Charts and result tables |

## Run locally
```
pip install -r requirements.txt
streamlit run app_residential.py
streamlit run app_warehouse.py
```
(Scripts 01 to 03 need `scikit-learn` as well: `pip install scikit-learn`.)

## Deploy on Streamlit Community Cloud
Create one app per dashboard at share.streamlit.io, with this repository, branch `main`, and main file `app_residential.py` or `app_warehouse.py`.
