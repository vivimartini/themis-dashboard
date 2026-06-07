# Themis Streamlit Dashboard

This is the Streamlit version of the Themis working model. It is designed as a public-facing / Carl-facing interactive simulator, not an RQ2 strategy module.

## What it does

The app walks users through Themis in layers:

1. **Themis at a glance** — simple headline result: selected price, coverage, T+, solved T−, joiners, non-joiners, and pool balance.
2. **Play with the world** — editable actor parameters and quick scenario buttons.
3. **How Themis chooses** — preference curves, weighted-quantile feasible price curve, and the c × p objective.
4. **Actor explorer** — country/archetype narratives, alpha parameters, willingness at outcome, join status, and transfer position.
5. **Nested archetypes** — Carl's A1–A7 taxonomy as the drill-down layer, with split-out controls.
6. **Financial flows** — contributor/beneficiary accounting using fixed world average ē = 6.6 and balanced T+/T− transfers.
7. **Robustness** — Monte Carlo uncertainty runs over Data Bible parameter ranges.
8. **Diagnostics** — audit checks for joiner thresholds, coverage, pool balance, and data validity.

## Files

- `streamlit_app.py` — Streamlit UI.
- `themis_engine.py` — Themis mechanism, transfer accounting, diagnostics, Monte Carlo.
- `data/actors_baseline.csv` — Data Bible actor calibration.
- `data/country_data.csv` — country-level drill-down data from the Data Bible.
- `data/carl_archetypes.csv` — Carl's A1–A7 country taxonomy.
- `requirements.txt` — dependencies for deployment.

## Core modelling choices

- Fixed world-average benchmark: **ē = 6.6 tCO₂e/cap**.
- Contributors pay using **T+** on emissions above ē.
- Beneficiaries receive using **T−**, solved endogenously so the pool balances.
- The baseline actor calibration comes from the Themis Data Bible.
- RQ2 / strategy-manipulation testing is intentionally excluded from this version.

## Deployment notes

The app is ready to deploy on Streamlit Community Cloud from a GitHub repository. The entrypoint is `streamlit_app.py`.
