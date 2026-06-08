# Themis Coalition Simulator

An interactive simulation of the Themis carbon pricing mechanism. The dashboard takes calibrated preference data for 9 geopolitical actors, runs the full Themis elicitation procedure, and shows what carbon price, coverage, and international transfers the mechanism produces.

## What is Themis?

Themis is a framework for international climate cooperation through a common carbon price. The core idea: nations declare how much carbon pricing they'd accept under different levels of global participation, and the mechanism selects the price-coverage combination that maximises total climate impact — while ensuring that no nation is committed to a price it said it wouldn't accept.

The mechanism has three key properties:
- **Consistency**: at the selected price, enough actors actually accept it to achieve the target coverage
- **Strategy-proofness**: for any fixed coverage level, no actor can improve their outcome by misreporting preferences (quantile-based voting)
- **Pool balance**: international transfers between contributors and beneficiaries sum to exactly zero

## The preference function

Each actor's willingness to accept a carbon price depends on three things:

```
p_i(c, T) = max(0,  α_base  +  α_cov × c  +  α_trf × transfer_effect)
```

| Term | What it captures | How it's calibrated |
|------|-----------------|-------------------|
| **α_base** | Baseline willingness — what the actor already accepts | Anchored in OECD Effective Carbon Rates and ICAP ETS data. For the EU: headline ETS price × auctioned share × sector coverage. For China: OECD Net ECR directly. |
| **α_cov × c** | Coverage sensitivity — willingness rises when the coalition is broad | Calibrated via a thought experiment: "at 70% global coverage with no transfers, what max price would this actor accept?" Then α_cov = (that price − α_base) / 0.7. Tested across Monte Carlo ranges. |
| **α_trf × transfer** | Transfer sensitivity — redistribution changes willingness | α_trf = k / GDP_per_capita (k=20,000, capped at 20). Transfers matter more to poor countries. |

The 70% calibration point is just a reference for computing the slope — the linear function works at all coverage levels. The mechanism evaluates it everywhere from 0% to 100%.

All α_cov values are modelling assumptions, not empirical measurements. They are stress-tested through 500 Monte Carlo runs using triangular distributions within Data Bible uncertainty ranges.

## The T+/T− transfer mechanism

Contributors (per-capita emissions above the world average of 6.6 tCO₂e) pay into an international pool. Beneficiaries (below 6.6) receive from it.

- **T+** is the contributor transfer rate, optimised by the mechanism alongside price and coverage
- **T−** is solved endogenously so the pool balances exactly: T− = T+ × (total contributor excess) / (total beneficiary deficit)

Transfers are proportional to the **gap** from the world average, not total emissions. This means a country at 7.0 tCO₂e/cap pays very little (barely above average), while a country at 23 tCO₂e/cap pays substantially.

T+ is not set by hand — the mechanism grid-searches over T+ values alongside coverage to find the combination that maximises c × p (total climate impact).

## The 9 actors

The simulation uses 9 geopolitical actors: the six largest GHG emitters individually (China, US, EU, India, Russia, Indonesia — per EDGAR 2025), plus three grouped archetypes.

| Actor | Role | α_base | Why this value |
|-------|------|--------|---------------|
| China | Industrial swing actor | €8.6 | OECD Net ECR directly |
| United States | Volatile super-emitter | €3.0 | No federal price; OECD explicit covers 7.4% of emissions |
| European Union | Carbon-pricing anchor | €18.5 | ETS headline × 57% auctioned × 40% coverage + taxes |
| India | Development-first giant | €4.0 | OECD Net ECR €8.09 politically adjusted (fuel excise ≠ international willingness) |
| Russia | Fossil-geopolitical spoiler | €0 | No carbon pricing; oil/gas ~20-25% of state budget |
| Indonesia | Emerging swing actor | €0 | OECD Net ECR negative (−€4.98), floored to zero |
| Advanced conditional joiners | UK, Japan, Canada, Australia, South Korea | €5.8 | Population-weighted from OECD/ICAP per member |
| Low-carbon frontier | Nigeria, Bangladesh, Ethiopia | €0 | No carbon pricing; join through transfers |
| Hydrocarbon rentiers | Saudi Arabia, UAE | €0 | Fossil fuels ~100% of energy system |

## How to use the dashboard

### 🤝 The Coalition (main page)

The headline result: what price, who joins, what flows. Each actor has an expandable card showing:
- Their financial position (carbon cost, net transfer, % of GDP)
- Their willingness curve (how willingness changes with coverage, with and without transfers)
- Monte Carlo join probability (how often they join across 500 parameter perturbations)
- For grouped actors: which countries are inside the group

**What to look for**: India and the US are marginal joiners — barely above the price. The EU barely pays anything in transfers despite being the anchor. Low-carbon frontier countries receive transformative transfers (2-3% of GDP).

### 🔀 What If

Test scenarios that change the world:
- **Quick scenarios**: "US withdraws", "China more cautious", "Double transfers", etc. Each reruns the full mechanism.
- **Drill into a group**: Split individual countries out of a grouped archetype. The split-out country gets its own calibrated parameters from the Data Bible. Key finding: Japan and South Korea wouldn't join individually despite being inside a "joining" group — the group average masked their low willingness.
- **Custom edit**: Change any parameter directly and rerun.

### ⚙️ The Mechanism

The technical plots that show HOW the mechanism selects its outcome:
- **Quantile price curve**: as coverage rises, the feasible price drops (you need less-willing actors to accept). The peak of c × p is the mechanism's choice.
- **T+ frontier**: shows that T+ is optimised, not arbitrary. The mechanism finds the transfer level that maximises coalition impact.
- **Monte Carlo distributions**: price and coverage histograms from 500 runs, plus join-reliability bars per actor.

### ✅ Audit

Seven diagnostic checks confirming the mechanism is internally consistent: all joiners accept the price, all non-joiners are below it, coverage meets target, pool balances, weights sum to 1, parameters present, benchmark correct.

## Baseline result

At baseline calibration, the mechanism produces:
- **Carbon price**: €26.54/ton
- **Coverage**: 87% of modelled emissions
- **T+**: 0.21 (optimised), **T−**: 0.30 (solved)
- **Pool**: €63bn/yr, balanced to €0.0000M
- **Joiners**: China, US, EU, India, Advanced joiners, Low-carbon frontier
- **Non-joiners**: Russia, Indonesia, Hydrocarbon rentiers

## Data sources

- **Emissions**: EDGAR 2025 (JRC) — per-capita GHG including CO₂, CH₄, N₂O, F-gases
- **Population and GDP**: World Bank 2024
- **Carbon pricing**: OECD Effective Carbon Rates 2025, ICAP ETS factsheets, EU Commission, Climate Action Tracker
- **Full parameter audit trail**: Themis Data Bible (Excel workbook with 7 tabs of sourced justification)

## Deployment

```
pip install streamlit pandas numpy plotly
streamlit run streamlit_app.py
```

First load pre-computes 500 Monte Carlo runs (~30-60 seconds). After that, all pages are instant.

## What this simulation does NOT do

This dashboard tests whether the mechanism produces coherent outcomes under **truthful reporting**. It does not test whether strategic actors can exploit the coverage-optimisation step to manipulate the outcome — that is the subject of RQ2, which uses EGTA and reinforcement learning in a separate analysis.
