from __future__ import annotations
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from themis_engine import (
    EngineConfig, EBAR_DEFAULT, normalise_actor_df, run_mechanism, diagnostics,
    run_monte_carlo, split_country_from_group, preference_values, price_for_coverage,
)

st.set_page_config(page_title="Themis Simulator", page_icon="⚖️", layout="wide")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

ROLE_TEXT = {
    "CHINA": "Pivotal industrial swing actor: large emissions weight, moderate effective carbon-price exposure, and high sensitivity to broad reciprocal coverage.",
    "UNITED STATES": "Volatile super-emitter: low federal baseline, high political uncertainty, but more willing when broad coverage reduces competitiveness concerns.",
    "EUROPEAN UNION": "Carbon-pricing anchor: existing ETS architecture and high willingness when a broader coalition prevents free-riding.",
    "INDIA": "Development-first transfer-sensitive giant: below world-average per-capita emissions, so transfers make participation materially easier.",
    "RUSSIA": "Fossil-geopolitical spoiler: low baseline willingness and weak responsiveness to coverage due to fossil rents and geopolitical incentives.",
    "INDONESIA": "Emerging swing actor: coal, land-use, nickel/industrial strategy and transition finance make willingness sensitive and uncertain.",
    "ADV. CARBON-PRICED CONDITIONAL JOINERS": "Advanced conditional joiners: rich economies with climate-policy architecture but competitiveness concerns unless major emitters join.",
    "LOW-CARBON FRONTIER": "Transfer-led beneficiaries: low per-capita emitters whose participation is driven by redistribution and fairness.",
    "HYDROCARBON RENTIERS": "Fossil-rent resisters: high per-capita emissions and fossil-revenue dependence make baseline willingness very low.",
}

CARL_MAP = pd.DataFrame([
    {"Carl archetype": "A1", "Broad type": "EU / developed service economies", "Simulation treatment": "European Union as institutional actor", "Why": "EU climate policy and ETS architecture are shared enough to model as a strategic bloc."},
    {"Carl archetype": "A2", "Broad type": "Advanced non-EU economies", "Simulation treatment": "Advanced conditional joiners; optional split-out for UK, Japan, Canada, Australia, South Korea", "Why": "Similar competitiveness/reciprocity logic, but important countries can be split for sensitivity."},
    {"Carl archetype": "A3", "Broad type": "United States-type actor", "Simulation treatment": "United States individual", "Why": "Large emissions weight and no federal price make aggregation misleading."},
    {"Carl archetype": "A4", "Broad type": "China bloc", "Simulation treatment": "China individual", "Why": "China dominates emissions weight and is a pivotal swing actor."},
    {"Carl archetype": "A5", "Broad type": "Hydrocarbon states", "Simulation treatment": "Hydrocarbon rentiers + Russia as separate spoiler", "Why": "Fossil-rent states resist for rent reasons; Russia also has geopolitical spoiler dynamics."},
    {"Carl archetype": "A6", "Broad type": "Transitioning / industrialising states", "Simulation treatment": "India and Indonesia individual; residuals can be explored", "Why": "A6 mixes very different transfer positions; India/Indonesia are too pivotal to bury."},
    {"Carl archetype": "A7", "Broad type": "Low-carbon / residual frontier", "Simulation treatment": "Low-carbon frontier group", "Why": "The core mechanism role is transfer-led participation."},
])

@st.cache_data
def load_data():
    actors = pd.read_csv(os.path.join(DATA_DIR, "actors_baseline.csv"))
    countries = pd.read_csv(os.path.join(DATA_DIR, "country_data.csv"))
    try:
        carl = pd.read_csv(os.path.join(DATA_DIR, "carl_archetypes.csv"))
    except Exception:
        carl = pd.DataFrame()
    return normalise_actor_df(actors), countries, carl

actors_base, countries_df, carl_df = load_data()

# Session state for interactive actors.
if "actors_current" not in st.session_state:
    st.session_state.actors_current = actors_base.copy()
if "scenario_name" not in st.session_state:
    st.session_state.scenario_name = "Data Bible baseline"


def reset_scenario():
    st.session_state.actors_current = actors_base.copy()
    st.session_state.scenario_name = "Data Bible baseline"


def metric_card(label: str, value: str, help_text: str = ""):
    st.metric(label, value, help=help_text)


def format_eur(x):
    return f"€{x:,.2f}"


def build_sidebar():
    st.sidebar.title("⚖️ Themis Simulator")
    st.sidebar.caption("Interactive working model: Data Bible → preferences → Themis outcome → transfers.")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Themis at a glance",
            "Play with the world",
            "How Themis chooses",
            "Actor explorer",
            "Nested archetypes",
            "Financial flows",
            "Robustness",
            "Diagnostics",
        ],
        index=0,
    )
    st.sidebar.divider()
    st.sidebar.write("**Global assumptions**")
    t_cap_choice = st.sidebar.selectbox("Contributor transfer cap", ["No cap", "0.30", "0.40", "0.50", "0.70", "1.00"], index=0)
    t_cap = None if t_cap_choice == "No cap" else float(t_cap_choice)
    c_steps = st.sidebar.slider("Coverage grid resolution", 40, 140, 100, 10)
    t_steps = st.sidebar.slider("Transfer grid resolution", 31, 121, 101, 10)
    st.sidebar.info("World average benchmark is fixed at ē = 6.6 tCO₂e/cap. T− is solved to balance the transfer pool.")
    if st.sidebar.button("Reset to Data Bible baseline", use_container_width=True):
        reset_scenario()
        st.rerun()
    return page, EngineConfig(ebar=EBAR_DEFAULT, c_steps=c_steps, t_steps=t_steps, t_cap=t_cap)

page, config = build_sidebar()
actors = normalise_actor_df(st.session_state.actors_current)
res = run_mechanism(actors, config=config)
actor_res = res["actor_results"]
acct = res["accounting"]
diag = diagnostics(actors, res)

st.title("Themis: Interactive Carbon Pricing Coalition Simulator")
st.caption("A working model of Themis for fast public explanation, policy exploration, and mechanism audit. RQ2 strategy testing is intentionally left out of this version.")

if page == "Themis at a glance":
    st.subheader("Themis in 30 seconds")
    st.markdown(
        """
        **Themis asks a simple question:** what common carbon price can enough emissions-weighted actors accept?  
        The mechanism selects a price, coverage level and contributor transfer rate, then solves the beneficiary payout rate so the international pool balances.
        """
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: metric_card("Carbon price p*", format_eur(res["p_star"]), "Selected common carbon price.")
    with col2: metric_card("Target coverage c*", f"{res['c_star']:.1%}", "Coverage selected by the mechanism.")
    with col3: metric_card("Actual coverage", f"{res['actual_coverage']:.1%}", "Emission-weighted coverage from realised joiners.")
    with col4: metric_card("T+", f"{res['Tplus_star']:.2f}", "Contributor transfer intensity.")
    with col5: metric_card("T−", f"{res['Tminus_actual']:.2f}", "Beneficiary payout rate solved to balance pool.")

    sent = acct["total_sent_mEUR"].sum(); received = acct["total_received_mEUR"].sum(); balance = sent - received
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Total sent into pool", f"€{sent/1000:,.1f}bn")
    with c2: metric_card("Total paid out", f"€{received/1000:,.1f}bn")
    with c3: metric_card("Pool balance", f"€{balance:,.4f}m")

    st.markdown("### Coalition result")
    left, right = st.columns([1.2, 1])
    with left:
        display = actor_res[["name", "join_status", "preference_at_solution", "weight", "e", "role"]].copy()
        display["weight"] = display["weight"].map(lambda x: f"{x:.1%}")
        display["preference_at_solution"] = display["preference_at_solution"].map(lambda x: f"€{x:.2f}")
        st.dataframe(display.rename(columns={"name":"Actor", "join_status":"Status", "preference_at_solution":"Willingness at outcome", "weight":"Emissions weight", "e":"tCO₂e/cap", "role":"Transfer role"}), use_container_width=True, hide_index=True)
    with right:
        fig = px.bar(acct, x="name", y="net_transfer_per_cap", color="status", title="Net international transfer per capita", labels={"name":"Actor", "net_transfer_per_cap":"€/capita"})
        fig.update_layout(xaxis_tickangle=-35, height=430)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### What this result means")
    st.success(
        "The model finds a broad coalition when the selected price is acceptable to enough emissions-weighted actors. "
        "High emitters pay into the pool; low emitters receive; T− is solved so what goes in equals what comes out."
    )

elif page == "Play with the world":
    st.subheader("Play with the world")
    st.markdown("Adjust actor assumptions and instantly rerun Themis. This is the Model UN-style sandbox.")
    work = actors.copy()
    edit_cols = ["name", "alpha_base", "alpha_cov", "alpha_trf", "e", "pop_m", "gdp_cap"]
    edited = st.data_editor(
        work[edit_cols],
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Actor", disabled=True),
            "alpha_base": st.column_config.NumberColumn("α_base", min_value=-50.0, max_value=120.0, step=0.5),
            "alpha_cov": st.column_config.NumberColumn("α_cov", min_value=0.0, max_value=150.0, step=1.0),
            "alpha_trf": st.column_config.NumberColumn("α_trf", min_value=0.0, max_value=25.0, step=0.1),
            "e": st.column_config.NumberColumn("tCO₂e/cap", min_value=-10.0, max_value=40.0, step=0.1),
            "pop_m": st.column_config.NumberColumn("Population M", min_value=0.0, max_value=2000.0, step=1.0),
            "gdp_cap": st.column_config.NumberColumn("GDP/cap", min_value=0.0, max_value=150000.0, step=100.0),
        },
    )
    if st.button("Apply edits and rerun", type="primary"):
        updated = work.copy()
        for col in edited.columns:
            updated[col] = edited[col]
        updated["weight"] = updated["pop_m"] * updated["e"].clip(lower=0)
        st.session_state.actors_current = normalise_actor_df(updated)
        st.session_state.scenario_name = "Edited scenario"
        st.rerun()

    st.markdown("### Quick scenario buttons")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Lower EU α_base to 12"):
            temp = actors.copy(); temp.loc[temp["name"].str.contains("EUROPEAN", case=False), "alpha_base"] = 12
            st.session_state.actors_current = normalise_actor_df(temp); st.session_state.scenario_name = "EU base lowered"; st.rerun()
    with c2:
        if st.button("Make US less cooperative"):
            temp = actors.copy(); m=temp["name"].str.contains("UNITED", case=False); temp.loc[m,"alpha_cov"] = temp.loc[m,"alpha_cov"]*0.65
            st.session_state.actors_current = normalise_actor_df(temp); st.session_state.scenario_name = "US less cooperative"; st.rerun()
    with c3:
        if st.button("Make China more cautious"):
            temp = actors.copy(); m=temp["name"].str.contains("CHINA", case=False); temp.loc[m,"alpha_cov"] = temp.loc[m,"alpha_cov"]*0.75
            st.session_state.actors_current = normalise_actor_df(temp); st.session_state.scenario_name = "China more cautious"; st.rerun()
    with c4:
        if st.button("Boost India transfer sensitivity"):
            temp = actors.copy(); m=temp["name"].str.contains("INDIA", case=False); temp.loc[m,"alpha_trf"] = temp.loc[m,"alpha_trf"]*1.25
            st.session_state.actors_current = normalise_actor_df(temp); st.session_state.scenario_name = "India transfer boost"; st.rerun()

    st.markdown("### Current scenario outcome")
    st.write(f"Scenario: **{st.session_state.scenario_name}**")
    st.dataframe(actor_res[["name", "preference_at_solution", "joins", "e", "weight", "alpha_base", "alpha_cov", "alpha_trf"]], use_container_width=True, hide_index=True)

elif page == "How Themis chooses":
    st.subheader("How Themis chooses the deal")
    st.markdown(
        """
        The mechanism has three visible steps:  
        **1.** Actors have willingness curves.  
        **2.** The weighted quantile gives the feasible price at each coverage level.  
        **3.** The mechanism chooses the point that maximises **coverage × price**.
        """
    )
    c_grid = np.linspace(config.c_min, config.c_max, 100)
    curves = []
    # Use selected T+ and T- for preference curves.
    for _, row in actors.iterrows():
        for c in c_grid:
            val = preference_values(
                np.array([row["e"]]), np.array([row["alpha_base"]]), np.array([row["alpha_cov"]]), np.array([row["alpha_trf"]]),
                float(c), res["Tplus_star"], res["Tminus_actual"], EBAR_DEFAULT
            )[0]
            curves.append({"coverage": c, "price": val, "actor": row["name"]})
    curve_df = pd.DataFrame(curves)
    fig = px.line(curve_df, x="coverage", y="price", color="actor", title="Actor willingness curves at selected transfer intensity", labels={"coverage":"Coverage c", "price":"Acceptable price €/t"})
    fig.add_hline(y=res["p_star"], line_dash="dash", annotation_text="p*")
    fig.add_vline(x=res["c_star"], line_dash="dash", annotation_text="c*")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.line(res["curve"], x="coverage", y="feasible_price", title="Weighted-quantile feasible price curve", labels={"coverage":"Coverage c", "feasible_price":"p(c) €/t"})
        fig2.add_scatter(x=[res["c_star"]], y=[res["p_star"]], mode="markers", marker=dict(size=12), name="selected")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.line(res["curve"], x="coverage", y="objective", title="Objective curve: c × p(c)", labels={"coverage":"Coverage c", "objective":"c × p"})
        fig3.add_scatter(x=[res["c_star"]], y=[res["objective"]], mode="markers", marker=dict(size=12), name="selected")
        st.plotly_chart(fig3, use_container_width=True)

elif page == "Actor explorer":
    st.subheader("Actor explorer")
    selected = st.selectbox("Choose an actor", actors["name"].tolist())
    row = actor_res[actor_res["name"] == selected].iloc[0]
    accrow = acct[acct["name"] == selected].iloc[0]
    st.markdown(f"### {selected}")
    st.info(ROLE_TEXT.get(selected, row.get("narrative", "No narrative available.")))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Emissions/capita", f"{row['e']:.2f}")
    c2.metric("Emissions weight", f"{row['weight']:.1%}")
    c3.metric("Willingness", format_eur(row["preference_at_solution"]))
    c4.metric("Selected p*", format_eur(res["p_star"]))
    c5.metric("Joins?", "Yes" if row["joins"] else "No")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Parameters")
        st.dataframe(pd.DataFrame([
            {"Parameter":"α_base", "Value":row["alpha_base"], "Interpretation":"baseline effective carbon-price willingness"},
            {"Parameter":"α_cov", "Value":row["alpha_cov"], "Interpretation":"coverage/reciprocity sensitivity"},
            {"Parameter":"α_trf", "Value":row["alpha_trf"], "Interpretation":"transfer sensitivity"},
        ]), use_container_width=True, hide_index=True)
        st.markdown("#### Financial position")
        st.dataframe(pd.DataFrame([
            {"Flow":"Sent/capita", "Value":accrow["sent_per_cap"]},
            {"Flow":"Received/capita", "Value":accrow["received_per_cap"]},
            {"Flow":"Net transfer/capita", "Value":accrow["net_transfer_per_cap"]},
            {"Flow":"Total net transfer mEUR", "Value":accrow["net_total_mEUR"]},
        ]), use_container_width=True, hide_index=True)
    with right:
        tmp=[]
        for c in np.linspace(0.01,1,100):
            tmp.append({"coverage":c,"price":preference_values(np.array([row["e"]]),np.array([row["alpha_base"]]),np.array([row["alpha_cov"]]),np.array([row["alpha_trf"]]),float(c),res["Tplus_star"],res["Tminus_actual"],EBAR_DEFAULT)[0]})
        fig=px.line(pd.DataFrame(tmp), x="coverage", y="price", title=f"{selected} willingness curve")
        fig.add_hline(y=res["p_star"], line_dash="dash", annotation_text="p*")
        fig.add_vline(x=res["c_star"], line_dash="dash", annotation_text="c*")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Nested archetypes":
    st.subheader("Nested archetypes")
    st.markdown("Carl's A1–A7 taxonomy is used as a **world map / drill-down layer**. The simulation core keeps strategically pivotal actors separate.")
    st.dataframe(CARL_MAP, use_container_width=True, hide_index=True)
    st.markdown("### Split a country out of a group")
    available = countries_df["country"].astype(str).tolist()
    split_country = st.selectbox("Country to split out", available)
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Split selected country and rerun", type="primary"):
            st.session_state.actors_current = split_country_from_group(actors, countries_df, split_country)
            st.session_state.scenario_name = f"Split out {split_country}"
            st.rerun()
    with col2:
        if st.button("Merge/reset all split-outs"):
            reset_scenario(); st.rerun()
    st.markdown("### Current runtime actor set")
    st.dataframe(actors[["name","e","pop_m","weight","alpha_base","alpha_cov","alpha_trf"]], use_container_width=True, hide_index=True)
    if not carl_df.empty:
        st.markdown("### Carl archetype file preview")
        st.dataframe(carl_df.head(120), use_container_width=True, hide_index=True)

elif page == "Financial flows":
    st.subheader("Transfer pool and financial flows")
    st.markdown("Fixed benchmark: **ē = 6.6 tCO₂e/cap**. Contributors pay using **T+**; beneficiaries receive using **T−**, which is solved so the pool balances.")
    sent = acct["total_sent_mEUR"].sum(); received = acct["total_received_mEUR"].sum(); balance = sent-received
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("T+", f"{res['Tplus_star']:.3f}")
    col2.metric("T−", f"{res['Tminus_actual']:.3f}")
    col3.metric("Total pool", f"€{sent/1000:,.1f}bn")
    col4.metric("Balance", f"€{balance:,.4f}m")
    st.latex(r"T^- = T^+ \times \frac{\sum_i pop_i \max(e_i - \bar{e}, 0)}{\sum_i pop_i \max(\bar{e} - e_i, 0)}")
    flows = acct[["name","joins","status","sent_per_cap","received_per_cap","net_transfer_per_cap","total_sent_mEUR","total_received_mEUR","net_total_mEUR"]].copy()
    st.dataframe(flows, use_container_width=True, hide_index=True)
    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(flows, x="name", y="net_transfer_per_cap", color="status", title="Per-capita net international transfer")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2=px.bar(flows, x="name", y="net_total_mEUR", color="status", title="Aggregate net international transfer")
        fig2.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Robustness":
    st.subheader("Monte Carlo robustness")
    st.markdown("This checks whether the basic Themis outcome survives uncertainty in the Data Bible parameters.")
    n = st.slider("Runs", 100, 2000, 500, 100)
    seed = st.number_input("Seed", min_value=1, max_value=99999, value=42)
    run = st.button("Run Monte Carlo", type="primary")
    if run:
        with st.spinner("Running simulations..."):
            mc, joins = run_monte_carlo(actors, n=int(n), seed=int(seed), config=EngineConfig(ebar=EBAR_DEFAULT, c_steps=50, t_steps=51, t_cap=config.t_cap))
        st.success("Monte Carlo complete")
        c1,c2,c3=st.columns(3)
        c1.metric("Mean p*", format_eur(mc["p_star"].mean()))
        c2.metric("Median coverage", f"{mc['actual_coverage'].median():.1%}")
        c3.metric("Mean T+", f"{mc['Tplus_star'].mean():.3f}")
        a,b,c=st.columns(3)
        with a: st.plotly_chart(px.histogram(mc, x="p_star", nbins=30, title="Distribution of selected price p*"), use_container_width=True)
        with b: st.plotly_chart(px.histogram(mc, x="actual_coverage", nbins=30, title="Distribution of actual coverage"), use_container_width=True)
        with c: st.plotly_chart(px.histogram(mc, x="Tplus_star", nbins=30, title="Distribution of T+"), use_container_width=True)
        st.markdown("### Join probabilities")
        st.dataframe(joins.sort_values("Join probability", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download Monte Carlo CSV", mc.to_csv(index=False), file_name="themis_monte_carlo.csv", mime="text/csv")
    else:
        st.info("Run Monte Carlo to see robustness distributions and join probabilities.")

elif page == "Diagnostics":
    st.subheader("Diagnostics / audit")
    st.markdown("This page checks that the mechanism is not a black box. Every rerun should pass these checks.")
    st.dataframe(diag, use_container_width=True, hide_index=True)
    failed = diag[diag["Status"] != "PASS"]
    if failed.empty:
        st.success("All core diagnostics pass.")
    else:
        st.error("One or more diagnostics failed. Inspect the table above.")
    st.markdown("### Mechanism frontier by T+")
    st.dataframe(res["frontier_by_Tplus"].tail(20), use_container_width=True, hide_index=True)
    st.markdown("### Current actor data")
    st.dataframe(actors, use_container_width=True, hide_index=True)

st.divider()
st.caption("Prototype Streamlit implementation. Baseline uses fixed world average ē = 6.6, T+/T− balanced transfers, and the Data Bible actor calibration. Strategy/RQ2 module intentionally excluded from this public v1.")
