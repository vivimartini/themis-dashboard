"""
Themis Coalition Simulator — Policymaker Edition
Pre-computes Monte Carlo on load. Every number shows its uncertainty.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st

from themis_engine import (
    EngineConfig, EBAR_DEFAULT, normalise_actor_df, run_mechanism,
    diagnostics, run_monte_carlo, split_country_from_group,
    preference_values, price_for_coverage, solve_tminus, arrays,
)

st.set_page_config(page_title="Themis Simulator", page_icon="⚖️", layout="wide")
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

ROLE = {
    "CHINA": ("Pivotal industrial swing actor", "The world's largest emitter. Willing to join a broad coalition at moderate prices — but not at EU levels. CBAM pressure makes inside cheaper than outside.", "#e74c3c"),
    "UNITED STATES": ("Volatile super-emitter", "No federal carbon price. Willingness depends on who's in power. Broad coverage eases competitiveness fears, but politics can override economics.", "#3498db"),
    "EUROPEAN UNION": ("Carbon-pricing anchor", "Already pricing carbon via the ETS. Themis gives the EU what it's been doing alone — but with everyone else sharing the commitment. **And it costs the EU almost nothing in transfers**, because EU emissions are close to the world average.", "#27ae60"),
    "INDIA": ("Development-first giant", "Below the world average in emissions. Receives transfers from day one. The question isn't whether India benefits — it's whether the price is low enough to join.", "#f39c12"),
    "RUSSIA": ("Fossil-geopolitical spoiler", "Fossil revenues fund the state. Carbon pricing threatens the economic model. Willingness stays very low regardless of who else joins.", "#7f8c8d"),
    "INDONESIA": ("Emerging swing actor", "Coal, palm oil, nickel, deforestation. A complex emerging economy where the right transition finance could unlock participation.", "#1abc9c"),
    "ADV. CARBON-PRICED CONDITIONAL JOINERS": ("Advanced conditional joiners", "UK, Japan, Canada, Australia, South Korea. Each has some carbon pricing. They'll join if the big emitters do — levelling the playing field.", "#9b59b6"),
    "LOW-CARBON FRONTIER": ("Transfer-led beneficiaries", "Nigeria, Bangladesh, Ethiopia. Emit almost nothing per person. Themis pays them for being below the world average.", "#2ecc71"),
    "HYDROCARBON RENTIERS": ("Fossil-rent resisters", "Saudi Arabia, UAE. Fossil fuels are nearly 100% of the energy system. Carbon pricing is existential.", "#e67e22"),
}

MEMBERS = {
    "ADV. CARBON-PRICED CONDITIONAL JOINERS": ["United Kingdom", "Japan", "Canada", "Australia", "South Korea"],
    "LOW-CARBON FRONTIER": ["Nigeria", "Bangladesh", "Ethiopia"],
    "HYDROCARBON RENTIERS": ["Saudi Arabia", "UAE"],
}

@st.cache_data
def load_data():
    actors = pd.read_csv(os.path.join(DATA_DIR, "actors_baseline.csv"))
    countries = pd.read_csv(os.path.join(DATA_DIR, "country_data.csv"))
    return normalise_actor_df(actors), countries

@st.cache_data
def precompute_mc(n=500):
    actors, _ = load_data()
    mc_config = EngineConfig(c_steps=60, t_steps=61)
    mc_df, join_probs = run_monte_carlo(actors, n=n, config=mc_config)
    return mc_df, join_probs

actors_base, countries_df = load_data()

if "actors" not in st.session_state:
    st.session_state.actors = actors_base.copy()
if "scenario" not in st.session_state:
    st.session_state.scenario = "Baseline"
if "splits" not in st.session_state:
    st.session_state.splits = []

def reset():
    st.session_state.actors = actors_base.copy()
    st.session_state.scenario = "Baseline"
    st.session_state.splits = []

# Pre-compute MC and baseline
mc_df, join_probs = precompute_mc(500)
config = EngineConfig(ebar=EBAR_DEFAULT, c_steps=100, t_steps=101)
actors = normalise_actor_df(st.session_state.actors)
res = run_mechanism(actors, config=config)
acct = res["accounting"]
ar = res["actor_results"]

# Merge join probs into actor results
jp_map = dict(zip(join_probs["Actor"], join_probs["Join probability"]))
ar["join_pct"] = ar["name"].map(jp_map).fillna(0.5)

# ── Sidebar ──
st.sidebar.title("⚖️ Themis")
st.sidebar.caption("Carbon pricing coalition simulator")
page = st.sidebar.radio("", [
    "🤝 The Coalition",
    "🔀 What If",
    "⚙️ The Mechanism",
    "✅ Audit",
])
st.sidebar.divider()
st.sidebar.markdown(f"**Scenario:** {st.session_state.scenario}")
st.sidebar.markdown(f"**Benchmark:** ē = {EBAR_DEFAULT} tCO₂e/cap")
# MC context
st.sidebar.markdown(f"**Monte Carlo:** 500 runs pre-computed")
st.sidebar.markdown(f"Price range: €{mc_df['p_star'].quantile(0.1):.0f}–{mc_df['p_star'].quantile(0.9):.0f}")
st.sidebar.markdown(f"Coverage range: {mc_df['actual_coverage'].quantile(0.1):.0%}–{mc_df['actual_coverage'].quantile(0.9):.0%}")
if st.sidebar.button("Reset to baseline", use_container_width=True):
    reset(); st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 1: THE COALITION (merged Deal + Actor Explorer + MC)
# ═══════════════════════════════════════════════════════════════
if page == "🤝 The Coalition":
    st.title("The Coalition")

    # ── Headline with MC context ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Carbon price", f"€{res['p_star']:.2f}/ton",
              help=f"90% MC range: €{mc_df['p_star'].quantile(0.1):.0f}–{mc_df['p_star'].quantile(0.9):.0f}")
    c2.metric("Coverage", f"{res['actual_coverage']:.0%}",
              help=f"90% MC range: {mc_df['actual_coverage'].quantile(0.1):.0%}–{mc_df['actual_coverage'].quantile(0.9):.0%}")
    c3.metric("T+ (contributors)", f"{res['Tplus_star']:.2f}",
              help="Contributor transfer rate, optimised by mechanism alongside p and c")
    c4.metric("T− (beneficiaries)", f"{res['Tminus_actual']:.2f}",
              help="Solved endogenously so the international pool balances exactly")
    sent = acct["total_sent_mEUR"].sum()
    c5.metric("Annual pool", f"€{sent/1000:,.0f}bn",
              help="Total international transfers per year")

    # ── Explanation ──
    joiners = ar[ar["joins"]]["name"].tolist()
    non_joiners = ar[~ar["joins"]]["name"].tolist()
    st.info(
        f"At **€{res['p_star']:.2f}/ton**, actors covering **{res['actual_coverage']:.0%}** of global emissions join. "
        f"T+ is optimised by the mechanism (not set by hand) — it's the transfer level that produces the best c×p outcome. "
        f"T− is then solved so the pool balances exactly."
    )

    st.divider()

    # ── Actor cards ──
    st.markdown("### Actor positions")
    st.caption("Click any actor to expand. Join probability from 500 Monte Carlo runs.")

    for _, row in ar.iterrows():
        name = row["name"]
        role_title, role_desc, color = ROLE.get(name, ("", "", "#95a5a6"))
        acc_row = acct[acct["name"] == name].iloc[0]
        joins = row["joins"]
        jp = row["join_pct"]

        # Status bar
        status = "✅ Joins" if joins else "❌ Stays out"
        margin = row["preference_at_solution"] - res["p_star"]

        with st.expander(f"{'🟢' if joins else '🔴'} **{name}** — {role_title} | {status} | Willing: €{row['preference_at_solution']:.1f} | MC join: {jp:.0%}"):
            left, right = st.columns([1.2, 1])

            with left:
                st.markdown(role_desc)
                st.divider()

                # Financial position
                m1, m2, m3 = st.columns(3)
                m1.metric("Carbon cost", f"€{acc_row['collected_per_cap']:.0f}/cap/yr")
                net_trf = acc_row["net_transfer_per_cap"]
                m2.metric("Net transfer", f"€{net_trf:+.1f}/cap/yr")
                gdp = row["gdp_cap"]
                pct_gdp = abs(net_trf) / gdp * 100 if gdp > 0 else 0
                m3.metric("% of GDP", f"{pct_gdp:.2f}%")

                # EU insight
                if "EUROPEAN" in name:
                    st.success(
                        f"**The EU barely feels the transfers.** At {row['e']:.2f} tCO₂e/cap (just {row['e']-EBAR_DEFAULT:.2f} above "
                        f"world average), the EU's net contribution is only €{abs(net_trf):.1f}/cap/yr — {pct_gdp:.2f}% of GDP. "
                        f"Joining Themis costs the EU almost nothing while giving it what it's been trying to do alone: "
                        f"a level playing field where everyone prices carbon."
                    )

                # Marginal joiner insight
                if joins and 0 < margin < 5:
                    st.warning(f"**Marginal joiner.** Willingness is only €{margin:.1f} above the price. Small changes in assumptions could flip this actor out.")
                elif not joins and margin > -10:
                    st.warning(f"**Near-joiner.** Only €{abs(margin):.1f} below the price. More transfers or lower price could bring them in.")

                if name in MEMBERS:
                    st.markdown(f"**Countries inside:** {', '.join(MEMBERS[name])}")

            with right:
                # Willingness curve
                c_range = np.linspace(0, 1, 60)
                arr_data = arrays(actors)
                idx = list(arr_data["names"]).index(name)

                base_curve = [max(0, float(row["alpha_base"]) + float(row["alpha_cov"]) * c) for c in c_range]
                full_curve = []
                for c in c_range:
                    tm = solve_tminus(arr_data["e"], arr_data["pop"], res["Tplus_star"])
                    prefs = preference_values(arr_data["e"], arr_data["alpha_base"], arr_data["alpha_cov"],
                                              arr_data["alpha_trf"], float(c), res["Tplus_star"], tm)
                    full_curve.append(float(prefs[idx]))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=c_range, y=base_curve, name="Without transfers", line=dict(dash="dash", color="#bdc3c7")))
                fig.add_trace(go.Scatter(x=c_range, y=full_curve, name="With transfers", line=dict(color=color, width=2.5)))
                fig.add_hline(y=res["p_star"], line_dash="dot", line_color="#e74c3c",
                              annotation_text=f"Price €{res['p_star']:.0f}")
                fig.add_vline(x=res["c_star"], line_dash="dot", line_color="#27ae60",
                              annotation_text=f"c*={res['c_star']:.0%}")
                fig.update_layout(height=280, margin=dict(t=20, b=30, l=40, r=10),
                                  xaxis_title="Coverage", yaxis_title="€/ton",
                                  legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, use_container_width=True)

    # ── Net transfers chart ──
    st.divider()
    st.markdown("### International transfers")
    fig = px.bar(
        acct.sort_values("net_transfer_per_cap"), x="net_transfer_per_cap", y="name",
        orientation="h", color="status",
        color_discrete_map={"Contributor": "#e74c3c", "Beneficiary": "#27ae60"},
        labels={"net_transfer_per_cap": "€/capita/year", "name": ""},
    )
    fig.update_layout(height=380, showlegend=False, margin=dict(l=0))
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2: WHAT IF
# ═══════════════════════════════════════════════════════════════
elif page == "🔀 What If":
    st.title("What If")
    st.markdown("Each scenario reruns the full mechanism. MC context shows how the baseline compares.")

    tab1, tab2, tab3 = st.tabs(["Quick scenarios", "Drill into a group", "Custom edit"])

    with tab1:
        cols = st.columns(3)
        scenarios = [
            ("US withdraws", "UNITED STATES", {"alpha_base": 0, "alpha_cov": 0}),
            ("China more cautious", "CHINA", {"alpha_cov": lambda x: x*0.5}),
            ("Pro-climate US", "UNITED STATES", {"alpha_cov": 45}),
            ("Double transfers", None, {"alpha_trf": lambda x: x*2}),
            ("Fossil bloc excluded", "RUSSIA|HYDROCARBON", {"alpha_base": 0, "alpha_cov": 0, "alpha_trf": 0}),
            ("EU less ambitious", "EUROPEAN", {"alpha_base": 10}),
        ]
        for i, (label, pattern, changes) in enumerate(scenarios):
            with cols[i % 3]:
                if st.button(label, use_container_width=True):
                    temp = actors_base.copy()
                    if pattern:
                        m = temp["name"].str.contains(pattern, case=False)
                    else:
                        m = pd.Series([True]*len(temp))
                    for k, v in changes.items():
                        if callable(v):
                            temp.loc[m, k] = v(temp.loc[m, k])
                        else:
                            temp.loc[m, k] = v
                    st.session_state.actors = normalise_actor_df(temp)
                    st.session_state.scenario = label
                    st.rerun()

    with tab2:
        st.markdown("Groups combine countries with similar strategic types. Split one out to test if it behaves differently.")
        # Find groups that still exist (including residual groups)
        available_groups = {}
        for orig_name, member_list in MEMBERS.items():
            # Check for original or residual version of the group
            remaining_members = [m for m in member_list if m not in st.session_state.splits]
            if not remaining_members:
                continue
            # Find matching actor (original name or residual)
            match = actors[actors["name"].astype(str).str.contains(orig_name[:20], case=False, na=False)]
            if not match.empty:
                available_groups[orig_name] = remaining_members

        if not available_groups:
            st.info("All group members have been split out.")
        else:
            sel = st.selectbox("Select group", list(available_groups.keys()))
            remaining = available_groups[sel]
            member_data = countries_df[countries_df["country"].isin(remaining)]
            if not member_data.empty:
                st.dataframe(member_data[["country","emissions_cap","population_m","gdp_cap","headline_price","notes"]],
                             use_container_width=True, hide_index=True)
            split_who = st.selectbox("Split out", remaining)
            if st.button(f"Split {split_who}", type="primary"):
                updated = split_country_from_group(st.session_state.actors, countries_df, split_who)
                if len(updated) > len(st.session_state.actors):
                    st.session_state.actors = updated
                    st.session_state.splits.append(split_who)
                    st.session_state.scenario = f"Split: {', '.join(st.session_state.splits)}"
                    st.rerun()
                else:
                    st.warning(f"{split_who} may already be split out or could not be found in the group.")

    with tab3:
        edited = st.data_editor(actors[["name","alpha_base","alpha_cov","alpha_trf","e","pop_m"]],
                                num_rows="fixed", use_container_width=True,
                                column_config={"name": st.column_config.TextColumn("Actor", disabled=True)})
        if st.button("Apply and rerun", type="primary"):
            temp = actors.copy()
            for col in edited.columns: temp[col] = edited[col]
            st.session_state.actors = normalise_actor_df(temp)
            st.session_state.scenario = "Custom"
            st.rerun()

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"€{res['p_star']:.2f}")
    c2.metric("Coverage", f"{res['actual_coverage']:.0%}")
    c3.metric("T+", f"{res['Tplus_star']:.2f}")
    c4.metric("T−", f"{res['Tminus_actual']:.2f}")


# ═══════════════════════════════════════════════════════════════
# PAGE 3: THE MECHANISM
# ═══════════════════════════════════════════════════════════════
elif page == "⚙️ The Mechanism":
    st.title("How Themis Chooses")
    st.markdown("""
    **The mechanism optimises three things simultaneously:** price (p), coverage (c), and transfer rate (T+).
    T− is then solved so the pool balances. The objective is to maximise **c × p** — total climate impact.
    """)

    # 1. Feasible price curve
    curve = res["curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["coverage"], y=curve["feasible_price"], name="Feasible price p(c)", line=dict(color="#2c3e50", width=2)))
    fig.add_trace(go.Scatter(x=curve["coverage"], y=curve["objective"], name="c × p(c)", line=dict(color="#e67e22", width=2, dash="dash")))
    fig.add_vline(x=res["c_star"], line_dash="dot", line_color="#27ae60", annotation_text=f"c*={res['c_star']:.0%}")
    fig.add_hline(y=res["p_star"], line_dash="dot", line_color="#e74c3c", annotation_text=f"p*=€{res['p_star']:.1f}")
    fig.update_layout(height=420, xaxis_title="Coverage", yaxis_title="€/ton", title="Quantile price curve and objective")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("As coverage rises, the price must drop to include less-willing actors. The peak of c×p is the mechanism's choice.")

    # 2. T+ frontier
    frontier = res["frontier_by_Tplus"]
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.line(frontier, x="Tplus", y="objective", title="How T+ affects total climate impact",
                        labels={"Tplus": "T+ (contributor rate)", "objective": "c × p"})
        fig2.add_vline(x=res["Tplus_star"], line_dash="dot", line_color="red", annotation_text=f"T*={res['Tplus_star']:.2f}")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("T+ is optimised — not set by hand. The mechanism finds the transfer level that maximises coalition impact.")

    with col2:
        fig3 = px.line(frontier, x="Tplus", y=["p", "c"], title="Price and coverage vs transfer rate",
                        labels={"Tplus": "T+", "value": "", "variable": ""})
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Higher transfers bring in more actors (coverage rises) but lower the price they'll accept.")

    # 3. MC distributions
    st.divider()
    st.markdown("### Robustness: 500 Monte Carlo runs")
    st.markdown("Every preference parameter perturbed within its Data Bible uncertainty range.")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(mc_df, x="p_star", nbins=40, title="Price distribution",
                           labels={"p_star": "€/ton"}, color_discrete_sequence=["#2c3e50"])
        fig.add_vline(x=res["p_star"], line_dash="dash", line_color="red", annotation_text="Baseline")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(mc_df, x="actual_coverage", nbins=40, title="Coverage distribution",
                           labels={"actual_coverage": "Coverage"}, color_discrete_sequence=["#27ae60"])
        fig.add_vline(x=res["actual_coverage"], line_dash="dash", line_color="red", annotation_text="Baseline")
        st.plotly_chart(fig, use_container_width=True)

    # Join probabilities
    st.markdown("### Join reliability")
    fig = px.bar(join_probs.sort_values("Join probability"), x="Join probability", y="Actor",
                 orientation="h", color="Join probability",
                 color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"], range_color=[0,1])
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Actors above 90% are reliable joiners. Actors below 50% are likely non-joiners. The middle ground is where assumptions matter most.")


# ═══════════════════════════════════════════════════════════════
# PAGE 4: AUDIT
# ═══════════════════════════════════════════════════════════════
elif page == "✅ Audit":
    st.title("Audit Trail")
    diag = diagnostics(actors, res)
    for _, row in diag.iterrows():
        icon = "✅" if row["Status"] == "PASS" else "❌"
        st.markdown(f"{icon} **{row['Check']}** — {row['Detail']}")

    st.divider()
    st.markdown("### Full transfer accounting")
    st.dataframe(acct[["name","status","e","pop_m","collected_per_cap","sent_per_cap","received_per_cap",
                        "net_transfer_per_cap","total_sent_mEUR","total_received_mEUR"]],
                 use_container_width=True, hide_index=True)
    balance = acct["total_sent_mEUR"].sum() - acct["total_received_mEUR"].sum()
    st.metric("Pool balance", f"€{balance:,.4f}M")

    st.divider()
    st.markdown("### Data sources")
    st.markdown("""
    | Data | Source |
    |---|---|
    | Emissions | EDGAR 2025 JRC |
    | Population / GDP | World Bank 2024 |
    | Carbon pricing | OECD ECR 2025, ICAP, EU Commission, Climate Action Tracker |
    | α_base | OECD Net ECR / explicit carbon prices (A/B) |
    | α_cov | 70% coverage thought experiment (C — tested via MC) |
    | α_trf | k/GDP_cap, k=20,000, capped at 20 (B/C) |
    """)
