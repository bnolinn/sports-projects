"""
MLB Matchup Dashboard — Statcast pitch-shape matchup model.

    streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matchup_engine import ( 
    PITCH_NAMES, MatchupEngine, available_years, edge_label, load_season, scouting_notes,
)

st.set_page_config(page_title="MLB Matchup Lab", layout="wide")

BAT_COLOR = "#d9480f"
PIT_COLOR = "#1c7ed6"
DIVERGING = [[0.0, PIT_COLOR], [0.5, "#f1f3f5"], [1.0, BAT_COLOR]]
PITCH_COLORS = {
    "FF": "#e03131", "SI": "#f08c00", "FC": "#9c36b5", "SL": "#1971c2", "ST": "#0c8599",
    "SV": "#5c7cfa", "CU": "#2f9e44", "KC": "#66a80f", "CS": "#94d82d", "CH": "#e64980",
    "FS": "#ae3ec9", "FO": "#862e9c", "SC": "#495057", "KN": "#868e96", "EP": "#adb5bd",
}
DEFAULT_TEAMS = {2026: ("TB", "MIL"), 2025: ("LAD", "TOR"), 2024: ("LAD", "NYY"), 2023: ("TEX", "AZ")}


# Cached loaders
@st.cache_resource(show_spinner="Loading Statcast season and building league pitch pools…")
def get_engine(year: int) -> MatchupEngine:
    df, names = load_season(year)
    return MatchupEngine(df, names)


@st.cache_data(show_spinner="Running matchup model…")
def get_grid(year: int, batters: tuple, pitchers: tuple) -> pd.DataFrame:
    return get_engine(year).grid(list(batters), list(pitchers))


@st.cache_data
def get_roster(year: int, team: str, role: str, min_n: int) -> pd.DataFrame:
    return get_engine(year).roster(team, role, min_n)


# render helpers
def fmt_rv(x: float) -> str:
    return f"{x:+.2f}"


def edge_badge(rv100: float) -> str:
    lab = edge_label(rv100)
    color = BAT_COLOR if rv100 > 0.4 else PIT_COLOR if rv100 < -0.4 else "#868e96"
    return (f"<span style='background:{color};color:white;padding:4px 10px;border-radius:12px;"
            f"font-weight:600'>{lab}</span>")


def strike_zone_shapes():
    return [dict(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                 line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)")]


def heat_grid(grid: pd.DataFrame, title: str) -> go.Figure:
    piv = grid.pivot(index="Batter", columns="Pitcher", values="rv100")
    conf = grid.pivot(index="Batter", columns="Pitcher", values="confidence")
    bb = grid.pivot(index="Batter", columns="Pitcher", values="batter_best")
    pb = grid.pivot(index="Batter", columns="Pitcher", values="pitcher_best")
    # Keep roster order
    piv = piv.loc[grid["Batter"].unique(), grid["Pitcher"].unique()]
    conf, bb, pb = (t.loc[piv.index, piv.columns] for t in (conf, bb, pb))
    lim = max(2.0, float(np.nanmax(np.abs(piv.values))))
    custom = np.dstack([conf.values, bb.values, pb.values])
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index, zmin=-lim, zmax=lim, colorscale=DIVERGING,
        text=np.vectorize(fmt_rv)(piv.values), texttemplate="%{text}", textfont=dict(size=11),
        customdata=custom,
        hovertemplate=("<b>%{y}</b> vs <b>%{x}</b><br>Edge: %{z:+.2f} RV/100<br>"
                       "Confidence: %{customdata[0]:.0%}<br>Hitter's best pitch: %{customdata[1]}<br>"
                       "Pitcher's best weapon: %{customdata[2]}<extra></extra>"),
        colorbar=dict(title="RV/100<br>(+ batter)"),
    ))
    fig.update_layout(title=dict(text=title, y=0.99, yanchor="top"), height=190 + 34 * len(piv.index),
                      margin=dict(l=10, r=10, t=150, b=10),
                      xaxis=dict(side="top", tickangle=-35), yaxis=dict(autorange="reversed"))
    return fig


# Sidebar
years = available_years()
if not years:
    st.error("No data found. Run `python build_data.py --year 2025` first.")
    st.stop()

with st.sidebar:
    st.title("Matchup Lab")
    year = st.selectbox("Season", years[::-1], index=0)
    page = st.radio("View", ["Team vs Team", "Matchup Breakdown", "Pitcher Game Plan", "Hitter Scouting Report"],
                    key="page")
    st.caption("Model: pitch-shape similarity kernel + Bayesian shrinkage on count-neutral run value. "
               "Positive RV/100 = batter edge, negative = pitcher edge.")

eng = get_engine(year)
teams = eng.teams()
dir_df = eng.directory()
hitters = dir_df[(dir_df.role == "batter") & (dir_df.n >= 50)]
pitchers = dir_df[(dir_df.role == "pitcher") & (dir_df.n >= 50)]
hit_label = dict(zip(hitters.id, hitters.label))
pit_label = dict(zip(pitchers.id, pitchers.label))
dA, dB = DEFAULT_TEAMS.get(year, (teams[0], teams[1]))


def goto_matchup(bid: int, pid: int):
    st.session_state["mb_batter"] = bid
    st.session_state["mb_pitcher"] = pid
    st.session_state["page"] = "Matchup Breakdown"


# Page: Team vs Team
def page_team_vs_team():
    st.header("Team vs Team Matchup Matrix")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    team_a = c1.selectbox("Team A", teams, index=teams.index(dA) if dA in teams else 0)
    team_b = c2.selectbox("Team B", teams, index=teams.index(dB) if dB in teams else 1)
    min_pa = c3.number_input("Min PA (hitters)", 50, 600, 150, step=25)
    min_p = c4.number_input("Min pitches (pitchers)", 50, 2000, 300, step=50)
    if team_a == team_b:
        st.info("Pick two different teams. To see one hitter against one pitcher, use Matchup Breakdown.")
        return

    def pick(team, role, n_default, key):
        r = get_roster(year, team, role, min_pa if role == "batter" else min_p)
        if role == "pitcher":
            labels = {i: f"{n} ({h}HP, {ro})" for i, n, h, ro in zip(r.id, r.name, r.hand, r.role)}
        else:
            labels = dict(zip(r.id, r.name))
        return st.multiselect(f"{team} {'hitters' if role == 'batter' else 'pitchers'}",
                              list(labels), default=list(labels)[:n_default],
                              format_func=lambda i: labels[i], key=key)

    with st.expander("Choose lineups and staffs", expanded=False):
        e1, e2 = st.columns(2)
        with e1:
            a_bat = pick(team_a, "batter", 10, f"abat_{team_a}_{year}")
            a_pit = pick(team_a, "pitcher", 12, f"apit_{team_a}_{year}")
        with e2:
            b_bat = pick(team_b, "batter", 10, f"bbat_{team_b}_{year}")
            b_pit = pick(team_b, "pitcher", 12, f"bpit_{team_b}_{year}")

    if not (a_bat and b_pit and b_bat and a_pit):
        st.info("Select at least one hitter and pitcher on each side.")
        return

    g_ab = get_grid(year, tuple(a_bat), tuple(b_pit))
    g_ba = get_grid(year, tuple(b_bat), tuple(a_pit))

    # Headline numbers
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{team_a} bats", fmt_rv(g_ab.rv100.mean()), help=f"{team_a} hitters vs {team_b} pitchers, avg RV/100")
    m2.metric(f"{team_b} bats", fmt_rv(g_ba.rv100.mean()), help=f"{team_b} hitters vs {team_a} pitchers, avg RV/100")
    net = g_ab.rv100.mean() - g_ba.rv100.mean()
    m3.metric("Net edge", f"{team_a if net > 0 else team_b} +{abs(net):.2f}")
    m4.metric("Avg confidence", f"{pd.concat([g_ab, g_ba]).confidence.mean():.0%}")

    t1, t2 = st.tabs([f"{team_a} hitters vs {team_b} pitchers", f"{team_b} hitters vs {team_a} pitchers"])
    for tab, g, bat_team, pit_team in [(t1, g_ab, team_a, team_b), (t2, g_ba, team_b, team_a)]:
        with tab:
            st.plotly_chart(heat_grid(g, f"{bat_team} hitters (rows) vs {pit_team} pitchers (cols)"),
                            use_container_width=True)
            s1, s2 = st.columns(2)
            with s1:
                st.subheader(f"=Best spots for {bat_team} hitters")
                top = g.sort_values("rv100", ascending=False).head(8)
                st.dataframe(top[["Batter", "Pitcher", "rv100", "batter_best", "confidence"]].rename(columns={
                    "rv100": "Edge", "batter_best": "Hunt this pitch", "confidence": "Conf."}).assign(**{"Conf.": lambda d: d["Conf."] * 100}),
                    hide_index=True, use_container_width=True,
                    column_config={"Edge": st.column_config.NumberColumn(format="%+.2f"),
                                   "Conf.": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")})
            with s2:
                st.subheader(f"Best spots for {pit_team} pitchers")
                bot = g.sort_values("rv100").head(8)
                st.dataframe(bot[["Pitcher", "Batter", "rv100", "pitcher_best", "confidence"]].rename(columns={
                    "rv100": "Edge", "pitcher_best": "Lean on", "confidence": "Conf."}).assign(**{"Conf.": lambda d: d["Conf."] * 100}),
                    hide_index=True, use_container_width=True,
                    column_config={"Edge": st.column_config.NumberColumn(format="%+.2f"),
                                   "Conf.": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")})
            s3, s4 = st.columns(2)
            with s3:
                st.markdown(f"**{bat_team} hitters — avg vs this staff**")
                bsum = g.groupby("Batter", sort=False).agg(Edge=("rv100", "mean"), Best=("rv100", "max"),
                                                           Worst=("rv100", "min")).sort_values("Edge", ascending=False)
                st.dataframe(bsum.style.format("{:+.2f}").background_gradient(cmap="RdBu_r", vmin=-2, vmax=2),
                             use_container_width=True)
            with s4:
                st.markdown(f"**{pit_team} pitchers — avg vs this lineup** (lower = better for pitcher)")
                psum = g.groupby("Pitcher", sort=False).agg(Edge=("rv100", "mean"), Toughest=("rv100", "max"),
                                                            Easiest=("rv100", "min")).sort_values("Edge")
                st.dataframe(psum.style.format("{:+.2f}").background_gradient(cmap="RdBu_r", vmin=-2, vmax=2),
                             use_container_width=True)
            st.markdown("**Drill into a matchup**")
            opts = list(zip(g.batter, g.pitcher))
            sel = st.selectbox("Matchup", opts, key=f"drill_{bat_team}_{pit_team}",
                               format_func=lambda bp: f"{eng.name(bp[0])} vs {eng.name(bp[1])}")
            st.button("Open full breakdown →", key=f"go_{bat_team}_{pit_team}",
                      on_click=goto_matchup, args=sel)


# Page: Matchup Breakdown
def player_pickers():
    h_ids, p_ids = list(hit_label), list(pit_label)
    if st.session_state.get("mb_batter") not in hit_label:
        st.session_state["mb_batter"] = h_ids[0]
    if st.session_state.get("mb_pitcher") not in pit_label:
        st.session_state["mb_pitcher"] = p_ids[0]
    c1, c2 = st.columns(2)
    bid = c1.selectbox("Batter", h_ids, format_func=hit_label.get, key="mb_batter")
    pid = c2.selectbox("Pitcher", p_ids, format_func=pit_label.get, key="mb_pitcher")
    return bid, pid


def page_matchup():
    st.header("Matchup Breakdown")
    bid, pid = player_pickers()
    m = eng.matchup(bid, pid)
    if not m.pitches:
        st.warning("Not enough data for this pitcher.")
        return
    bname, pname = eng.name(bid), eng.name(pid)
    loc_edge = eng.location_edge(bid, pid)
    h2h = eng.head_to_head(bid, pid)
    h2h_pa = h2h.groupby(["game_pk", "at_bat_number"]).ngroups if len(h2h) else 0

    st.markdown(f"### {bname} ({m.stand}HH) vs {pname} ({m.p_throws}HP) &nbsp; {edge_badge(m.rv100)}",
                unsafe_allow_html=True)
    k = st.columns(3) + st.columns(3)
    k[0].metric("Matchup RV/100", fmt_rv(m.rv100), help="Usage-weighted expected count-neutral run value per 100 pitches, batter view.")
    k[1].metric("Proj. Whiff%", f"{m.weighted('whiff'):.1%}",
                delta=f"{(m.weighted('whiff') - m.weighted('whiff', 'baseline')) * 100:+.1f} vs lg", delta_color="inverse")
    k[2].metric("Proj. Chase%", f"{m.weighted('chase'):.1%}",
                delta=f"{(m.weighted('chase') - m.weighted('chase', 'baseline')) * 100:+.1f} vs lg", delta_color="inverse")
    k[3].metric("Proj. xwOBAcon", f"{m.weighted('xwoba_con'):.3f}",
                delta=f"{(m.weighted('xwoba_con') - m.weighted('xwoba_con', 'baseline')) * 1000:+.0f} pts vs lg")
    k[4].metric("Location fit", fmt_rv(loc_edge), help="Where this pitcher locates vs this side × where the hitter does damage (RV/100).")
    k[5].metric("Confidence", f"{m.confidence:.0%}", help="Share of the batter estimate driven by his own data vs league prior.")

    a, b = st.columns(2)
    bat_notes, pit_notes = scouting_notes(eng, m)
    with a:
        st.markdown(f"#### {bname} advantages")
        for n in bat_notes or ["No clear edges for the hitter."]:
            st.markdown(f"- {n}")
    with b:
        st.markdown(f"#### {pname} advantages")
        for n in pit_notes or ["No clear edges for the pitcher."]:
            st.markdown(f"- {n}")

    st.markdown("#### Pitch-by-pitch model")
    rows = []
    for p in m.pitches:
        rows.append({
            "Pitch": PITCH_NAMES.get(p.pitch_type, p.pitch_type), "Usage": p.usage * 100,
            "Velo": p.shape["velo"], "IVB": p.shape["ivb"], "HB (arm+)": p.shape["hb"],
            "Spin": p.shape["spin"], "Ext": p.shape["ext"],
            "Stuff baseline": -p.baseline["rv"] * 100,
            "Pitcher vs stuff": -(p.pitcher["rv"] - p.baseline["rv"]) * 100,
            "Batter vs shape": (p.batter_on_similar["rv"] - p.baseline["rv"]) * 100,
            "Expected RV/100": p.edge_rv100,
            "Whiff% exp": p.expected["whiff"] * 100, "Whiff% lg": p.baseline["whiff"] * 100,
            "xwOBAcon exp": p.expected["xwoba_con"], "xwOBAcon lg": p.baseline["xwoba_con"],
            "Similar seen": p.n_eff_batter,
        })
    tbl = pd.DataFrame(rows)
    st.dataframe(
        tbl, hide_index=True, use_container_width=True,
        column_config={
            "Usage": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "Velo": st.column_config.NumberColumn(format="%.1f"),
            "IVB": st.column_config.NumberColumn(format="%+.1f\""),
            "HB (arm+)": st.column_config.NumberColumn(format="%+.1f\""),
            "Spin": st.column_config.NumberColumn(format="%.0f"),
            "Ext": st.column_config.NumberColumn(format="%.1f"),
            "Stuff baseline": st.column_config.NumberColumn(format="%+.2f", help="Pitcher RV/100 an average arm gets with this shape vs this side (+ = good for pitcher)."),
            "Pitcher vs stuff": st.column_config.NumberColumn(format="%+.2f", help="How much better (+) this pitcher's results are than the shape predicts, shrunk."),
            "Batter vs shape": st.column_config.NumberColumn(format="%+.2f", help="Hitter's RV/100 vs similar pitches relative to league (+ = hitter better)."),
            "Expected RV/100": st.column_config.NumberColumn(format="%+.2f", help="Final projection, batter view."),
            "Whiff% exp": st.column_config.NumberColumn(format="%.1f%%"),
            "Whiff% lg": st.column_config.NumberColumn(format="%.1f%%"),
            "xwOBAcon exp": st.column_config.NumberColumn(format="%.3f"),
            "xwOBAcon lg": st.column_config.NumberColumn(format="%.3f"),
            "Similar seen": st.column_config.NumberColumn(format="%.0f", help="Effective # of physically similar pitches the hitter has seen."),
        })

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(movement_figure(bid, m), use_container_width=True)
    with c2:
        fig = go.Figure()
        order = sorted(m.pitches, key=lambda p: p.edge_rv100)
        fig.add_bar(
            y=[PITCH_NAMES.get(p.pitch_type, p.pitch_type) for p in order],
            x=[p.edge_rv100 for p in order], orientation="h",
            marker_color=[BAT_COLOR if p.edge_rv100 > 0 else PIT_COLOR for p in order],
            text=[f"{p.edge_rv100:+.2f} ({p.usage:.0%})" for p in order], textposition="inside", insidetextanchor="middle",
        )
        fig.update_layout(title="Expected RV/100 by pitch (usage in parentheses)", height=420,
                          xaxis_title="← pitcher edge | batter edge →", margin=dict(l=10, r=10, t=50, b=10))
        fig.add_vline(x=0, line_color="black")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Location: where he throws vs where the hitter hurts you")
    pt_opts = [None] + [p.pitch_type for p in m.pitches]
    pt_sel = st.radio("Pitch", pt_opts, horizontal=True,
                      format_func=lambda t: "All pitches" if t is None else PITCH_NAMES.get(t, t))
    z1, z2 = st.columns(2)
    dmg = eng.batter_damage_map(bid, m.p_throws)
    lim = max(3.0, float(np.nanmax(np.abs(dmg))))
    f1 = go.Figure(go.Heatmap(z=dmg, x=eng.XS, y=eng.ZS, colorscale=DIVERGING, zmin=-lim, zmax=lim,
                              colorbar=dict(title="RV/100"), hovertemplate="x %{x:.2f} z %{y:.2f}: %{z:+.1f}<extra></extra>"))
    f1.update_layout(title=f"{bname} damage map vs {m.p_throws}HP (catcher view)", shapes=strike_zone_shapes(),
                     height=460, xaxis=dict(range=[-1.6, 1.6]), yaxis=dict(scaleanchor="x", range=[0.8, 4.2]), margin=dict(l=10, r=10, t=50, b=10))
    z1.plotly_chart(f1, use_container_width=True)
    loc = eng.pitcher_location_map(pid, m.stand, pt_sel)
    f2 = go.Figure(go.Heatmap(z=loc, x=eng.XS, y=eng.ZS, colorscale="Greys", showscale=False,
                              hovertemplate="x %{x:.2f} z %{y:.2f}<extra></extra>"))
    f2.add_contour(z=dmg, x=eng.XS, y=eng.ZS, contours=dict(start=1, end=lim, size=1.5, coloring="lines"),
                   colorscale=[[0, BAT_COLOR], [1, BAT_COLOR]], showscale=False, line_width=2, name="hitter hot")
    f2.add_contour(z=-dmg, x=eng.XS, y=eng.ZS, contours=dict(start=1, end=lim, size=1.5, coloring="lines"),
                   colorscale=[[0, PIT_COLOR], [1, PIT_COLOR]], showscale=False, line_width=2, name="hitter cold")
    f2.update_layout(title=f"{pname} location vs {m.stand}HH (shade) — hitter hot/cold zones (lines)",
                     shapes=strike_zone_shapes(), height=460, xaxis=dict(range=[-1.6, 1.6]),
                     yaxis=dict(scaleanchor="x", range=[0.8, 4.2]), margin=dict(l=10, r=10, t=50, b=10))
    z2.plotly_chart(f2, use_container_width=True)

    t1, t2, t3 = st.tabs(["Pitcher arsenal (vs this side)", "Hitter vs pitch types", f"Head-to-head ({h2h_pa} PA)"])
    with t1:
        st.dataframe(eng.arsenal(pid, m.stand).style.format({
            "Usage": "{:.1%}", "Velo": "{:.1f}", "Spin": "{:.0f}", "IVB": "{:+.1f}", "HB (arm+)": "{:+.1f}",
            "Ext": "{:.1f}", "Whiff%": "{:.1%}", "CSW%": "{:.1%}", "Chase%": "{:.1%}",
            "xwOBAcon": "{:.3f}", "RV/100 (pit)": "{:+.2f}"}, na_rep="—"), use_container_width=True)
    with t2:
        st.dataframe(eng.batter_vs_types(bid, m.p_throws).style.format({
            "Swing%": "{:.1%}", "Whiff%": "{:.1%}", "Chase%": "{:.1%}", "xwOBAcon": "{:.3f}",
            "HardHit%": "{:.1%}", "EV": "{:.1f}", "RV/100 (bat)": "{:+.2f}"}, na_rep="—")
            .background_gradient(subset=["RV/100 (bat)"], cmap="RdBu_r", vmin=-5, vmax=5),
            use_container_width=True)
        st.caption(f"All {m.p_throws}HP faced. RV/100 is count-neutral, batter perspective.")
    with t3:
        if h2h.empty:
            st.info("These two have never faced each other in this season's data.")
        else:
            pa = h2h[h2h["events"].notna()][["game_date", "game_type", "pitch_name", "release_speed", "balls",
                                             "strikes", "events", "launch_speed", "launch_angle",
                                             "estimated_woba_using_speedangle"]]
            pa.columns = ["Date", "Type", "Final pitch", "Velo", "B", "S", "Result", "EV", "LA", "xwOBA"]
            st.dataframe(pa, hide_index=True, use_container_width=True)
            st.caption(f"{len(h2h)} pitches. Small samples — the model above leans on pitch-shape evidence instead.")


def movement_figure(bid: int, m) -> go.Figure:
    mv = eng.batter_movement_map(bid, m.p_throws)
    lim = max(3.0, float(np.nanmax(np.abs(mv))))
    fig = go.Figure(go.Heatmap(z=mv, x=eng.HBS, y=eng.IVBS, colorscale=DIVERGING, zmin=-lim, zmax=lim,
                               opacity=0.85, colorbar=dict(title="Hitter<br>RV/100"),
                               hovertemplate="HB %{x:.0f}\" IVB %{y:.0f}\": %{z:+.1f}<extra></extra>"))
    p_all = eng.pitcher_df(m.pitcher)
    for p in m.pitches:
        s = p_all[p_all["pitch_type"] == p.pitch_type]
        col = PITCH_COLORS.get(p.pitch_type, "#000")
        nm = PITCH_NAMES.get(p.pitch_type, p.pitch_type)
        fig.add_scatter(x=s["hb_arm"].sample(min(len(s), 150), random_state=0),
                        y=s["ivb"].sample(min(len(s), 150), random_state=0), mode="markers",
                        marker=dict(color=col, size=4, opacity=0.35), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=[p.shape["hb"]], y=[p.shape["ivb"]], mode="markers+text", name=nm,
                        marker=dict(color=col, size=10 + 40 * p.usage, line=dict(color="black", width=1.5)),
                        text=[nm], textposition="top center",
                        hovertemplate=(f"<b>{nm}</b><br>{p.shape['velo']:.1f} mph · {p.shape['spin']:.0f} rpm<br>"
                                       f"Usage {p.usage:.0%}<br>Expected {p.edge_rv100:+.2f} RV/100<extra></extra>"))
    fig.update_layout(title=f"Movement map: {eng.name(m.batter)} RV/100 by pitch shape, with {eng.name(m.pitcher)}'s arsenal",
                      xaxis_title="Horizontal break, arm-side + (in)", yaxis_title="Induced vertical break (in)",
                      height=480, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig


# Page: Pitcher game plan
def plan_rows(pairs):
    rows = []
    for bid, pid in pairs:
        m = eng.matchup(bid, pid)
        if not m.pitches:
            continue
        by_rv = sorted(m.pitches, key=lambda p: p.edge_rv100)
        by_wh = max(m.pitches, key=lambda p: p.expected["whiff"] - p.baseline["whiff"])
        bat, pit = scouting_notes(eng, m)
        rows.append({
            "batter": bid, "pitcher": pid, "Batter": eng.name(bid), "Pitcher": eng.name(pid),
            "Side": f"{m.stand}HH v {m.p_throws}HP", "Edge": m.rv100, "Verdict": edge_label(m.rv100),
            "Attack with": PITCH_NAMES.get(by_rv[0].pitch_type, by_rv[0].pitch_type),
            "Avoid": PITCH_NAMES.get(by_rv[-1].pitch_type, by_rv[-1].pitch_type) if by_rv[-1].edge_rv100 > 0 else "—",
            "Put-away pitch": PITCH_NAMES.get(by_wh.pitch_type, by_wh.pitch_type),
            "Proj Whiff%": m.weighted("whiff") * 100, "Proj xwOBAcon": m.weighted("xwoba_con"),
            "Location fit": eng.location_edge(bid, pid), "Conf.": m.confidence * 100,
            "Key note": (pit or bat or [""])[0].replace("**", ""),
        })
    return pd.DataFrame(rows)


PLAN_COLS = {
    "Edge": st.column_config.NumberColumn(format="%+.2f", help="RV/100, + = batter edge"),
    "Proj Whiff%": st.column_config.NumberColumn(format="%.1f%%"),
    "Proj xwOBAcon": st.column_config.NumberColumn(format="%.3f"),
    "Location fit": st.column_config.NumberColumn(format="%+.2f"),
    "Conf.": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
}


def page_pitcher_plan():
    st.header("Pitcher Game Plan")
    c1, c2, c3 = st.columns([2, 1, 1])
    pid = c1.selectbox("Pitcher", list(pit_label), format_func=pit_label.get, key="gp_pitcher")
    opp = c2.selectbox("Opponent", teams, index=teams.index(dA) if dA in teams else 0, key="gp_opp")
    min_pa = c3.number_input("Min PA", 25, 600, 150, step=25, key="gp_minpa")
    r = get_roster(year, opp, "batter", min_pa)
    lineup = st.multiselect("Lineup", list(r.id), default=list(r.id)[:9], format_func=eng.name, key=f"gp_lu_{opp}")
    if not lineup:
        return
    df = plan_rows([(b, pid) for b in lineup])
    st.metric(f"{eng.name(pid)} vs {opp} lineup", fmt_rv(df.Edge.mean()) + " RV/100", edge_label(df.Edge.mean()),
              delta_color="off")
    st.dataframe(df.drop(columns=["batter", "pitcher", "Pitcher"]).sort_values("Edge"),
                 hide_index=True, use_container_width=True, column_config=PLAN_COLS)
    st.markdown("**Arsenal overview**")
    st.dataframe(eng.arsenal(pid).style.format({
        "Usage": "{:.1%}", "Velo": "{:.1f}", "Spin": "{:.0f}", "IVB": "{:+.1f}", "HB (arm+)": "{:+.1f}",
        "Ext": "{:.1f}", "Whiff%": "{:.1%}", "CSW%": "{:.1%}", "Chase%": "{:.1%}",
        "xwOBAcon": "{:.3f}", "RV/100 (pit)": "{:+.2f}"}, na_rep="—"), use_container_width=True)
    sel = st.selectbox("Open breakdown", lineup, format_func=eng.name, key="gp_open")
    st.button("Open full breakdown →", on_click=goto_matchup, args=(sel, pid), key="gp_go")


# Page: Hitter scouting report
def page_hitter_report():
    st.header("Hitter Scouting Report")
    c1, c2, c3 = st.columns([2, 1, 1])
    bid = c1.selectbox("Hitter", list(hit_label), format_func=hit_label.get, key="hr_batter")
    opp = c2.selectbox("Opposing staff", teams, index=teams.index(dB) if dB in teams else 0, key="hr_opp")
    min_p = c3.number_input("Min pitches", 50, 2000, 300, step=50, key="hr_minp")
    r = get_roster(year, opp, "pitcher", min_p)
    staff = st.multiselect("Pitchers", list(r.id), default=list(r.id)[:12], format_func=eng.name, key=f"hr_st_{opp}")
    if not staff:
        return
    rows = []
    for pid in staff:
        m = eng.matchup(bid, pid)
        if not m.pitches:
            continue
        by_rv = sorted(m.pitches, key=lambda p: -p.edge_rv100)
        rows.append({
            "pitcher": pid, "Pitcher": eng.name(pid), "Side": f"{m.stand}HH v {m.p_throws}HP",
            "Edge": m.rv100, "Verdict": edge_label(m.rv100),
            "Hunt": PITCH_NAMES.get(by_rv[0].pitch_type, by_rv[0].pitch_type) if by_rv[0].edge_rv100 > 0 else "—",
            "Lay off / beware": PITCH_NAMES.get(by_rv[-1].pitch_type, by_rv[-1].pitch_type),
            "Proj Whiff%": m.weighted("whiff") * 100, "Proj xwOBAcon": m.weighted("xwoba_con"),
            "Location fit": eng.location_edge(bid, pid), "Conf.": m.confidence * 100,
        })
    df = pd.DataFrame(rows).sort_values("Edge", ascending=False)
    st.metric(f"{eng.name(bid)} vs {opp} staff", fmt_rv(df.Edge.mean()) + " RV/100", edge_label(df.Edge.mean()),
              delta_color="off")
    st.dataframe(df.drop(columns=["pitcher"]), hide_index=True, use_container_width=True, column_config=PLAN_COLS)

    st.markdown("**Hitter profile by pitch type**")
    hand = st.radio("vs", ["R", "L"], horizontal=True, format_func=lambda h: f"{h}HP", key="hr_hand")
    st.dataframe(eng.batter_vs_types(bid, hand).style.format({
        "Swing%": "{:.1%}", "Whiff%": "{:.1%}", "Chase%": "{:.1%}", "xwOBAcon": "{:.3f}",
        "HardHit%": "{:.1%}", "EV": "{:.1f}", "RV/100 (bat)": "{:+.2f}"}, na_rep="—")
        .background_gradient(subset=["RV/100 (bat)"], cmap="RdBu_r", vmin=-5, vmax=5), use_container_width=True)
    sel = st.selectbox("Open breakdown", list(df.pitcher), format_func=eng.name, key="hr_open")
    st.button("Open full breakdown →", on_click=goto_matchup, args=(bid, sel), key="hr_go")


PAGES = {
    "Team vs Team": page_team_vs_team,
    "Matchup Breakdown": page_matchup,
    "Pitcher Game Plan": page_pitcher_plan,
    "Hitter Scouting Report": page_hitter_report,
}
PAGES[page]()

with st.expander("How the model works"):
    st.markdown("""
- **Run value** is Statcast's per-pitch change in run expectancy, *count-neutralized* (minus the league average for that ball-strike count), shown per 100 pitches from the hitter's view.
- **Stuff baseline**: for each pitch in the arsenal, every league pitch from a same-handed pitcher to a same-sided hitter is weighted by physical similarity (velo, IVB, arm-side break, spin, release height/side, extension). That gives what an average hitter does vs that exact shape.
- **Pitcher estimate**: his actual results with the pitch vs that side, shrunk toward the stuff baseline (Bayesian pseudo-counts).
- **Hitter estimate**: the same similarity kernel applied to every pitch the hitter saw — so a hitter who crushes 97 mph ride fastballs gets credit vs *any* pitch shaped like that, whatever it's called — also shrunk toward league.
- **Combined**: pitcher + hitter effects add (on the log-odds scale for whiff/chase), then get weighted by usage vs that side.
- **Location fit** overlays the pitcher's location density with the hitter's smoothed zone damage map.
- **Confidence** reflects how many effectively-similar pitches the hitter has seen; low-confidence cells are mostly league-prior.
""")
