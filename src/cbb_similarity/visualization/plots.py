"""Plotly figures for archetype maps and team comparison radars.

Functions return a ``go.Figure``; callers decide whether to ``.show()`` it or
write it to disk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_archetypes(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    x_title: str = "",
    y_title: str = "",
    title: str = "Team Archetypes",
    hover_cols: Sequence[str] | None = None,
) -> go.Figure:
    """Scatter plot of team-seasons on PCA axes, coloured by archetype."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=list(hover_cols or []),
        title=title,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.6), selector=dict(mode="markers"))
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="Archetype",
        title_x=0.5,
        template="simple_white",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1)

    # Quadrant lines through the origin of the (centred) PCA space
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig


def plot_radar_comparison(
    comps: pd.DataFrame,
    metrics: Sequence[str],
    stat_labels: Mapping[str, str] | None = None,
    n_teams: int = 3,
    title: str | None = None,
) -> go.Figure:
    """Radar chart of season percentiles for the first ``n_teams`` rows of a comps table.

    Parameters
    ----------
    comps
        Output of :func:`cbb_similarity.models.similarity.generate_comps_table`.
    metrics
        Metric columns to plot as radar axes.
    stat_labels
        Optional metric key -> display label mapping.
    n_teams
        Number of teams to overlay (target team plus ``n_teams - 1`` comps).
    title
        Figure title; defaults to ``"<target team> Comparison"``.
    """
    stat_labels = stat_labels or {}
    teams = comps.head(n_teams)
    theta = [stat_labels.get(m, m) for m in metrics]

    fig = go.Figure()
    for _, row in teams.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=row[list(metrics)].astype(float).to_numpy(),
                theta=theta,
                fill="toself",
                fillcolor=row["hexColor1"],
                opacity=0.5,
                name=row["Team"],
                line=dict(color=row["hexColor2"], width=2),
            )
        )

    fig.update_traces(hovertemplate="Percentile: %{r:.2f}<br>Stat: %{theta}")
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=title or f"{teams['Team'].iloc[0]} Comparison",
        title_x=0.5,
    )
    return fig
