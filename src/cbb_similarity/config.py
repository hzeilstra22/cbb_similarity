"""Project-wide configuration: file paths, metric groups and archetype model specs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
STATS_PATH = RAW_DATA_DIR / "combined_stats.csv"
DATA_DICTIONARY_PATH = DATA_DIR / "data_dictionary.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Column conventions
# ---------------------------------------------------------------------------
SEASON_COL = "competitionId"
SEASON_NAME_COL = "competitionName"
TEAM_ID_COL = "teamId"
TEAM_COL = "teamMarket"

ID_COLUMNS: tuple[str, ...] = (
    TEAM_ID_COL,
    SEASON_COL,
    TEAM_COL,
    "teamName",
    SEASON_NAME_COL,
    "hexColor1",
    "hexColor2",
)

# Display labels for metrics that are missing from (or unclear in) the data dictionary.
LABEL_OVERRIDES: dict[str, str] = {
    "tovPctAgst": "Opponent TOV%",
    "lane2FgPctAgst": "Opponent LANE2 FG%",
    "lane2FgaFreqAgst": "Opponent LANE2 FGA%",
    "fga3RateAgst": "Opponent 3PAr",
    "fg3PctAgst": "Opponent 3P%",
}

# ---------------------------------------------------------------------------
# Metric groups
# ---------------------------------------------------------------------------
OVERALL_METRICS: tuple[str, ...] = (
    "stlPct", "blkPct", "pfPct", "atr2FgaFreq", "atr2FgPct", "freqChncTr",
    "paint2FgPct", "paint2FgaFreq", "mid2FgPct", "mid2FgaFreq", "fga3Rate",
    "orbPct", "drbPct", "astPct", "ortg", "drtg", "pace", "ftaRate", "tovPct",
    "fg3Pct", "lane2FgPctAgst", "lane2FgaFreqAgst", "mid2FgPctAgst",
    "mid2FgaFreqAgst", "fga3RateAgst", "fg3PctAgst",
)  # fmt: skip

SHOOTING_METRICS: tuple[str, ...] = (
    "atr2FgaFreq", "atr2FgPct", "paint2FgPct", "paint2FgaFreq",
    "mid2FgPct", "mid2FgaFreq", "fga3Rate", "fg3Pct",
)  # fmt: skip

PLAYSTYLE_METRICS: tuple[str, ...] = ("astPct", "tovPct", "pace", "orbPct", "ftaRate")

DEFENSE_METRICS: tuple[str, ...] = (
    "blkPct", "stlPct", "drbPct", "pfPct", "tovPctAgst",
    "lane2FgPctAgst", "lane2FgaFreqAgst", "fga3RateAgst", "fg3PctAgst",
)  # fmt: skip

METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "overall": OVERALL_METRICS,
    "shooting": SHOOTING_METRICS,
    "playstyle": PLAYSTYLE_METRICS,
    "defense": DEFENSE_METRICS,
}

# Every metric used by any model; these are standardized and percentile-ranked.
ALL_METRICS: tuple[str, ...] = tuple(
    dict.fromkeys(m for group in METRIC_GROUPS.values() for m in group)
)


# ---------------------------------------------------------------------------
# Archetype (KMeans) model specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchetypeSpec:
    """Configuration for one KMeans archetype model.

    ``cluster_names`` maps KMeans label ids to human-readable names. The ids
    depend on the data and ``random_state``, so if the input data changes,
    re-check the cluster profiles written by the pipeline before trusting
    the names.
    """

    name: str
    metrics: tuple[str, ...]
    cluster_names: dict[int, str]
    random_state: int
    n_clusters: int = 4
    plot_title: str = ""
    x_title: str = ""
    y_title: str = ""


ARCHETYPE_SPECS: dict[str, ArchetypeSpec] = {
    "shooting": ArchetypeSpec(
        name="shooting",
        metrics=SHOOTING_METRICS,
        random_state=3,
        cluster_names={
            0: "Paint Dominant",
            1: "Rim and Three",
            2: "Elite Shooting",
            3: "Midrange Heavy",
        },
        plot_title="Team Shooting Archetypes",
        x_title="<- Mid-range heavy | 3PT heavy ->",
        y_title="<- Paint heavy | Rim heavy ->",
    ),
    "playstyle": ArchetypeSpec(
        name="playstyle",
        metrics=PLAYSTYLE_METRICS,
        random_state=0,
        cluster_names={
            0: "Controlled Ball Movement",  # high assists, low turnovers, slow pace
            1: "High-Tempo Aggressive",  # high pace, turnovers, rebounding, FT rate
            2: "Inefficient Isolation",  # very high turnovers, low rebounding and assists
            3: "Slow and Physical",  # high rebounding, slow pace
        },
        plot_title="Team Playstyle Archetypes",
        x_title="<- Slow and Controlled | Fast and Aggressive ->",
        y_title="<- Physical Rebounding | High Turnovers ->",
    ),
    "defense": ArchetypeSpec(
        name="defense",
        metrics=DEFENSE_METRICS,
        random_state=1,
        cluster_names={
            0: "Paint Containment Defense",  # few lane attempts allowed, many 3PA allowed
            1: "Turnover-Forcing Pressure",  # high steals, opp TOV% and fouls; weak DREB
            2: "Conservative Perimeter Defense",  # few 3PA allowed, concedes lane, low pressure
            3: "Disruptive Rim Protection",  # high blocks, low lane FG% and 3P% allowed
        },
        plot_title="Team Defensive Archetypes",
        x_title="<- Low Pressure | Forces Turnovers ->",
        y_title="<- Allows Paint Shots | Forces 3-pointers ->",
    ),
}
