# CBB Team Similarity & Archetypes

Compare college basketball team-seasons and group them into play-style archetypes.

- **Similarity comps.** For any team-season, find the most similar team-seasons from 2020-21 through 2025-26. Similarity is cosine similarity on metrics standardized within each season.
- **Archetypes.** KMeans clusters teams into named archetypes in three areas: shooting, playstyle and defense. A 2-D PCA projection is used to visualize the clusters.

All metrics are standardized and percentile-ranked **within each season**, so league-wide trends don't dominate the comparisons. One example is the rise in three-point rate.

## Project structure

```
cbbProject/
├── data/
│   ├── raw/combined_stats.csv     # one row per team-season (model input)
│   └── data_dictionary.csv        # metric keys -> display labels
├── notebooks/
│   └── team_similarity_model.ipynb  # exploration / presentation, uses the package
├── outputs/                       # generated (git-ignored)
├── src/cbb_similarity/
│   ├── config.py                  # paths, metric groups, archetype specs and names
│   ├── data/loader.py             # load + validate stats, labels, season lookup
│   ├── features/preprocessing.py  # season z-scores and percentile ranks
│   ├── models/similarity.py       # cosine-similarity comps
│   ├── models/archetypes.py       # KMeans + PCA archetype model
│   ├── visualization/plots.py     # archetype scatter + radar comparison (Plotly)
│   ├── pipeline.py                # end-to-end archetype workflow
│   └── cli.py                     # `cbb-similarity` command
├── tests/
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,notebook for Jupyter support
```

## Usage

### Command line

```bash
# Fit all archetype models and write CSVs and interactive plots to outputs/
cbb-similarity archetypes

# Ten most similar team-seasons to 2025-26 Duke, on shooting metrics, with a radar chart
cbb-similarity comps --team Duke --season 2025-26 --group shooting --radar

# Overall comps restricted to the same season (season can also be a competitionId)
cbb-similarity comps --team Michigan --season 41097 --same-season
```

Metric groups: `overall`, `shooting`, `playstyle`, `defense` (defined in `config.py`).

### Python

```python
from cbb_similarity.data import load_stats
from cbb_similarity.models import generate_comps_table
from cbb_similarity.pipeline import build_features, fit_archetypes

features = build_features(load_stats())
comps = generate_comps_table(features.scaled, features.ranks, "Duke", 41097,
                             metrics=["astPct", "tovPct", "pace", "orbPct", "ftaRate"])
results = fit_archetypes(features)   # results.assignments, results.profiles, results.models
```

## Outputs

| Path | Contents |
| --- | --- |
| `outputs/team_archetypes.csv` | Every team-season with cluster id, archetype name and PCA x/y for each model |
| `outputs/profiles/<model>_profile.csv` | Mean season percentile of each metric per archetype, plus team counts |
| `outputs/plots/<model>_archetypes.html` | Interactive archetype scatter plots |
| `outputs/comps/<team>_<season>_<group>.csv` | Comps tables (`*_radar.html` with `--radar`) |

## Archetype names

Cluster names are mapped from KMeans cluster ids in `config.ARCHETYPE_SPECS`. The ids depend on the input data and the random seed. **After refreshing the data, re-run `cbb-similarity archetypes` and check `outputs/profiles/` to confirm the names still describe each cluster.**

## Development

```bash
pytest               # unit tests (synthetic data, no CSV needed)
ruff check src tests
ruff format src tests
```
