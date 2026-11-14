# Customer Segmentation Framework

Python research code for customer segmentation using K-Means, Bisecting K-Means and DBSCAN, weighted dense-rank selection, and interpretable surrogate decision trees.

The workflow connects clustering configuration searches, internal validation, multicriteria evaluation, reference-model fitting, and rule-based explanations in original feature units.

## Status

`segmentation_framework.py` integrates the supplied `codigo.py`, `evaluacion.py`, `ejecuta.py`, and `dt.py` into a common command-line workflow, with additional exports for metrics, profiles, models, and rules. **This integrated version has not been executed or tested.**

Customer-level data and historical result tables are not included in this code package.

## Repository files

```text
.
├── segmentation_framework.py
├── requirements.txt
└── README.md
```

## Installation

Install the dependencies listed in [`requirements.txt`](requirements.txt):

```bash
python -m pip install -r requirements.txt
```

Dependencies include NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn, Plotly, tqdm, and joblib. Versions are not pinned to an assumed historical environment. When the program is run, `environment.json` records the software environment and command-line arguments of that execution.

## Data

The study case comprises 400 registered customers in 2024. Its clustering features are:

| Feature | Definition |
| --- | --- |
| Age | Customer age recorded in 2024. |
| Annual transaction amount | Customer-level transaction amount for the full calendar year, expressed in Peruvian soles (PEN). |
| Annual visit count | Every recorded visit during 2024, counted individually, including multiple visits by the same customer on one day. |

Prepare a customer-level CSV with one row per customer and annual activity measures already computed. The default input settings are a semicolon separator and Latin-1 encoding.

The original feature names are `Edad`, `Monto de Operaciones`, and `Número de Visitas`. The loader excludes `ID_cliente`, `Genero`, and `Género` when present. By default, the remaining numeric columns become the clustering features. Use `--features` to select columns explicitly when the input contains additional numeric attributes.

Numeric features are transformed with Min–Max scaling to `[0, 1]`. The reference clustering labels are attached to features recovered in their original units before fitting the surrogate tree.

## Workflow

```text
Customer-level CSV
        |
Feature selection and Min–Max scaling
        |
K-Means / Bisecting K-Means / DBSCAN searches
        |
Internal metrics and configuration records
        |
Global dense ranks and weighted score
        |
Explicit reference-model refit
        |
Original-unit customer profiles and surrogate tree
        |
Training fidelity, confusion counts, and terminal rules
```

The command-line stages are:

| Command | Purpose |
| --- | --- |
| `search` | Generate metric records for the selected clustering algorithms. |
| `evaluate` | Calculate global dense ranks and weighted scores from a consolidated table. |
| `refit` | Fit the explicitly specified reference configuration and export labeled customers in original units. |
| `tree` | Fit the surrogate and export training-fidelity statistics and rules. |
| `all` | Run the four stages in sequence. |

## Usage

The following commands are usage examples; they have not been executed as part of preparing this package.

### Full workflow

```bash
python segmentation_framework.py all --data "customers.csv" --output-dir "results"
```

The default research grid contains 400,000 configurations per centroid method and 400,000 DBSCAN parameter pairs. Existing result tables can instead be evaluated with the `evaluate` stage.

### Search

```bash
python segmentation_framework.py search --data "customers.csv" --output-dir "results" --algorithms KMeans BisectingKMeans DBSCAN
```

To specify the three study features explicitly:

```bash
python segmentation_framework.py search --data "customers.csv" --output-dir "results" --features "Edad" "Monto de Operaciones" "Número de Visitas"
```

### Evaluate newly generated results

```bash
python segmentation_framework.py evaluate --output-dir "results"
```

This uses the newly consolidated search records without introducing an additional rounding rule.

### Evaluate the historical consolidated table

```bash
python segmentation_framework.py evaluate --ranking-input "ejecuciones.csv" --ranking-delimiter ";" --ranking-encoding latin1 --output-dir "results"
```

Use `--ranking-input` with the historical consolidated table when targeting the paper's reported ranks. Its stored numerical precision is part of that evaluation. The original per-algorithm exports retain more digits; ranking those values directly can produce different dense ranks and scores.

### Refit the study reference

```bash
python segmentation_framework.py refit --data "customers.csv" --output-dir "results" --reference-algorithm KMeans --reference-k 5 --reference-seed 86723
```

The reference is explicit. This stage does not automatically replace it with the first row of a new ranking or introduce a secondary criterion for tied scores.

### Fit the surrogate tree

```bash
python segmentation_framework.py tree --output-dir "results" --tree-depth 4 --tree-min-leaf 40 --tree-seed 42
```

By default, this reads `data_clusterizada.csv` from the output directory. To supply another labeled table:

```bash
python segmentation_framework.py tree --clustered-input "results/data_clusterizada.csv" --output-dir "results"
```

### Visualizations

Add `--plots` to request figure exports and `--show` to display them interactively. For the surrogate boundary visualization:

```bash
python segmentation_framework.py tree --output-dir "results" --plots --boundary-plot
```

The boundary view uses a synthetic grid of 35 values per feature axis. It visualizes the tree's predictions; fidelity statistics are calculated on the labeled customer records.

Individual centroid boxplots group runs by requested cluster count. Individual DBSCAN boxplots group by `min_samples`; the integrated overview groups DBSCAN configurations by detected cluster count.

Add `--explore` to `search`, `refit`, or `all` for descriptive exports. Combine it with `--plots` for distributions and correlation visualizations.

## Default study configuration

| Component | Settings |
| --- | --- |
| Scaling | Min–Max to `[0, 1]`. |
| Centroid cluster counts | `k = 2, 3, 4, 5`. |
| External random states | Integers `0` through `99999` for each `k` and centroid method. |
| Centroid search size | 100,000 states per cluster count; 400,000 configurations per method. |
| DBSCAN radii | `epsilon_j = j / 100000`, for `j = 1, ..., 100000`. |
| DBSCAN neighbors | `min_samples = 2, 3, 4, 5`, crossed with all radii. |
| DBSCAN feasibility | Two to five clusters and noise fraction at most `0.05`. |
| Reference model | K-Means, `k = 5`, `random_state = 86723`. |
| Surrogate | `max_depth = 4`, `min_samples_leaf = 40`, `random_state = 42`. |

Estimator options not explicitly set retain the defaults of the installed library. Search limits and reference parameters can be changed through command-line arguments.

## Metrics and selection

Clustering is evaluated on scaled features using Silhouette, Calinski–Harabasz, Davies–Bouldin, and inertia. For DBSCAN, metrics use observations assigned to clusters, excluding the noise label `-1`. DBSCAN inertia is calculated as the sum of squared distances to the centroids of the assigned groups. Coverage is `1 - noise_fraction`.

Dense ranks are calculated globally across algorithms and cluster counts. Equal metric values share a rank, and the next distinct value receives the next integer rank:

| Criterion | Preferred direction | Weight |
| --- | --- | --- |
| Silhouette | Higher | 0.40 |
| Calinski–Harabasz | Higher | 0.25 |
| Davies–Bouldin | Lower | 0.20 |
| Inertia | Lower | 0.15 |

```text
Score = 0.40 * rank_S
      + 0.25 * rank_CH
      + 0.20 * rank_DB
      + 0.15 * rank_W
```

Lower scores are preferred. Equal aggregate scores remain tied.

`perfiles_metricos.csv` summarizes identical four-metric tuples. These are **metric profiles**, not partitions whose membership equivalence has been verified.

The surrogate is fitted and evaluated on the same labeled customer table. Its accuracy therefore measures **in-sample fidelity to clustering labels**, rather than performance on held-out customers or agreement with independently annotated customer categories.

## Generated outputs

Files depend on the stages and visualization options selected.

| Output | Contents |
| --- | --- |
| `ejecuciones_KMeans.csv` | K-Means configurations and metric values. |
| `ejecuciones_BisectingKMeans.csv` | Bisecting K-Means configurations and metric values. |
| `ejecuciones_DBSCAN.csv` | Retained DBSCAN configurations, noise fractions, and metrics. |
| `resumen_busqueda.csv` | Configuration counts by algorithm and cluster count. |
| `ejecuciones_consolidadas.csv` | Consolidated records from a new search. |
| `entrada_ranking_utilizada.csv` | Numeric input used for ranking. |
| `ejecuciones_rank.csv` | Dense criterion ranks and weighted scores. |
| `configuraciones_score_minimo.csv` | Records tied at the minimum score. |
| `perfiles_metricos.csv` | Representatives of identical metric tuples. |
| `data_clusterizada.csv` | Reference cluster labels and original-unit features. |
| `perfiles_segmentos.csv` | Segment counts, shares, and feature means. |
| `predicciones_arbol.csv` | Customer-level surrogate predictions, leaf IDs, and agreement. |
| `matriz_confusion.csv` | Counts comparing clustering labels with surrogate predictions. |
| `reporte_clasificacion.csv` | Class-level and aggregate training-fidelity metrics. |
| `reglas_arbol.csv`, `reglas_arbol.txt` | Terminal conditions and a text representation of the tree. |
| `fidelidad_reglas.csv` | Leaf support, agreement, disagreements, and fidelity. |
| `resultados_arbol.json` | Overall training-fidelity statistics and explicit tree settings. |
| `reference_configuration.json` | Reference-model settings and feature names. |
| `environment.json` | Environment and arguments recorded when the program is executed. |
| `models/` | Serialized scaler, reference clustering model, and surrogate. |
| `figures/` | Requested figure exports and interactive partition visualization. |

The environment record describes the execution that creates it; it is not a reconstruction of the historical study environment. Generated customer-level tables are not part of this distributed code package.
