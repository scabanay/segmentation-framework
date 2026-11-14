#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integrated customer-segmentation workflow for supplementary research code.

Assembled from the supplied codigo.py, evaluacion.py, ejecuta.py and dt.py.
This integrated version has not been executed, syntax-checked, or tested.

Stages: search -> evaluate -> refit -> tree.
The paper's explicit reference defaults to K-Means, k=5, random_state=86723.
Ranking uses dense ranks and the supplied weights. A historical ranking-input
table can be supplied separately to preserve its stored numerical precision.
New search results are consolidated without adding a rounding rule.

Customer records are not distributed with this code. The reference year is
2024; visits are counted individually, including multiple visits on one day.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import unicodedata
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import BisectingKMeans, DBSCAN, KMeans
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from tqdm import tqdm


METRICS = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "Inercia"]
WEIGHTS = {
    "Silhouette": 0.40,
    "Calinski-Harabasz": 0.25,
    "Davies-Bouldin": 0.20,
    "Inercia": 0.15,
}
ALGORITHMS = ["KMeans", "BisectingKMeans", "DBSCAN"]


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def save_json(path, content):
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def slug(value):
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def output_folders(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["models", "figures"]:
        (output_dir / name).mkdir(exist_ok=True)


def record_environment(args):
    """Record the environment of a future execution, not the historical one."""
    versions = {}
    for package in [
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
        "seaborn", "plotly", "tqdm", "joblib",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    save_json(args.output_dir / "environment.json", {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "arguments": vars(args),
        "reference_year": 2024,
        "weights": WEIGHTS,
    })


def load_features(args):
    if args.data is None:
        raise ValueError("This stage requires --data with the customer CSV path.")
    original = pd.read_csv(
        args.data, encoding=args.data_encoding, sep=args.data_delimiter,
    )
    # Explicit exclusions implement the representation described in the paper.
    numeric = original.drop(
        columns=["ID_cliente", "Genero", "Género"], errors="ignore",
    )
    numeric = numeric.loc[:, ~numeric.columns.str.startswith("Unnamed:")]
    if args.features:
        numeric = numeric[args.features].copy()
    else:
        numeric = numeric.select_dtypes(include=[np.number]).copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = pd.DataFrame(
        scaler.fit_transform(numeric), columns=numeric.columns, index=numeric.index,
    )
    return original, numeric, scaled, scaler


def metric_values(features, labels, inertia):
    return {
        "Silhouette": silhouette_score(features, labels),
        "Davies-Bouldin": davies_bouldin_score(features, labels),
        "Calinski-Harabasz": calinski_harabasz_score(features, labels),
        "Inercia": inertia,
    }


def density_inertia(features, labels):
    values = np.asarray(features)
    total = 0.0
    for cluster in np.unique(labels):
        points = values[labels == cluster]
        center = points.mean(axis=0)
        total += ((points - center) ** 2).sum()
    return total


def search_centroid(algorithm, scaled, args):
    estimator = KMeans if algorithm == "KMeans" else BisectingKMeans
    rows = []
    execution = 0
    for k in tqdm(range(args.k_min, args.k_max + 1), desc=f"{algorithm}: k"):
        for seed in tqdm(
            range(args.seed_start, args.seed_start + args.seed_count),
            desc="random_state", leave=False,
        ):
            model = estimator(n_clusters=k, random_state=seed, verbose=0)
            model.fit(scaled)
            rows.append({
                "Algoritmo": algorithm,
                "Ejecución": execution,
                "Número de Clusters": k,
                "Semilla": seed,
                "Valor eps": np.nan,
                "Min Vecinos": np.nan,
                "Ruido": 0.0,
                **metric_values(scaled, model.labels_, model.inertia_),
            })
            execution += 1
    return pd.DataFrame(rows)


def search_density(scaled, args):
    rows = []
    radii = np.linspace(0, 1, args.eps_count + 1)[1:]
    # The complete product has eps_count * number_of_neighbor_settings pairs.
    for minimum in tqdm(
        range(args.min_samples_min, args.min_samples_max + 1),
        desc="DBSCAN: min_samples",
    ):
        for radius in tqdm(radii, desc="eps", leave=False):
            model = DBSCAN(eps=radius, min_samples=minimum)
            model.fit(scaled)
            labels = model.labels_
            assigned = labels != -1
            count = len(set(labels[assigned]))
            noise = float((labels == -1).mean())
            if (
                args.k_min <= count <= args.k_max
                and assigned.sum() > count
                and noise <= args.max_noise
            ):
                features = scaled.loc[assigned]
                kept_labels = labels[assigned]
                rows.append({
                    "Algoritmo": "DBSCAN",
                    "Ejecución": len(rows),
                    "Número de Clusters": count,
                    "Semilla": np.nan,
                    "Valor eps": radius,
                    "Min Vecinos": minimum,
                    "Ruido": noise,
                    **metric_values(
                        features, kept_labels, density_inertia(features, kept_labels),
                    ),
                })
    columns = [
        "Algoritmo", "Ejecución", "Número de Clusters", "Semilla", "Valor eps",
        "Min Vecinos", "Ruido", "Silhouette", "Davies-Bouldin",
        "Calinski-Harabasz", "Inercia",
    ]
    return pd.DataFrame(rows, columns=columns)


def finish_figure(fig, path, args):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=250, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)


def plot_metric_boxes(frame, algorithm, args):
    if frame.empty:
        return
    group = "Min Vecinos" if algorithm == "DBSCAN" else "Número de Clusters"
    for metric in METRICS:
        summary = frame.groupby(group)[metric].agg(["min", "max"])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=frame, x=group, y=metric, ax=ax)
        positions = np.arange(len(summary))
        minimize = metric in ["Davies-Bouldin", "Inercia"]
        for statistic, favorable in [("min", minimize), ("max", not minimize)]:
            ax.plot(
                positions, summary[statistic], marker="o", linewidth=0.5,
                color="green" if favorable else "red",
                label="best" if favorable else "worst",
            )
        ax.set_title(f"{algorithm}: {metric}")
        ax.legend()
        fig.tight_layout()
        finish_figure(
            fig, args.output_dir / "figures" / f"{algorithm}_{slug(metric)}_boxplot", args,
        )


def plot_combined_metrics(frame, args):
    """Integrated overview; DBSCAN groups by detected k in this overview."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), squeeze=False)
    for row, metric in enumerate(METRICS):
        for column, algorithm in enumerate(ALGORITHMS):
            ax = axes[row, column]
            subset = frame.loc[frame["Algoritmo"] == algorithm]
            if not subset.empty:
                sns.boxplot(data=subset, x="Número de Clusters", y=metric, ax=ax)
            ax.set_title(algorithm)
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel(metric)
    fig.tight_layout()
    finish_figure(
        fig, args.output_dir / "figures" / "Boxplot_general_en_v1.4.01", args,
    )


def save_single_index_candidates(frame, algorithm, output_dir):
    if frame.empty:
        return
    for metric in METRICS:
        grouped = frame.groupby("Número de Clusters")[metric]
        indices = (
            grouped.idxmin() if metric in ["Davies-Bouldin", "Inercia"]
            else grouped.idxmax()
        )
        frame.loc[indices].sort_values("Número de Clusters").to_csv(
            output_dir / f"mejores_{algorithm}_{slug(metric)}.csv", index=False,
        )


def normalize_records(frame, algorithm=None):
    frame = frame.rename(columns={
        "Davies Bouldin": "Davies-Bouldin",
        "Calinski Harabasz": "Calinski-Harabasz",
        "% Ruido": "Ruido",
    }).copy()
    index_columns = [c for c in frame.columns if c.startswith("Unnamed:")]
    if "Ejecución" not in frame.columns:
        if index_columns:
            frame = frame.rename(columns={index_columns.pop(0): "Ejecución"})
        else:
            frame["Ejecución"] = np.arange(len(frame))
    frame = frame.drop(columns=index_columns, errors="ignore")
    if "Algoritmo" not in frame.columns and algorithm is not None:
        frame["Algoritmo"] = algorithm
    if "Algoritmo" in frame.columns:
        frame["Algoritmo"] = frame["Algoritmo"].replace({"Kmeans": "KMeans"})
    return frame


def consolidate(args):
    frames = []
    for algorithm in args.algorithms:
        path = args.output_dir / f"ejecuciones_{algorithm}.csv"
        frames.append(normalize_records(pd.read_csv(path), algorithm))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(
        args.output_dir / "ejecuciones_consolidadas.csv", sep=";", index=False,
    )
    return combined


def search(scaled, args):
    for algorithm in args.algorithms:
        records = (
            search_density(scaled, args) if algorithm == "DBSCAN"
            else search_centroid(algorithm, scaled, args)
        )
        records.to_csv(args.output_dir / f"ejecuciones_{algorithm}.csv", index=False)
        save_single_index_candidates(records, algorithm, args.output_dir)
        if args.plots:
            plot_metric_boxes(records, algorithm, args)
    combined = consolidate(args)
    combined.groupby(["Algoritmo", "Número de Clusters"]).size().rename(
        "Registros",
    ).reset_index().to_csv(args.output_dir / "resumen_busqueda.csv", index=False)
    if args.plots:
        plot_combined_metrics(combined, args)
    return combined


def evaluate(args):
    if args.ranking_input is not None:
        # Preserve this table's numerical values; do not invent historical rounding.
        table = pd.read_csv(
            args.ranking_input, sep=args.ranking_delimiter,
            encoding=args.ranking_encoding,
        )
        table = normalize_records(table)
    else:
        path = args.output_dir / "ejecuciones_consolidadas.csv"
        table = (
            normalize_records(pd.read_csv(path, sep=";"))
            if path.exists() else consolidate(args)
        )
    table.to_csv(args.output_dir / "entrada_ranking_utilizada.csv", index=False)
    for metric in METRICS:
        table[f"RANK_{metric}"] = table[metric].rank(
            ascending=metric in ["Davies-Bouldin", "Inercia"], method="dense",
        ).astype(int)
    table["Score"] = (
        0.40 * table["RANK_Silhouette"]
        + 0.25 * table["RANK_Calinski-Harabasz"]
        + 0.20 * table["RANK_Davies-Bouldin"]
        + 0.15 * table["RANK_Inercia"]
    )
    if "Ruido" in table.columns:
        table["Cobertura"] = 1.0 - table["Ruido"].fillna(0.0)
    ranked = table.sort_values(by=["Score"], ascending=True)
    ranked.to_csv(args.output_dir / "ejecuciones_rank.csv", index=False)

    # A profile is an identical metric tuple, not a verified distinct partition.
    presentation = ranked
    if all(c in ranked.columns for c in ["Algoritmo", "Semilla", "Número de Clusters"]):
        reference = ranked.loc[
            (ranked["Algoritmo"] == args.reference_algorithm)
            & (ranked["Número de Clusters"] == args.reference_k)
            & (ranked["Semilla"] == args.reference_seed)
        ]
        presentation = pd.concat([reference, ranked], ignore_index=True)
    profiles = presentation.drop_duplicates(subset=METRICS).sort_values("Score")
    profiles = profiles.copy()
    profiles.insert(0, "Posición perfil", np.arange(1, len(profiles) + 1))
    profiles.to_csv(args.output_dir / "perfiles_metricos.csv", index=False)
    best = ranked.loc[ranked["Score"] == ranked["Score"].min()]
    best.to_csv(args.output_dir / "configuraciones_score_minimo.csv", index=False)
    return ranked


def plot_selected_partition(selected, features, args):
    if len(features) != 3:
        return
    plot_data = selected.copy()
    plot_data["Cluster"] = plot_data["Cluster"].astype(str)
    figure = px.scatter_3d(
        plot_data, x=features[0], y=features[1], z=features[2],
        color="Cluster", hover_data=list(plot_data.columns), opacity=0.75,
        title="Selected customer partition",
    )
    figure.update_traces(marker=dict(size=5))
    figure.write_html(
        args.output_dir / "figures" / "scatter_3d_kmeans.html", include_plotlyjs=True,
    )
    if args.show:
        figure.show()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        selected[features[0]], selected[features[1]], selected[features[2]],
        c=pd.Categorical(selected["Cluster"]).codes, s=35, alpha=0.8,
    )
    for setter, label in zip(
        [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel], features,
    ):
        setter(label)
    finish_figure(
        fig, args.output_dir / "figures" / "kmeans_current_view_vector", args,
    )


def refit_reference(scaled, scaler, args):
    if args.reference_algorithm == "KMeans":
        model = KMeans(
            n_clusters=args.reference_k, random_state=args.reference_seed, verbose=0,
        )
    elif args.reference_algorithm == "BisectingKMeans":
        model = BisectingKMeans(
            n_clusters=args.reference_k, random_state=args.reference_seed, verbose=0,
        )
    else:
        if args.reference_eps is None:
            raise ValueError("A DBSCAN reference requires --reference-eps.")
        model = DBSCAN(
            eps=args.reference_eps, min_samples=args.reference_min_samples,
        )
    model.fit(scaled)
    selected = pd.DataFrame(
        scaler.inverse_transform(scaled), columns=scaled.columns, index=scaled.index,
    )
    selected["Cluster"] = model.labels_
    selected.to_csv(args.output_dir / "data_clusterizada.csv", index=False)
    assigned = selected.loc[selected["Cluster"] != -1]
    grouped = assigned.groupby("Cluster")
    profiles = grouped[list(scaled.columns)].mean()
    profiles.insert(0, "n", grouped.size())
    profiles.insert(1, "share", profiles["n"] / len(selected))
    profiles.to_csv(args.output_dir / "perfiles_segmentos.csv")
    joblib.dump(model, args.output_dir / "models" / "reference_clustering.joblib")
    joblib.dump(scaler, args.output_dir / "models" / "scaler.joblib")
    save_json(args.output_dir / "reference_configuration.json", {
        "algorithm": args.reference_algorithm,
        "k": args.reference_k,
        "seed": args.reference_seed,
        "eps": args.reference_eps,
        "min_samples": args.reference_min_samples,
        "features": list(scaled.columns),
        "selection": "Explicit reference configuration; no automatic aggregate-tie resolution.",
    })
    if args.plots:
        plot_selected_partition(selected, list(scaled.columns), args)
    return selected


def leaf_rule_rows(model, features, target, prediction):
    tree = model.tree_
    leaf_ids = model.apply(features)
    rows = []

    def visit(node, conditions):
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == right:
            mask = leaf_ids == node
            counts = target.loc[mask].value_counts()
            agreements = int((prediction[mask] == target.loc[mask].to_numpy()).sum())
            count = int(mask.sum())
            rows.append({
                "Leaf": int(node),
                "Antecedent": " and ".join(conditions) or "all observations",
                "Predicted cluster": model.classes_[int(np.argmax(tree.value[node][0]))],
                "n": count,
                "Support": count / len(features),
                "Agreements": agreements,
                "Disagreements": count - agreements,
                "Fidelity": agreements / count,
                "Reference class counts": json.dumps(
                    {str(key): int(value) for key, value in counts.items()},
                    ensure_ascii=False,
                ),
            })
            return
        name = features.columns[tree.feature[node]]
        threshold = tree.threshold[node]
        visit(left, conditions + [f"{name} <= {threshold:.12g}"])
        visit(right, conditions + [f"{name} > {threshold:.12g}"])

    visit(0, [])
    return pd.DataFrame(rows)


def plot_tree_boundaries(model, features, target, args):
    """Visualization on a synthetic grid; not a new-customer fidelity estimate."""
    if features.shape[1] != 3:
        return
    size = args.boundary_grid
    axes = [np.linspace(features[c].min(), features[c].max(), size) for c in features]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    grid = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel(), zz.ravel()], columns=features.columns,
    )
    predicted = model.predict(grid).reshape(size, size, size)
    boundary = np.zeros_like(predicted, dtype=bool)
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = slice(None, -1)
        high[axis] = slice(1, None)
        low, high = tuple(low), tuple(high)
        changes = predicted[low] != predicted[high]
        boundary[low] |= changes
        boundary[high] |= changes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        features.iloc[:, 0], features.iloc[:, 1], features.iloc[:, 2],
        c=pd.Categorical(target, categories=model.classes_).codes,
        s=45, edgecolor="k", alpha=0.9,
    )
    colors = pd.Categorical(predicted[boundary], categories=model.classes_).codes
    ax.scatter(xx[boundary], yy[boundary], zz[boundary], c=colors, s=8, alpha=0.35)
    ax.set_xlabel(features.columns[0])
    ax.set_ylabel(features.columns[1])
    ax.set_zlabel(features.columns[2])
    ax.set_title("Surrogate decision boundaries: synthetic-grid visualization")
    finish_figure(fig, args.output_dir / "figures" / "tree_boundaries_3d", args)


def fit_tree(args):
    path = args.clustered_input or args.output_dir / "data_clusterizada.csv"
    frame = pd.read_csv(path)
    frame = frame.drop(
        columns=[c for c in frame.columns if c.startswith("Unnamed:")], errors="ignore",
    )
    features = frame.drop(columns=["Cluster"])
    target = frame["Cluster"]
    model = DecisionTreeClassifier(
        max_depth=args.tree_depth,
        min_samples_leaf=args.tree_min_leaf,
        random_state=args.tree_seed,
    )
    model.fit(features, target)
    prediction = model.predict(features)
    fidelity = accuracy_score(target, prediction)
    predicted_table = frame.copy()
    predicted_table["Tree prediction"] = prediction
    predicted_table["Tree leaf"] = model.apply(features)
    predicted_table["Agreement"] = target.to_numpy() == prediction
    predicted_table.to_csv(args.output_dir / "predicciones_arbol.csv", index=False)
    matrix = confusion_matrix(target, prediction, labels=model.classes_)
    pd.DataFrame(
        matrix,
        index=pd.Index(model.classes_, name="Reference cluster"),
        columns=pd.Index(model.classes_, name="Tree prediction"),
    ).to_csv(args.output_dir / "matriz_confusion.csv")
    report = classification_report(
        target, prediction, output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        args.output_dir / "reporte_clasificacion.csv",
    )
    rules = leaf_rule_rows(model, features, target, prediction)
    rules.to_csv(args.output_dir / "reglas_arbol.csv", index=False)
    rules.to_csv(args.output_dir / "fidelidad_reglas.csv", index=False)
    (args.output_dir / "reglas_arbol.txt").write_text(
        export_text(model, feature_names=list(features.columns), decimals=6),
        encoding="utf-8",
    )
    save_json(args.output_dir / "resultados_arbol.json", {
        "evaluation": "in-sample agreement with clustering pseudo-labels",
        "samples": len(features),
        "fidelity": fidelity,
        "agreements": int((target.to_numpy() == prediction).sum()),
        "leaves": model.get_n_leaves(),
        "depth": model.get_depth(),
        "explicit_parameters": {
            "max_depth": args.tree_depth,
            "min_samples_leaf": args.tree_min_leaf,
            "random_state": args.tree_seed,
        },
    })
    joblib.dump(model, args.output_dir / "models" / "surrogate_tree.joblib")
    if args.plots:
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model, feature_names=list(features.columns),
            class_names=[str(c) for c in model.classes_],
            filled=True, rounded=True, ax=ax,
        )
        finish_figure(fig, args.output_dir / "figures" / "decision_tree_paper", args)
        if args.boundary_plot:
            plot_tree_boundaries(model, features, target, args)
    return model


def explore(original, features, args):
    original.describe(include="all").transpose().to_csv(
        args.output_dir / "estadistica_descriptiva.csv",
    )
    original.isna().sum().rename("Missing").to_csv(
        args.output_dir / "valores_faltantes.csv",
    )
    if not args.plots:
        return
    for column in features:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(features[column], kde=True, ax=ax)
        finish_figure(
            fig, args.output_dir / "figures" / f"distribution_{slug(column)}", args,
        )
    for method in ["pearson", "spearman"]:
        correlation = features.corr(method=method)
        correlation.to_csv(args.output_dir / f"correlacion_{method}.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            correlation, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax,
        )
        fig.tight_layout()
        finish_figure(fig, args.output_dir / "figures" / f"correlation_{method}", args)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["search", "evaluate", "refit", "tree", "all"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-delimiter", default=";")
    parser.add_argument("--data-encoding", default="latin-1")
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--algorithms", choices=ALGORITHMS, nargs="+", default=ALGORITHMS)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100000)
    parser.add_argument("--eps-count", type=int, default=100000)
    parser.add_argument("--min-samples-min", type=int, default=2)
    parser.add_argument("--min-samples-max", type=int, default=5)
    parser.add_argument("--max-noise", type=float, default=0.05)
    parser.add_argument("--ranking-input", type=Path)
    parser.add_argument("--ranking-delimiter", default=";")
    parser.add_argument("--ranking-encoding", default="latin-1")
    parser.add_argument("--reference-algorithm", choices=ALGORITHMS, default="KMeans")
    parser.add_argument("--reference-k", type=int, default=5)
    parser.add_argument("--reference-seed", type=int, default=86723)
    parser.add_argument("--reference-eps", type=float)
    parser.add_argument("--reference-min-samples", type=int, default=2)
    parser.add_argument("--clustered-input", type=Path)
    parser.add_argument("--tree-depth", type=int, default=4)
    parser.add_argument("--tree-min-leaf", type=int, default=40)
    parser.add_argument("--tree-seed", type=int, default=42)
    parser.add_argument("--boundary-grid", type=int, default=35)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--boundary-plot", action="store_true")
    parser.add_argument("--explore", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    output_folders(args.output_dir)
    record_environment(args)
    sns.set_theme(style="whitegrid", palette="tab10")
    loaded = None
    if args.command in ["search", "refit", "all"]:
        loaded = load_features(args)
        original, features, scaled, scaler = loaded
        if args.explore:
            explore(original, features, args)
    if args.command in ["search", "all"]:
        search(loaded[2], args)
    if args.command in ["evaluate", "all"]:
        evaluate(args)
    if args.command in ["refit", "all"]:
        refit_reference(loaded[2], loaded[3], args)
    if args.command in ["tree", "all"]:
        fit_tree(args)


if __name__ == "__main__":
    main()
