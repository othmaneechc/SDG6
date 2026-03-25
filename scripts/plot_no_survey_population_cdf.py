#!/usr/bin/env python3
"""Plot population-weighted CDFs for no-survey countries."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURVEY_COUNTRY_MAP = {
    "ANG": "Angola",
    "BEN": "Benin",
    "BFO": "Burkina Faso",
    "BOT": "Botswana",
    "CAM": "Cameroon",
    "CBZ": "Congo",
    "CDI": "Cote d'Ivoire",
    "CVE": "Cabo Verde",
    "ESW": "eSwatini",
    "ETH": "Ethiopia",
    "GAB": "Gabon",
    "GAM": "Gambia",
    "GHA": "Ghana",
    "GUI": "Guinea",
    "KEN": "Kenya",
    "LES": "Lesotho",
    "LIB": "Liberia",
    "MAD": "Madagascar",
    "MAU": "Mauritius",
    "MLI": "Mali",
    "MLW": "Malawi",
    "MOR": "Morocco",
    "MOZ": "Mozambique",
    "MTA": "Mauritania",
    "NAM": "Namibia",
    "NGR": "Niger",
    "NIG": "Nigeria",
    "SAF": "South Africa",
    "SEN": "Senegal",
    "SEY": "Seychelles",
    "SRL": "Sierra Leone",
    "STP": "Sao Tome and Principe",
    "SUD": "Sudan",
    "SWZ": "eSwatini",
    "TAN": "Tanzania",
    "TOG": "Togo",
    "TUN": "Tunisia",
    "UGA": "Uganda",
    "ZAM": "Zambia",
    "ZIM": "Zimbabwe",
}

COUNTRY_ALIASES = {
    "Cote d'Ivoire": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Eq. Guinea": "Equatorial Guinea",
    "Dem. Rep. Congo": "Democratic Republic of the Congo",
    "Central African Rep.": "Central African Republic",
}

SPECS = [
    ("Piped water", "Piped Water Access (%)", "Piped (JMP)"),
    ("Sanitation", "Sewage Access (%)", "Safely managed"),
]

THRESHOLDS = (10.0, 15.0)


def canonical_country_name(name: object) -> object:
    if pd.isna(name):
        return name
    return COUNTRY_ALIASES.get(str(name).strip(), str(name).strip())


def normalize_country_key(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("&", "and")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return text


def build_survey_country_keys(survey_csv_paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for csv_path in survey_csv_paths:
        survey_df = pd.read_csv(csv_path, usecols=["RESPNO"])
        country_names = (
            survey_df["RESPNO"]
            .astype(str)
            .str[:3]
            .map(SURVEY_COUNTRY_MAP)
            .map(canonical_country_name)
        )
        keys.update(normalize_country_key(name) for name in country_names.dropna().unique())
    return keys


def no_survey_service_frame(
    summary: pd.DataFrame,
    survey_country_keys: set[str],
    pred_col: str,
    gt_col: str,
) -> pd.DataFrame:
    sub = summary[["Country", pred_col, gt_col, "Total Population", "country_key"]].copy()
    sub[pred_col] = pd.to_numeric(sub[pred_col], errors="coerce")
    sub[gt_col] = pd.to_numeric(sub[gt_col], errors="coerce")
    sub["Total Population"] = pd.to_numeric(sub["Total Population"], errors="coerce")
    sub = sub.dropna(subset=[pred_col, gt_col, "Total Population"])
    sub = sub[sub["country_key"].map(lambda k: k not in survey_country_keys)].copy()
    sub["abs_error_pp"] = (sub[pred_col] - sub[gt_col]).abs()
    sub["weight"] = sub["Total Population"].clip(lower=1.0)
    return sub[["Country", "abs_error_pp", "weight"]].sort_values("abs_error_pp")


def weighted_cdf(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    errors = frame["abs_error_pp"].to_numpy(dtype=float)
    weights = frame["weight"].to_numpy(dtype=float)
    cumsum = np.cumsum(weights)
    total = float(weights.sum())
    x = np.concatenate(([0.0], errors))
    y = np.concatenate(([0.0], (100.0 * cumsum / total)))
    return x, y


def summarize_service(frame: pd.DataFrame) -> dict[str, float]:
    weights = frame["weight"].to_numpy(dtype=float)
    errors = frame["abs_error_pp"].to_numpy(dtype=float)
    total_pop = float(weights.sum())
    return {
        "countries": int(len(frame)),
        "population": total_pop,
        "weighted_mae": float(np.average(errors, weights=weights)),
    }


def coverage_at_threshold(frame: pd.DataFrame, threshold: float) -> tuple[float, float]:
    covered = float(frame.loc[frame["abs_error_pp"] <= threshold, "weight"].sum())
    total = float(frame["weight"].sum())
    pct = 100.0 * covered / total if total else 0.0
    return covered, pct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/home/mila/e/echchabo/projects/SDG6-Tracker/data/inference_summary.csv"),
    )
    parser.add_argument(
        "--r7-csv",
        type=Path,
        default=Path("/home/mila/e/echchabo/scratch/SDG6-Tracker/R7.csv"),
    )
    parser.add_argument(
        "--r8-csv",
        type=Path,
        default=Path("/home/mila/e/echchabo/scratch/SDG6-Tracker/R8.csv"),
    )
    parser.add_argument(
        "--r9-csv",
        type=Path,
        default=Path("/home/mila/e/echchabo/scratch/SDG6-Tracker/R9.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("/home/mila/e/echchabo/projects/SDG6-Tracker/notebooks/no_survey_population_coverage_cdf.png"),
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=Path("/home/mila/e/echchabo/projects/SDG6-Tracker/notebooks/no_survey_population_coverage_summary.csv"),
    )
    parser.add_argument(
        "--output-cdf-csv",
        type=Path,
        default=Path("/home/mila/e/echchabo/projects/SDG6-Tracker/notebooks/no_survey_population_coverage_cdf_points.csv"),
    )
    args = parser.parse_args()

    survey_country_keys = build_survey_country_keys([args.r7_csv, args.r8_csv, args.r9_csv])

    summary = pd.read_csv(args.summary_csv)
    summary["country_key"] = summary["Country"].map(canonical_country_name).map(normalize_country_key)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    service_styles = {
        "Piped water": {"color": "#2a7f62"},
        "Sanitation": {"color": "#c65a39"},
    }
    annotation_offsets = {
        ("Piped water", 10.0): (6, 8),
        ("Piped water", 15.0): (6, -16),
        ("Sanitation", 10.0): (6, -16),
        ("Sanitation", 15.0): (6, 8),
    }

    summary_rows: list[dict[str, object]] = []
    cdf_rows: list[dict[str, float]] = []
    max_error = 0.0

    for service, pred_col, gt_col in SPECS:
        frame = no_survey_service_frame(summary, survey_country_keys, pred_col, gt_col)
        if frame.empty:
            continue

        x, y = weighted_cdf(frame)
        max_error = max(max_error, float(frame["abs_error_pp"].max()))
        stats = summarize_service(frame)
        style = service_styles[service]

        ax.step(
            x,
            y,
            where="post",
            linewidth=2.3,
            color=style["color"],
            label=f"{service} (n={stats['countries']})",
        )

        cdf_rows.append(
            {
                "service": service,
                "threshold_pp": 0.0,
                "population_covered": 0.0,
                "population_covered_pct": 0.0,
            }
        )
        cum_pop = frame["weight"].cumsum().to_numpy(dtype=float)
        for threshold_pp, pop_cov, pct_cov in zip(
            frame["abs_error_pp"].to_numpy(dtype=float),
            cum_pop,
            y[1:],
        ):
            cdf_rows.append(
                {
                    "service": service,
                    "threshold_pp": float(threshold_pp),
                    "population_covered": float(pop_cov),
                    "population_covered_pct": float(pct_cov),
                }
            )

        for threshold in THRESHOLDS:
            covered_pop, covered_pct = coverage_at_threshold(frame, threshold)
            ax.scatter([threshold], [covered_pct], color=style["color"], s=35, zorder=5)
            dx, dy = annotation_offsets[(service, threshold)]
            ax.annotate(
                f"{covered_pop / 1e6:.1f}M ({covered_pct:.1f}%)",
                xy=(threshold, covered_pct),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                color=style["color"],
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#d1d5db", "alpha": 0.9},
            )
            summary_rows.append(
                {
                    "service": service,
                    "countries": stats["countries"],
                    "population": stats["population"],
                    "weighted_mae": stats["weighted_mae"],
                    "threshold_pp": threshold,
                    "population_covered": covered_pop,
                    "population_covered_pct": covered_pct,
                }
            )

    for threshold in THRESHOLDS:
        ax.axvline(threshold, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            threshold,
            1.2,
            f"{int(threshold)} pp",
            rotation=90,
            ha="center",
            va="bottom",
            color="#6b7280",
            fontsize=9,
        )

    ax.set_xlim(0, max(max_error + 2.0, 20.0))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Absolute error threshold vs JMP (percentage points)")
    ax.set_ylabel("Population coverage within threshold (%)")
    ax.set_title("No-survey countries: population-weighted CDF of absolute error")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=300, bbox_inches="tight")

    pd.DataFrame(summary_rows).to_csv(args.output_summary_csv, index=False)
    pd.DataFrame(cdf_rows).to_csv(args.output_cdf_csv, index=False)

    print(f"Saved figure: {args.output_figure}")
    print(f"Saved summary: {args.output_summary_csv}")
    print(f"Saved CDF points: {args.output_cdf_csv}")


if __name__ == "__main__":
    main()
