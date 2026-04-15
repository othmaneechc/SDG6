"""Converted from notebook: urban_rural_split_analysis.ipynb"""

# %% [cell 1]
from pathlib import Path
import csv
import os
from collections import Counter

try:
    from IPython.display import Markdown, display
except ImportError:
    def Markdown(text):
        return text

    def display(obj):
        print(obj)

try:
    import pandas as pd
except ImportError:
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SDG6_DATA_ROOT", REPO_ROOT / "data")).expanduser()
RUNS_ROOT = Path(os.environ.get("SDG6_RUNS_ROOT", REPO_ROOT / "runs")).expanduser()

survey_paths = {
    'R7': DATA_ROOT / "R7.csv",
    'R8': DATA_ROOT / "R8.csv",
    'R9': DATA_ROOT / "R9.csv",
}

split_test_roots = {
    'PW-s': DATA_ROOT / "PW-s" / "test",
    'SW-s': DATA_ROOT / "SW-s" / "test",
}

urbanicity_map = {
    '1': 'Urban',
    '1.0': 'Urban',
    '2': 'Rural',
    '2.0': 'Rural',
}

def norm_coord(value):
    return f"{float(value):.6f}"

def pct(part, total):
    return 100.0 * part / total if total else 0.0

def display_rows(rows, title):
    display(Markdown(f"### {title}"))
    if pd is not None:
        display(pd.DataFrame(rows))
    else:
        for row in rows:
            print(row)

def analyze_surveys(paths):
    round_row_counts = {}
    round_location_counts = {}
    round_conflicts = {}
    coord_to_urbanicity = {}
    all_row_counts = Counter()
    all_location_counts = Counter()

    for round_name, path in paths.items():
        row_counts = Counter()
        coord_seen = {}
        conflicts = 0

        with open(path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            idx_ur = header.index('URBRUR')
            idx_lat = header.index('EA_GPS_LA')
            idx_lon = header.index('EA_GPS_LO')

            for row in reader:
                urbanicity = urbanicity_map.get(row[idx_ur].strip())
                lat = row[idx_lat].strip()
                lon = row[idx_lon].strip()

                if not urbanicity or not lat or not lon:
                    continue

                row_counts[urbanicity] += 1
                coord = (norm_coord(lat), norm_coord(lon))

                if coord in coord_seen and coord_seen[coord] != urbanicity:
                    conflicts += 1
                else:
                    coord_seen.setdefault(coord, urbanicity)
                    coord_to_urbanicity.setdefault(coord, urbanicity)

        location_counts = Counter(coord_seen.values())
        round_row_counts[round_name] = row_counts
        round_location_counts[round_name] = location_counts
        round_conflicts[round_name] = conflicts
        all_row_counts.update(row_counts)
        all_location_counts.update(location_counts)

    return {
        'round_row_counts': round_row_counts,
        'round_location_counts': round_location_counts,
        'round_conflicts': round_conflicts,
        'all_row_counts': all_row_counts,
        'all_location_counts': all_location_counts,
        'coord_to_urbanicity': coord_to_urbanicity,
    }

def analyze_test_split(root, coord_to_urbanicity):
    image_counts = Counter()
    location_labels = {}
    unmatched_images = 0
    total_images = 0

    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue

        with os.scandir(class_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue

                total_images += 1
                name = entry.name
                if not name.startswith('sentinel_image_') or not name.endswith('.tif'):
                    unmatched_images += 1
                    continue

                stem = name[len('sentinel_image_'):-len('.tif')]
                parts = stem.split('_')
                if len(parts) < 3:
                    unmatched_images += 1
                    continue

                coord = (norm_coord(parts[0]), norm_coord(parts[1]))
                urbanicity = coord_to_urbanicity.get(coord)
                if urbanicity is None:
                    unmatched_images += 1
                    continue

                image_counts[urbanicity] += 1
                location_labels.setdefault(coord, urbanicity)

    location_counts = Counter(location_labels.values())
    return {
        'total_images': total_images,
        'matched_images': sum(image_counts.values()),
        'unmatched_images': unmatched_images,
        'image_counts': image_counts,
        'location_counts': location_counts,
    }

# %% [cell 2]
survey_summary = analyze_surveys(survey_paths)
coord_to_urbanicity = survey_summary['coord_to_urbanicity']

survey_row_rows = []
survey_location_rows = []

for round_name in ['R7', 'R8', 'R9']:
    row_counts = survey_summary['round_row_counts'][round_name]
    row_total = sum(row_counts.values())
    survey_row_rows.append({
        'round': round_name,
        'total_rows': row_total,
        'rural_rows': row_counts['Rural'],
        'urban_rows': row_counts['Urban'],
        'rural_pct': round(pct(row_counts['Rural'], row_total), 2),
        'urban_pct': round(pct(row_counts['Urban'], row_total), 2),
        'conflicting_coord_labels': survey_summary['round_conflicts'][round_name],
    })

    location_counts = survey_summary['round_location_counts'][round_name]
    location_total = sum(location_counts.values())
    survey_location_rows.append({
        'round': round_name,
        'total_locations': location_total,
        'rural_locations': location_counts['Rural'],
        'urban_locations': location_counts['Urban'],
        'rural_pct': round(pct(location_counts['Rural'], location_total), 2),
        'urban_pct': round(pct(location_counts['Urban'], location_total), 2),
    })

display_rows(survey_row_rows, 'Survey Rows by Round')
display_rows(survey_location_rows, 'Unique Survey Locations by Round')

all_row_counts = survey_summary['all_row_counts']
all_row_total = sum(all_row_counts.values())
all_location_counts = survey_summary['all_location_counts']
all_location_total = sum(all_location_counts.values())

overall_rows = [
    {
        'scope': 'All survey rows (R7 + R8 + R9)',
        'total': all_row_total,
        'rural_count': all_row_counts['Rural'],
        'urban_count': all_row_counts['Urban'],
        'rural_pct': round(pct(all_row_counts['Rural'], all_row_total), 2),
        'urban_pct': round(pct(all_row_counts['Urban'], all_row_total), 2),
    },
    {
        'scope': 'All survey unique locations (R7 + R8 + R9)',
        'total': all_location_total,
        'rural_count': all_location_counts['Rural'],
        'urban_count': all_location_counts['Urban'],
        'rural_pct': round(pct(all_location_counts['Rural'], all_location_total), 2),
        'urban_pct': round(pct(all_location_counts['Urban'], all_location_total), 2),
    },
]

display_rows(overall_rows, 'Overall Survey Composition')

# %% [cell 3]
test_split_summary = {
    split_name: analyze_test_split(root, coord_to_urbanicity)
    for split_name, root in split_test_roots.items()
}

test_image_rows = []
test_location_rows = []

for split_name, info in test_split_summary.items():
    image_total = info['matched_images']
    location_total = sum(info['location_counts'].values())

    test_image_rows.append({
        'split': split_name,
        'total_images': info['total_images'],
        'matched_images': image_total,
        'unmatched_images': info['unmatched_images'],
        'unmatched_pct': round(pct(info['unmatched_images'], info['total_images']), 2),
        'rural_images': info['image_counts']['Rural'],
        'urban_images': info['image_counts']['Urban'],
        'rural_pct': round(pct(info['image_counts']['Rural'], image_total), 2),
        'urban_pct': round(pct(info['image_counts']['Urban'], image_total), 2),
    })

    test_location_rows.append({
        'split': split_name,
        'matched_locations': location_total,
        'rural_locations': info['location_counts']['Rural'],
        'urban_locations': info['location_counts']['Urban'],
        'rural_pct': round(pct(info['location_counts']['Rural'], location_total), 2),
        'urban_pct': round(pct(info['location_counts']['Urban'], location_total), 2),
    })

display_rows(test_image_rows, 'Matched Test Images by Split')
display_rows(test_location_rows, 'Matched Test Locations by Split')

same_image_mix = len({(row['rural_pct'], row['urban_pct']) for row in test_image_rows}) == 1
same_location_mix = len({(row['rural_pct'], row['urban_pct']) for row in test_location_rows}) == 1
display(Markdown(
    f"`PW-s` and `SW-s` have identical matched image proportions: **{same_image_mix}**.  \\\n"
    f"`PW-s` and `SW-s` have identical matched location proportions: **{same_location_mix}**."
))

# %% [cell 4]
reference_split = test_location_rows[0]
reference_split_images = test_image_rows[0]

comparison_rows = []
for row in survey_location_rows:
    comparison_rows.append({
        'baseline': f"{row['round']} survey locations",
        'survey_rural_pct': row['rural_pct'],
        'test_rural_pct': reference_split['rural_pct'],
        'difference_pp': round(reference_split['rural_pct'] - row['rural_pct'], 2),
    })

comparison_rows.append({
    'baseline': 'All survey locations',
    'survey_rural_pct': round(pct(all_location_counts['Rural'], all_location_total), 2),
    'test_rural_pct': reference_split['rural_pct'],
    'difference_pp': round(reference_split['rural_pct'] - pct(all_location_counts['Rural'], all_location_total), 2),
})

comparison_rows.append({
    'baseline': 'All survey rows',
    'survey_rural_pct': round(pct(all_row_counts['Rural'], all_row_total), 2),
    'test_rural_pct': reference_split_images['rural_pct'],
    'difference_pp': round(reference_split_images['rural_pct'] - pct(all_row_counts['Rural'], all_row_total), 2),
})

display_rows(comparison_rows, 'Rural Percentage Comparison Against the Test Split')

max_round_gap = max(abs(row['difference_pp']) for row in comparison_rows[:-2])
overall_location_gap = abs(comparison_rows[-2]['difference_pp'])
overall_row_gap = abs(comparison_rows[-1]['difference_pp'])

recommendation = "No new split needed for the initial rural-vs-urban analysis."
reasoning = [
    f"The matched test-location mix is {reference_split['rural_pct']:.2f}% rural / {reference_split['urban_pct']:.2f}% urban.",
    f"That is only {overall_location_gap:.2f} percentage points away from the overall survey-location mix.",
    f"Across individual rounds, the test-location mix stays within {max_round_gap:.2f} percentage points of each round's survey-location mix.",
    f"At the image level, the matched test mix is {reference_split_images['rural_pct']:.2f}% rural / {reference_split_images['urban_pct']:.2f}% urban, which is {overall_row_gap:.2f} percentage points more rural than the overall survey-row mix.",
    f"Only {test_image_rows[0]['unmatched_images']} of {test_image_rows[0]['total_images']} test images ({test_image_rows[0]['unmatched_pct']:.2f}%) remain unmatched after coordinate normalization.",
]

display(Markdown('## Recommendation'))
display(Markdown(f"**{recommendation}**"))
for item in reasoning:
    display(Markdown(f"- {item}"))

display(Markdown('## Suggested Next Step'))
display(Markdown(
    'Use the existing `PW-s` / `SW-s` test sets and report rural-vs-urban metrics separately on the **matched** test samples first. '
    'If the rural/urban performance gap turns out to be large, then it becomes worth designing a dedicated stratified split or a balanced evaluation subset.'
))

# %% [cell 5]
from collections import defaultdict
from statistics import mean

prediction_roots = {
    'PW-s': RUNS_ROOT / "dinov2-knn" / "PW-s" / "teacher_checkpoint.pth" / "confusion",
    'SW-s': RUNS_ROOT / "dinov2-knn" / "SW-s" / "teacher_checkpoint.pth" / "confusion",
}

service_info = {
    'PW-s': {
        'name': 'Piped Water',
        'negative_label': 'no_pipedwater',
        'positive_label': 'pipedwater',
    },
    'SW-s': {
        'name': 'Sewage system access',
        'negative_label': 'no_sewage',
        'positive_label': 'sewage',
    },
}

area_order = {'Overall': 0, 'Rural': 1, 'Urban': 2}

def parse_prediction_coord(path_value):
    name = Path(path_value).name
    if not name.startswith('sentinel_image_') or not name.endswith('.tif'):
        return None

    stem = name[len('sentinel_image_'):-len('.tif')]
    parts = stem.split('_')
    if len(parts) < 3:
        return None

    try:
        return (norm_coord(parts[0]), norm_coord(parts[1]))
    except (TypeError, ValueError):
        return None

def safe_div(num, den):
    return num / den if den else 0.0

def to_pct(value):
    return round(100.0 * value, 2)

def binary_metrics(rows, negative_label, positive_label):
    n = len(rows)
    if n == 0:
        return {
            'n': 0,
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'balanced_accuracy': 0.0,
            'positive_precision': 0.0,
            'positive_recall': 0.0,
            'positive_f1': 0.0,
            'negative_recall': 0.0,
            'positive_support': 0,
            'negative_support': 0,
            'mean_confidence': 0.0,
        }

    tp = sum(row['true_label'] == positive_label and row['pred_label'] == positive_label for row in rows)
    tn = sum(row['true_label'] == negative_label and row['pred_label'] == negative_label for row in rows)
    fp = sum(row['true_label'] == negative_label and row['pred_label'] == positive_label for row in rows)
    fn = sum(row['true_label'] == positive_label and row['pred_label'] == negative_label for row in rows)

    pos_precision = safe_div(tp, tp + fp)
    pos_recall = safe_div(tp, tp + fn)
    neg_recall = safe_div(tn, tn + fp)
    pos_f1 = safe_div(2 * pos_precision * pos_recall, pos_precision + pos_recall)
    neg_precision = safe_div(tn, tn + fn)
    neg_f1 = safe_div(2 * neg_precision * neg_recall, neg_precision + neg_recall)
    mean_confidence = mean(float(row['confidence']) for row in rows)

    return {
        'n': n,
        'accuracy': safe_div(tp + tn, n),
        'macro_f1': (pos_f1 + neg_f1) / 2.0,
        'balanced_accuracy': (pos_recall + neg_recall) / 2.0,
        'positive_precision': pos_precision,
        'positive_recall': pos_recall,
        'positive_f1': pos_f1,
        'negative_recall': neg_recall,
        'positive_support': sum(row['true_label'] == positive_label for row in rows),
        'negative_support': sum(row['true_label'] == negative_label for row in rows),
        'mean_confidence': mean_confidence,
    }

def aggregate_location_predictions(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['coord'], row['urbanicity'])].append(row)

    location_rows = []
    for (coord, urbanicity), items in grouped.items():
        label_counts = Counter(item['pred_label'] for item in items)
        pred_label = max(
            sorted(label_counts),
            key=lambda label: (
                label_counts[label],
                mean(float(item['confidence']) for item in items if item['pred_label'] == label),
                label,
            ),
        )
        location_rows.append({
            'coord': coord,
            'urbanicity': urbanicity,
            'true_label': items[0]['true_label'],
            'pred_label': pred_label,
            'confidence': mean(float(item['confidence']) for item in items),
            'image_count': len(items),
        })

    return location_rows

def load_prediction_rows(prediction_roots, coord_to_urbanicity):
    matched_rows = []
    coverage_rows = []

    for service_key, prediction_root in prediction_roots.items():
        prediction_files = sorted(
            prediction_root.glob('test_predictions_k*.csv'),
            key=lambda path: int(path.stem.split('_k')[-1]),
        )
        for prediction_path in prediction_files:
            k = int(prediction_path.stem.split('_k')[-1])
            total_predictions = 0
            matched_predictions = 0

            with open(prediction_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_predictions += 1
                    coord = parse_prediction_coord(row['path'])
                    urbanicity = coord_to_urbanicity.get(coord) if coord else None
                    if urbanicity is None:
                        continue

                    matched_predictions += 1
                    matched_rows.append({
                        'service_key': service_key,
                        'service_name': service_info[service_key]['name'],
                        'k': k,
                        'coord': coord,
                        'urbanicity': urbanicity,
                        'true_label': row['true_label'],
                        'pred_label': row['pred_label'],
                        'confidence': row['confidence'],
                    })

            coverage_rows.append({
                'service': service_info[service_key]['name'],
                'k': k,
                'total_predictions': total_predictions,
                'matched_predictions': matched_predictions,
                'unmatched_predictions': total_predictions - matched_predictions,
                'matched_pct': round(pct(matched_predictions, total_predictions), 2),
            })

    return matched_rows, coverage_rows

# %% [cell 6]
matched_prediction_rows, prediction_coverage_rows = load_prediction_rows(prediction_roots, coord_to_urbanicity)

image_metric_rows = []
image_gap_rows = []
location_metric_rows = []
location_gap_rows = []
best_image_rows = []
best_location_rows = []
best_image_summary = {}
best_location_summary = {}

for service_key, info in service_info.items():
    service_rows = [row for row in matched_prediction_rows if row['service_key'] == service_key]
    by_k = defaultdict(list)
    for row in service_rows:
        by_k[row['k']].append(row)

    location_rows_by_k = {
        k: aggregate_location_predictions(rows)
        for k, rows in by_k.items()
    }

    best_image_k = max(
        sorted(by_k),
        key=lambda k: binary_metrics(by_k[k], info['negative_label'], info['positive_label'])['accuracy'],
    )
    best_location_k = max(
        sorted(location_rows_by_k),
        key=lambda k: binary_metrics(location_rows_by_k[k], info['negative_label'], info['positive_label'])['accuracy'],
    )

    best_image_summary[service_key] = {
        'k': best_image_k,
        'overall': binary_metrics(by_k[best_image_k], info['negative_label'], info['positive_label']),
        'rural': binary_metrics([row for row in by_k[best_image_k] if row['urbanicity'] == 'Rural'], info['negative_label'], info['positive_label']),
        'urban': binary_metrics([row for row in by_k[best_image_k] if row['urbanicity'] == 'Urban'], info['negative_label'], info['positive_label']),
    }
    best_location_summary[service_key] = {
        'k': best_location_k,
        'overall': binary_metrics(location_rows_by_k[best_location_k], info['negative_label'], info['positive_label']),
        'rural': binary_metrics([row for row in location_rows_by_k[best_location_k] if row['urbanicity'] == 'Rural'], info['negative_label'], info['positive_label']),
        'urban': binary_metrics([row for row in location_rows_by_k[best_location_k] if row['urbanicity'] == 'Urban'], info['negative_label'], info['positive_label']),
    }

    for k in sorted(by_k):
        image_rows = by_k[k]
        location_rows = location_rows_by_k[k]

        image_metrics_by_area = {}
        location_metrics_by_area = {}

        for area in ['Overall', 'Rural', 'Urban']:
            area_image_rows = image_rows if area == 'Overall' else [row for row in image_rows if row['urbanicity'] == area]
            image_metrics = binary_metrics(area_image_rows, info['negative_label'], info['positive_label'])
            image_metrics_by_area[area] = image_metrics
            image_metric_rows.append({
                'service': info['name'],
                'k': k,
                'area': area,
                'samples': image_metrics['n'],
                'accuracy_pct': to_pct(image_metrics['accuracy']),
                'macro_f1_pct': to_pct(image_metrics['macro_f1']),
                'balanced_accuracy_pct': to_pct(image_metrics['balanced_accuracy']),
                'positive_precision_pct': to_pct(image_metrics['positive_precision']),
                'positive_recall_pct': to_pct(image_metrics['positive_recall']),
                'positive_f1_pct': to_pct(image_metrics['positive_f1']),
                'negative_recall_pct': to_pct(image_metrics['negative_recall']),
                'positive_support': image_metrics['positive_support'],
                'negative_support': image_metrics['negative_support'],
                'mean_confidence': round(image_metrics['mean_confidence'], 4),
            })

            area_location_rows = location_rows if area == 'Overall' else [row for row in location_rows if row['urbanicity'] == area]
            location_metrics = binary_metrics(area_location_rows, info['negative_label'], info['positive_label'])
            location_metrics_by_area[area] = location_metrics
            location_metric_rows.append({
                'service': info['name'],
                'k': k,
                'area': area,
                'locations': location_metrics['n'],
                'accuracy_pct': to_pct(location_metrics['accuracy']),
                'macro_f1_pct': to_pct(location_metrics['macro_f1']),
                'balanced_accuracy_pct': to_pct(location_metrics['balanced_accuracy']),
                'positive_precision_pct': to_pct(location_metrics['positive_precision']),
                'positive_recall_pct': to_pct(location_metrics['positive_recall']),
                'positive_f1_pct': to_pct(location_metrics['positive_f1']),
                'negative_recall_pct': to_pct(location_metrics['negative_recall']),
                'positive_support': location_metrics['positive_support'],
                'negative_support': location_metrics['negative_support'],
            })

        image_gap_rows.append({
            'service': info['name'],
            'k': k,
            'urban_minus_rural_accuracy_pp': round(to_pct(image_metrics_by_area['Urban']['accuracy']) - to_pct(image_metrics_by_area['Rural']['accuracy']), 2),
            'urban_minus_rural_macro_f1_pp': round(to_pct(image_metrics_by_area['Urban']['macro_f1']) - to_pct(image_metrics_by_area['Rural']['macro_f1']), 2),
            'urban_minus_rural_positive_recall_pp': round(to_pct(image_metrics_by_area['Urban']['positive_recall']) - to_pct(image_metrics_by_area['Rural']['positive_recall']), 2),
        })

        location_gap_rows.append({
            'service': info['name'],
            'k': k,
            'urban_minus_rural_accuracy_pp': round(to_pct(location_metrics_by_area['Urban']['accuracy']) - to_pct(location_metrics_by_area['Rural']['accuracy']), 2),
            'urban_minus_rural_macro_f1_pp': round(to_pct(location_metrics_by_area['Urban']['macro_f1']) - to_pct(location_metrics_by_area['Rural']['macro_f1']), 2),
            'urban_minus_rural_positive_recall_pp': round(to_pct(location_metrics_by_area['Urban']['positive_recall']) - to_pct(location_metrics_by_area['Rural']['positive_recall']), 2),
        })

    for area in ['Overall', 'Rural', 'Urban']:
        metrics = best_image_summary[service_key][area.lower()] if area != 'Overall' else best_image_summary[service_key]['overall']
        best_image_rows.append({
            'service': info['name'],
            'best_k': best_image_k,
            'area': area,
            'samples': metrics['n'],
            'accuracy_pct': to_pct(metrics['accuracy']),
            'macro_f1_pct': to_pct(metrics['macro_f1']),
            'balanced_accuracy_pct': to_pct(metrics['balanced_accuracy']),
            'positive_precision_pct': to_pct(metrics['positive_precision']),
            'positive_recall_pct': to_pct(metrics['positive_recall']),
            'positive_f1_pct': to_pct(metrics['positive_f1']),
            'negative_recall_pct': to_pct(metrics['negative_recall']),
            'positive_support': metrics['positive_support'],
            'negative_support': metrics['negative_support'],
        })

        location_metrics = best_location_summary[service_key][area.lower()] if area != 'Overall' else best_location_summary[service_key]['overall']
        best_location_rows.append({
            'service': info['name'],
            'best_k': best_location_k,
            'area': area,
            'locations': location_metrics['n'],
            'accuracy_pct': to_pct(location_metrics['accuracy']),
            'macro_f1_pct': to_pct(location_metrics['macro_f1']),
            'balanced_accuracy_pct': to_pct(location_metrics['balanced_accuracy']),
            'positive_precision_pct': to_pct(location_metrics['positive_precision']),
            'positive_recall_pct': to_pct(location_metrics['positive_recall']),
            'positive_f1_pct': to_pct(location_metrics['positive_f1']),
            'negative_recall_pct': to_pct(location_metrics['negative_recall']),
            'positive_support': location_metrics['positive_support'],
            'negative_support': location_metrics['negative_support'],
        })

prediction_coverage_rows = sorted(prediction_coverage_rows, key=lambda row: (row['service'], row['k']))
image_metric_rows = sorted(image_metric_rows, key=lambda row: (row['service'], row['k'], area_order[row['area']]))
image_gap_rows = sorted(image_gap_rows, key=lambda row: (row['service'], row['k']))
location_metric_rows = sorted(location_metric_rows, key=lambda row: (row['service'], row['k'], area_order[row['area']]))
location_gap_rows = sorted(location_gap_rows, key=lambda row: (row['service'], row['k']))
best_image_rows = sorted(best_image_rows, key=lambda row: (row['service'], area_order[row['area']]))
best_location_rows = sorted(best_location_rows, key=lambda row: (row['service'], area_order[row['area']]))

display_rows(prediction_coverage_rows, 'Prediction Coverage After Filename Matching')
display_rows(image_metric_rows, 'Image-Level Metrics by k and Area')
display_rows(image_gap_rows, 'Image-Level Urban Minus Rural Gaps (percentage points)')
display_rows(best_image_rows, 'Best Image-Level k per Service')
display_rows(location_metric_rows, 'Location-Level Metrics by k and Area')
display_rows(location_gap_rows, 'Location-Level Urban Minus Rural Gaps (percentage points)')
display_rows(best_location_rows, 'Best Location-Level k per Service')

# %% [cell 7]
pw_image = best_image_summary['PW-s']
sw_image = best_image_summary['SW-s']
pw_location = best_location_summary['PW-s']
sw_location = best_location_summary['SW-s']

matched_prediction_example = prediction_coverage_rows[0]

pw_image_accuracy_gap = to_pct(pw_image['urban']['accuracy']) - to_pct(pw_image['rural']['accuracy'])
pw_image_recall_gap = to_pct(pw_image['urban']['positive_recall']) - to_pct(pw_image['rural']['positive_recall'])
sw_image_accuracy_gap = to_pct(sw_image['urban']['accuracy']) - to_pct(sw_image['rural']['accuracy'])
sw_image_recall_gap = to_pct(sw_image['urban']['positive_recall']) - to_pct(sw_image['rural']['positive_recall'])

pw_location_accuracy_gap = to_pct(pw_location['urban']['accuracy']) - to_pct(pw_location['rural']['accuracy'])
sw_location_accuracy_gap = to_pct(sw_location['urban']['accuracy']) - to_pct(sw_location['rural']['accuracy'])

display(Markdown('## DINOv2 Findings Summary'))
display(Markdown(
    f"- Filename matching recovered **{matched_prediction_example['matched_predictions']:,} / {matched_prediction_example['total_predictions']:,}** test predictions per service "
    f"(**{matched_prediction_example['matched_pct']:.2f}%**), leaving **{matched_prediction_example['unmatched_predictions']:,}** unmatched images."
))
display(Markdown(
    f"- **Piped Water** is clearly easier in urban areas at the image level. At the best image-level setting (**k={pw_image['k']}**), "
    f"urban accuracy is **{to_pct(pw_image['urban']['accuracy']):.2f}%** versus **{to_pct(pw_image['rural']['accuracy']):.2f}%** in rural areas "
    f"(**{pw_image_accuracy_gap:.2f} pp** urban minus rural). Urban positive-class recall is **{to_pct(pw_image['urban']['positive_recall']):.2f}%** "
    f"versus **{to_pct(pw_image['rural']['positive_recall']):.2f}%** (**{pw_image_recall_gap:.2f} pp**)."
))
display(Markdown(
    f"- **Sewage system access** shows the opposite pattern if you only look at accuracy: at the best image-level setting (**k={sw_image['k']}**), "
    f"urban accuracy is **{to_pct(sw_image['urban']['accuracy']):.2f}%** versus **{to_pct(sw_image['rural']['accuracy']):.2f}%** in rural areas "
    f"(**{sw_image_accuracy_gap:.2f} pp** urban minus rural). But that rural advantage is driven by the dominant negative class. "
    f"For the positive class itself, urban recall is **{to_pct(sw_image['urban']['positive_recall']):.2f}%** versus **{to_pct(sw_image['rural']['positive_recall']):.2f}%** "
    f"(**{sw_image_recall_gap:.2f} pp** urban minus rural)."
))
display(Markdown(
    f"- The location-level view tells the same story. **Piped Water** remains better in urban areas at its best location-level setting (**k={pw_location['k']}**), "
    f"with an urban-minus-rural accuracy gap of **{pw_location_accuracy_gap:.2f} pp**. "
    f"For **Sewage system access**, the best location-level setting is **k={sw_location['k']}** and the urban-minus-rural accuracy gap is **{sw_location_accuracy_gap:.2f} pp**, "
    f"again favoring rural locations on accuracy while urban locations stay stronger on positive-class recovery."
))
