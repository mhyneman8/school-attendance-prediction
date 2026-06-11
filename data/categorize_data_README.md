# categorize_data.py

Clusters school attendance records by year-over-year attendance rates using PCA and K-Means, then labels each cluster by average attendance level.

## Prerequisites

- Python 3
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

Run `cleaning_data.py` first to produce the input file `cleaned_data.csv`.

## Usage

From the project root:

```bash
python data/categorize_data.py
```

## Input

| File | Description |
|------|-------------|
| `data/cleaned_data.csv` | Preprocessed school attendance data |

**Clustering features** (attendance rates only):

- `2019_2020_attendance_rate`
- `2020_2021_attendance_rate`
- `2021_2022_attendance_rate`

**Descriptor columns** (used for profiling, not clustering):

- `2019_2020_student_count`
- `2020_2021_student_count`
- `2021_2022_student_count`

Student counts are kept separate because mixing counts with rates would skew cluster boundaries.

## Pipeline

1. **PCA** — Decorrelates the three attendance columns. Keeps enough components to explain at least 95% of variance.
2. **K selection** — Tests `k` from 2 to 8 using the elbow method (inertia) and silhouette score. Picks the `k` with the highest silhouette score.
3. **K-Means** — Fits the final model on the reduced PCA features and assigns each row a cluster.
4. **Labeling** — Ranks clusters by mean attendance and assigns human-readable labels:
   - `cluster_1_high_attendance` — highest average attendance
   - `cluster_N_low_attendance` — lowest average attendance
   - `cluster_N_mid_attendance` — clusters in between

## Output

All outputs are written to `data/output/`:

| File | Description |
|------|-------------|
| `categorized_data.csv` | Input data with `cluster` and `cluster_label` columns added |
| `elbow_silhouette.png` | Side-by-side plots of inertia and silhouette score vs. `k` |
| `clusters_pca.png` | 2D scatter of clusters projected onto the first two PCA components |

The script also prints cluster profiles (mean attendance rates and student counts), cluster sizes, and category distribution per cluster to the console.

## Configuration

Key constants at the top of the script:

- `OUTPUT_DIR` — output directory (default: `data/output`)
- `K_RANGE` — range of cluster counts to evaluate (default: 2–8)
- `random_state=42` — fixed seed for reproducible PCA and K-Means results
