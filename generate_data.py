"""Create a reproducible demonstration dataset for the SEC C project."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def generate_dataset(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    attendance = np.clip(rng.normal(78, 13, rows), 25, 100)
    study_hours = np.clip(rng.gamma(2.2, 1.8, rows), 0, 12)
    previous_marks = np.clip(rng.normal(62, 16, rows), 10, 100)
    assignment_marks = np.clip(rng.normal(68, 17, rows), 0, 100)
    internal_marks = np.clip(
        0.30 * previous_marks + 0.25 * assignment_marks
        + 0.25 * attendance + 0.20 * (study_hours * 8)
        + rng.normal(0, 8, rows), 0, 100
    )
    final_marks = np.clip(
        0.18 * attendance + 1.55 * study_hours
        + 0.25 * previous_marks + 0.18 * assignment_marks
        + 0.32 * internal_marks + rng.normal(0, 9, rows) - 20,
        0, 100
    )
    df = pd.DataFrame({
        "student_id": [f"SEC-C-{i:04d}" for i in range(1, rows + 1)],
        "attendance": attendance.round(1),
        "study_hours": study_hours.round(1),
        "previous_marks": previous_marks.round(1),
        "assignment_marks": assignment_marks.round(1),
        "internal_marks": internal_marks.round(1),
        "final_exam_marks": final_marks.round(1),
    })
    return df

if __name__ == "__main__":
    output = DATA_DIR / "student_performance.csv"
    generate_dataset().to_csv(output, index=False)
    print(f"Created {output}")
