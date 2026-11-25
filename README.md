# 🎓 Student Performance Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Swaraj-Halder/student-performance-prediction/blob/main/notebooks/student_performance_colab.ipynb)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end machine learning project that predicts student academic performance from the
[UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
(649 / 395 Portuguese secondary-school students, 30+ demographic, social and school features).

- **Regression** – predict the final grade `G3` (0–20)
- **Classification** – predict pass / fail (`G3 >= 10`)
- Automatic dataset download, preprocessing pipeline, 3-model comparison with cross-validation,
  optional hyper-parameter tuning, evaluation plots, permutation feature importance, saved model + prediction CLI.

## 🚀 Quick start

### Google Colab (no setup)
Click the badge above, then **Runtime → Run all**. (Replace `YOUR_USERNAME` in the badge and in the notebook's `REPO_URL` after pushing.)

### Local
```bash
git clone https://github.com/YOUR_USERNAME/student-performance-prediction.git
cd student-performance-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train                                  # regression, G3
python -m src.train --task classification            # pass/fail
python -m src.train --drop-grades --tune             # early-warning model + tuning
python -m src.train --subject both                   # math + Portuguese
```
The dataset is downloaded automatically into `data/` on first run.

### Use your own dataset
```bash
python -m src.train --data my_students.csv --target final_score
```
CSV (`,` or `;` separated) with one row per student and a numeric target column. Numeric and categorical
features are detected automatically. For classification, "pass" means target `>= 10`
(edit `PASS_MARK` in `src/config.py`).

### Predict
```bash
python -m src.predict --input new_students.csv --output predictions.csv
```
The input needs the same feature columns as training (the target column is not required).

## 📁 Structure
```
├── notebooks/student_performance_colab.ipynb   # Colab notebook
├── src/
│   ├── config.py        # paths & constants
│   ├── data.py          # download / load / synthetic demo data
│   ├── preprocess.py    # X/y split + sklearn ColumnTransformer
│   ├── train.py         # CV comparison, tuning, evaluation, saving
│   └── predict.py       # inference
├── tests/test_pipeline.py
├── data/  models/  outputs/    # generated (git-ignored)
└── requirements.txt
```

## 📊 Outputs (`outputs/` and `models/`)
`student_model.joblib`, `metrics.json`, `model_comparison.csv`, `predictions.png`, `feature_importance.png`

## ⚠️ Notes
- `G1` and `G2` (earlier grades) dominate prediction of `G3`. Use `--drop-grades` if you want a model that
  works *before* grades exist, e.g. for early intervention; expect lower scores.
- `--synthetic` generates fake data in the same schema for offline demos/tests only. Never report its metrics.
- Predictions are statistical estimates; use them to support, not replace, educators' judgment.

## 🧪 Tests
```bash
pytest -q
```

## 🌐 Publish to GitHub
```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/student-performance-prediction.git
git push -u origin main
```

## 📚 Citation
P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* FUBUTEC 2008.

## License
MIT
