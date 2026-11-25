# Lab VII - AI & Machine Learning (Weeks 5-12)

B.Sc. (CA) VII Semester, Department of Computer Science, AMU (2026-27).
Course code for the lab manual: CABSMJ7P01.

Python programs from the Lab VII manual, one file per week.
Every program was run and its output verified for the lab record.

## Weeks

| File | Week | Topics |
|---|---|---|
| week05.py | Week 5 | Data Cleaning and Integration (Titanic, IQR outliers) |
| week06.py | Week 6 | Feature Selection, Normalization, Discretization (California Housing, Iris, Adult Income) |
| week07.py | Week 7 | Supervised Learning: KNN and Linear Regression |
| week08.py | Week 8 | Decision Trees and Logistic Regression |
| week09.py | Week 9 | Artificial Neural Networks: Perceptron and MLP |
| week10.py | Week 10 | Unsupervised Learning: K-Means and Hierarchical Clustering |
| week11.py | Week 11 | Underfitting, Overfitting, and Data Separability |
| week12.py | Week 12 | Data Splitting and K-Fold Cross-Validation |

## Run

```bash
pip install pandas numpy scikit-learn matplotlib
python week05.py
```

Two programs need a data file next to the script: `train.csv` (Titanic,
Week 5) and `adult.data` (Adult Income, Week 6). Everything else uses
built-in or generated data (California Housing downloads once via sklearn).
