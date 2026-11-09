# ESE 5420 White Wine Quality Project

This project compares standardized logistic regression and standardized k-nearest
neighbors for identifying white wines with a quality score of at least 7.

## Baseline checkpoint

The checkpoint removes exact duplicate rows before splitting, creates one
stratified 80/20 development/test split with random seed 5420, and leaves the
test set unevaluated. Hyperparameters are selected on the development set with
five repeats of stratified five-fold cross-validation using ROC-AUC.

Run from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python Baseline/baseline_checkpoint.py
```

The script prints the dataset audit, selected hyperparameters, mean validation
ROC-AUC, standard deviation, and standard error. The held-out test set is not
evaluated by this checkpoint script.

## Data

The source is the public UCI Wine Quality dataset, white-wine subset:
https://archive.ics.uci.edu/dataset/186/wine+quality

The dataset is distributed under CC BY 4.0. The analysis uses the original
`winequality-white.csv` file in `Data/`.
