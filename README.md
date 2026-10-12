# Medical Insurance Charge Prediction

A neural network that predicts an individual's annual medical insurance charges from six personal characteristics. A Keras regression model is trained from raw CSV, benchmarked against three non neural baselines, and saved along with every metric and figure it produces.

**Test MAE $2,754, R² 0.872** on charges averaging $13,270. The network has 161 parameters.

## Run it

```bash
pip install -r requirements.txt
python train_insurance_nn.py
```

Roughly two minutes on CPU. The script writes `results.json`, five figures to `figures/`, and the trained model to `models/insurance_nn.keras`.

## Results

| Model | Test MAE | Test RMSE | Test R² |
| --- | --- | --- | --- |
| Predict the mean | $9,593 | $12,466 | 0.00 |
| Linear regression | $4,181 | $5,796 | 0.784 |
| Random forest | $2,554 | $4,588 | 0.864 |
| **Neural network** | **$2,754** | **$4,451** | **0.872** |

Train MAE was $2,795 against test MAE $2,754, a gap of about $40. The model is neither underfitting nor overfitting.

## The model

A single hidden layer of 16 units, ReLU activation with He initialization, and a linear output. Total parameters: 161.

The architecture was chosen by comparing four candidates on a validation split carved out of the training data. The test set was not consulted during selection.

| Configuration | Parameters | Best validation MAE |
| --- | --- | --- |
| **16** | **161** | **$2,841** |
| 64-32, no dropout | 2,689 | $3,052 |
| 64-32, dropout 0.1 | 2,689 | $3,036 |
| 128-64-32, dropout 0.2 | 11,521 | $3,325 |

The smallest network won decisively. With roughly 1,070 training rows and a largely additive relationship, the extra capacity had nothing to learn and fit noise instead.

Training ran up to 300 epochs at batch size 32, with `EarlyStopping` at patience 40 and `restore_best_weights=True`, plus `ReduceLROnPlateau` at factor 0.5 and patience 15. It stopped at epoch 266 with the best validation loss at epoch 226.

Loss is MSE, the tracked metric is MAE, and the optimizer is Adam at the default learning rate of 0.001.

## Input features

| Field | Type | Values |
| --- | --- | --- |
| `age` | int | Age in years |
| `sex` | str | `male` or `female` |
| `bmi` | float | Body mass index |
| `children` | int | Number of dependents covered |
| `smoker` | str | `yes` or `no` |
| `region` | str | `northeast`, `northwest`, `southeast`, or `southwest` |

Target: `charges`, the annual billed amount in dollars.

## Preprocessing

The 80/20 split happens before any transformer is fit, so scaler means and one hot categories learn from training data only.

`StandardScaler` is applied to age, bmi and children. `OneHotEncoder` with `drop="first"` is applied to sex, smoker and region, giving 8 input features in total. There are no missing values in the dataset, so no imputation is needed.

The target is standardized as well. This keeps MSE near 1 rather than in the hundreds of millions and lets Adam run at its default learning rate. All reported metrics are converted back to dollars.

```python
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
])
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)
```

## What the model learned

**Smoking dominates every other feature.** Shuffling the smoker column costs $7,022 of MAE, against $2,621 for age and $1,656 for BMI. Region and sex are close to noise. Smokers average $32,050 in charges against $8,434 for non smokers.

**A neural network was not clearly the right tool.** Random forest matched it out of the box, winning on MAE while losing slightly on RMSE and R². On small tabular data, tree ensembles remain strong competitors.

**There is an error floor this feature set cannot cross.** The residual plot shows a tight band of small positive residuals and a long tail of large negative ones: a subset of beneficiaries cost far more than their features predict. Six demographic fields cannot encode a chronic diagnosis or a major procedure, so no amount of extra capacity fixes it.

## Project structure

```
insurance-charges-nn/
├── train_insurance_nn.py       # the full pipeline in one script
├── insurance.csv               # dataset, 1,338 records
├── requirements.txt
├── results.json                # every metric, written by the script
├── figures/
│   ├── 01_eda.png
│   ├── 02_training_history.png
│   ├── 03_predictions.png
│   ├── 04_model_comparison.png
│   └── 05_feature_importance.png
└── models/
    └── insurance_nn.keras
```

## Limitations

- A single 268 row test split gives a point estimate rather than an error bar. K fold cross validation would be more honest.
- The target is skewed at 1.52. Predicting `log(charges)` would stop a few very expensive cases from dominating MSE.
- A point estimate is the wrong output for a pricing use case. Quantile regression or prediction intervals would give a range.
- A smoker by BMI interaction term is visible in the EDA scatter and is probably worth more than any further architecture change.

## Reproducibility

The numbers above come from the pinned versions in `requirements.txt`: TensorFlow 2.17.1 and scikit-learn 1.5.2. The same code on TensorFlow 2.7 and scikit-learn 0.24 lands at roughly $2,898 MAE and R² 0.866, because default initializers and RNG streams change between versions.

Two calls in the script require scikit-learn 1.0 or newer: `get_feature_names_out`, and `handle_unknown="ignore"` combined with `drop="first"` on `OneHotEncoder`.

## Dataset

1,338 health insurance beneficiaries. Source: [TripleTen-DS/Dataset](https://raw.githubusercontent.com/TripleTen-DS/Dataset/refs/heads/main/insurance.csv)
