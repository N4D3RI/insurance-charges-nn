"""
Predicting individual medical insurance charges with a Keras neural network.

Sprint 12 project, TripleTen AI/ML.

Pipeline:
  load -> explore -> split -> preprocess (fit on train only) -> build ->
  train with callbacks -> evaluate -> visualize -> compare against baselines

The ordering matters. Everything that learns a statistic from the data
(one hot categories, scaler means, target scaler) is fit on the training
split only and then applied to the test split. Fitting before the split
leaks test information into training and inflates the score.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: no display available
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

DATA_URL = "https://raw.githubusercontent.com/TripleTen-DS/Dataset/refs/heads/main/insurance.csv"
LOCAL_CSV = "insurance.csv"
FIG_DIR = "figures"
MODEL_DIR = "models"

NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
TARGET = "charges"


# ----------------------------------------------------------------------
# Part 1: Load and explore
# ----------------------------------------------------------------------
def load_and_explore():
    """Read the dataset and print everything the brief asks about."""
    df = pd.read_csv(LOCAL_CSV) if os.path.exists(LOCAL_CSV) else pd.read_csv(DATA_URL)

    print("=" * 70)
    print("PART 1: DATA LOADING AND EXPLORATION")
    print("=" * 70)
    print(f"\nRecords: {df.shape[0]}   Columns: {df.shape[1]}")
    print("\nFirst five rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())
    print(f"Total missing: {int(df.isnull().sum().sum())}")

    print("\nDuplicate rows:", int(df.duplicated().sum()))

    print("\nTarget (charges) distribution:")
    print(df[TARGET].describe())
    print(f"  skewness: {df[TARGET].skew():.3f}")
    print(f"  median:   {df[TARGET].median():,.2f}")
    print(f"  mean:     {df[TARGET].mean():,.2f}")

    print("\nNumeric feature summary:")
    print(df[NUMERIC].describe())

    print("\nCategorical value counts:")
    for col in CATEGORICAL:
        print(f"\n  {col}:")
        print(df[col].value_counts().to_string())

    print("\nMean charges grouped by each categorical feature:")
    for col in CATEGORICAL:
        print(f"\n  by {col}:")
        print(df.groupby(col)[TARGET].agg(["mean", "median", "count"]).round(2).to_string())

    # The single most important relationship in this dataset
    smoker_gap = df.groupby("smoker")[TARGET].mean()
    print(
        f"\nSmokers pay {smoker_gap['yes'] / smoker_gap['no']:.1f}x what non smokers pay "
        f"(${smoker_gap['yes']:,.0f} vs ${smoker_gap['no']:,.0f})."
    )

    return df


def plot_eda(df):
    """Target distribution and the smoker split that dominates it."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(df[TARGET], bins=50, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Distribution of charges")
    axes[0].set_xlabel("charges ($)")
    axes[0].set_ylabel("count")

    for smoker, color in [("no", "#55A868"), ("yes", "#C44E52")]:
        axes[1].hist(
            df[df["smoker"] == smoker][TARGET],
            bins=40, alpha=0.7, label=f"smoker = {smoker}", color=color,
        )
    axes[1].set_title("Charges by smoking status")
    axes[1].set_xlabel("charges ($)")
    axes[1].legend()

    colors = df["smoker"].map({"yes": "#C44E52", "no": "#55A868"})
    axes[2].scatter(df["age"], df[TARGET], c=colors, alpha=0.6, s=18)
    axes[2].set_title("Charges vs age, colored by smoker")
    axes[2].set_xlabel("age")
    axes[2].set_ylabel("charges ($)")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {FIG_DIR}/01_eda.png")


# ----------------------------------------------------------------------
# Part 2: Preprocess
# ----------------------------------------------------------------------
def preprocess(df):
    """Split first, then fit every transformer on the training split only."""
    print("\n" + "=" * 70)
    print("PART 2: DATA PREPROCESSING")
    print("=" * 70)

    X = df[NUMERIC + CATEGORICAL]
    y = df[[TARGET]]

    # Split BEFORE fitting anything. This is the step that prevents leakage.
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"\nTrain: {len(X_train_raw)} rows    Test: {len(X_test_raw)} rows (80/20)")

    # One hot the categoricals, standardize the numerics.
    # handle_unknown='ignore' keeps an unseen category from crashing inference.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ]
    )

    X_train = preprocessor.fit_transform(X_train_raw)   # fit on train
    X_test = preprocessor.transform(X_test_raw)         # transform only

    feature_names = (
        NUMERIC
        + list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    )
    print(f"\nFeatures after encoding: {X_train.shape[1]}")
    print("  " + ", ".join(feature_names))

    # Normalize the target too. Charges span roughly $1k to $64k, which makes
    # raw MSE land in the hundreds of millions and forces an awkward learning
    # rate. Scaling the target keeps the loss and the gradients in a sane range.
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw)
    y_test = y_scaler.transform(y_test_raw)

    print(f"\nTarget scaling: mean ${y_scaler.mean_[0]:,.2f}, sd ${y_scaler.scale_[0]:,.2f}")
    print("  Predictions are converted back to dollars before any metric is reported.")

    return {
        "X_train": X_train.astype("float32"),
        "X_test": X_test.astype("float32"),
        "y_train": y_train.astype("float32"),
        "y_test": y_test.astype("float32"),
        "y_train_raw": y_train_raw.values.ravel(),
        "y_test_raw": y_test_raw.values.ravel(),
        "preprocessor": preprocessor,
        "y_scaler": y_scaler,
        "feature_names": feature_names,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
    }


# ----------------------------------------------------------------------
# Part 3: Build
# ----------------------------------------------------------------------
def build_model(n_features, hidden=(64, 32), dropout=0.1, lr=0.001):
    """A small funnel with He initialization, matching the ReLU activations."""
    model = keras.Sequential(name="insurance_charges_mlp")
    model.add(keras.Input(shape=(n_features,)))

    for i, units in enumerate(hidden):
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_initializer="he_normal",
                name=f"hidden_{i + 1}",
            )
        )
        if dropout:
            model.add(layers.Dropout(dropout, name=f"dropout_{i + 1}"))

    # Linear output: regression, the value must be free to take any magnitude.
    model.add(layers.Dense(1, activation="linear", name="output"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ----------------------------------------------------------------------
# Part 4: Train
# ----------------------------------------------------------------------
def train_model(model, data, epochs=300, batch_size=32):
    print("\n" + "=" * 70)
    print("PART 4: TRAINING")
    print("=" * 70)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=40, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=0
        ),
    ]

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_split=0.2,     # carved out of train, test stays untouched
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    ran = len(history.history["loss"])
    best = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\nRan {ran} epochs of a possible {epochs}; best val_loss at epoch {best}.")
    print("Early stopping restored the weights from that epoch.")
    return history


# ----------------------------------------------------------------------
# Part 5: Evaluate
# ----------------------------------------------------------------------
def to_dollars(y_scaled, y_scaler):
    return y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


def evaluate(model, data, history):
    print("\n" + "=" * 70)
    print("PART 5: EVALUATION")
    print("=" * 70)

    y_scaler = data["y_scaler"]

    pred_train = to_dollars(model.predict(data["X_train"], verbose=0), y_scaler)
    pred_test = to_dollars(model.predict(data["X_test"], verbose=0), y_scaler)
    true_train = data["y_train_raw"]
    true_test = data["y_test_raw"]

    def metrics(true, pred):
        return {
            "mae": float(mean_absolute_error(true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(true, pred))),
            "r2": float(r2_score(true, pred)),
        }

    m_train = metrics(true_train, pred_train)
    m_test = metrics(true_test, pred_test)

    # Keras reports loss in scaled units; convert so the numbers mean dollars.
    test_loss_scaled, test_mae_scaled = model.evaluate(
        data["X_test"], data["y_test"], verbose=0
    )
    print(f"\nKeras test loss (scaled MSE): {test_loss_scaled:.4f}")
    print(f"Keras test MAE (scaled):      {test_mae_scaled:.4f}")
    print(f"  -> in dollars, MAE = {test_mae_scaled * y_scaler.scale_[0]:,.2f}")

    print("\nMetrics in dollars:")
    print(f"{'':<10}{'MAE':>14}{'RMSE':>14}{'R2':>10}")
    print(f"{'train':<10}{m_train['mae']:>14,.2f}{m_train['rmse']:>14,.2f}{m_train['r2']:>10.4f}")
    print(f"{'test':<10}{m_test['mae']:>14,.2f}{m_test['rmse']:>14,.2f}{m_test['r2']:>10.4f}")

    gap = m_test["mae"] - m_train["mae"]
    print(f"\nTest MAE minus train MAE: ${gap:,.2f}")
    if m_test["r2"] < 0.5:
        verdict = "underfitting: the model is not capturing the structure"
    elif gap > 0.25 * m_train["mae"]:
        verdict = "overfitting: test error is materially worse than train error"
    else:
        verdict = "well balanced: train and test errors are close"
    print(f"Diagnosis: {verdict}")

    # Qualitative check the brief asks for
    print("\nTen test passengers, predicted vs actual:")
    sample = data["X_test_raw"].head(10).copy()
    sample["actual"] = true_test[:10]
    sample["predicted"] = pred_test[:10]
    sample["error"] = sample["predicted"] - sample["actual"]
    print(sample.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    return {
        "train": m_train,
        "test": m_test,
        "verdict": verdict,
        "pred_test": pred_test,
        "true_test": true_test,
        "pred_train": pred_train,
    }


def baselines(data):
    """A score means nothing without something to compare it against."""
    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)

    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte = data["y_train_raw"], data["y_test_raw"]

    results = {}
    for name, est in [
        ("predict the mean", DummyRegressor(strategy="mean")),
        ("linear regression", LinearRegression()),
        ("random forest", RandomForestRegressor(n_estimators=300, random_state=SEED)),
    ]:
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        results[name] = {
            "mae": float(mean_absolute_error(yte, pred)),
            "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "r2": float(r2_score(yte, pred)),
        }
        print(f"{name:<20} MAE ${results[name]['mae']:>10,.2f}   R2 {results[name]['r2']:.4f}")

    return results


# ----------------------------------------------------------------------
# Part 6: Visualize
# ----------------------------------------------------------------------
def plot_history(history, y_scaler):
    h = history.history
    scale = y_scaler.scale_[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(h["loss"], label="training loss")
    axes[0].plot(h["val_loss"], label="validation loss")
    axes[0].set_title("Loss (scaled MSE) over epochs")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].plot(np.array(h["mae"]) * scale, label="training MAE")
    axes[1].plot(np.array(h["val_mae"]) * scale, label="validation MAE")
    best = int(np.argmin(h["val_loss"]))
    axes[1].axvline(best, color="grey", ls="--", lw=1, label=f"best epoch ({best + 1})")
    axes[1].set_title("MAE in dollars over epochs")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MAE ($)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/02_training_history.png")


def plot_predictions(ev):
    true, pred = ev["true_test"], ev["pred_test"]
    resid = pred - true

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    lim = [0, max(true.max(), pred.max()) * 1.05]
    axes[0].scatter(true, pred, alpha=0.6, s=22, color="#4C72B0")
    axes[0].plot(lim, lim, "r--", lw=1.5, label="perfect prediction")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("actual charges ($)")
    axes[0].set_ylabel("predicted charges ($)")
    axes[0].set_title(f"Predicted vs actual (test, R2 = {ev['test']['r2']:.3f})")
    axes[0].legend()

    axes[1].scatter(pred, resid, alpha=0.6, s=22, color="#8172B2")
    axes[1].axhline(0, color="red", ls="--", lw=1.5)
    axes[1].set_xlabel("predicted charges ($)")
    axes[1].set_ylabel("residual ($)")
    axes[1].set_title("Residuals vs predicted")

    axes[2].hist(resid, bins=40, color="#CCB974", edgecolor="white")
    axes[2].axvline(0, color="red", ls="--", lw=1.5)
    axes[2].set_xlabel("residual ($)")
    axes[2].set_title("Residual distribution")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/03_predictions.png")


def plot_model_comparison(nn_test, base):
    names = ["predict the mean", "linear regression", "random forest", "neural network"]
    maes = [base[n]["mae"] for n in names[:3]] + [nn_test["mae"]]
    colors = ["#BBBBBB", "#55A868", "#CCB974", "#4C72B0"]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, maes, color=colors)
    for b, v in zip(bars, maes):
        plt.text(b.get_x() + b.get_width() / 2, v + 60, f"${v:,.0f}",
                 ha="center", fontsize=10)
    plt.ylabel("test MAE ($, lower is better)")
    plt.title("The network against three reference points")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/04_model_comparison.png")


def permutation_importance_nn(model, data):
    """Which features matter: shuffle one column, see how much MAE worsens."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (permutation)")
    print("=" * 70)

    X, y_true = data["X_test"].copy(), data["y_test_raw"]
    y_scaler = data["y_scaler"]
    base = mean_absolute_error(y_true, to_dollars(model.predict(X, verbose=0), y_scaler))

    rng = np.random.default_rng(SEED)
    rows = []
    for i, name in enumerate(data["feature_names"]):
        drops = []
        for _ in range(10):            # average over shuffles to cut noise
            Xp = X.copy()
            Xp[:, i] = rng.permutation(Xp[:, i])
            drops.append(
                mean_absolute_error(y_true, to_dollars(model.predict(Xp, verbose=0), y_scaler))
            )
        rows.append((name, float(np.mean(drops) - base)))

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\nBaseline test MAE: ${base:,.2f}")
    print("Increase in MAE when the feature is shuffled:\n")
    for name, d in rows:
        print(f"  {name:<22} +${d:>9,.2f}")

    plt.figure(figsize=(8, 4.5))
    plt.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color="#4C72B0")
    plt.xlabel("increase in test MAE when shuffled ($)")
    plt.title("Permutation feature importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {FIG_DIR}/05_feature_importance.png")
    return rows


# ----------------------------------------------------------------------
# Experiment: a few architectures, chosen on validation, not on test
# ----------------------------------------------------------------------
def run_experiments(data):
    print("\n" + "=" * 70)
    print("ARCHITECTURE EXPERIMENTS (selected on validation MAE)")
    print("=" * 70)

    configs = [
        {"hidden": (16,), "dropout": 0.0, "label": "16"},
        {"hidden": (64, 32), "dropout": 0.0, "label": "64-32, no dropout"},
        {"hidden": (64, 32), "dropout": 0.1, "label": "64-32, dropout 0.1"},
        {"hidden": (128, 64, 32), "dropout": 0.2, "label": "128-64-32, dropout 0.2"},
    ]

    scale = data["y_scaler"].scale_[0]
    results = []
    for cfg in configs:
        keras.utils.set_random_seed(SEED)
        m = build_model(data["X_train"].shape[1], cfg["hidden"], cfg["dropout"])
        hist = m.fit(
            data["X_train"], data["y_train"],
            validation_split=0.2, epochs=300, batch_size=32, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=40, restore_best_weights=True)],
        )
        val_mae = float(np.min(hist.history["val_mae"])) * scale
        cfg["params"] = int(m.count_params())
        cfg["val_mae"] = val_mae
        cfg["epochs"] = len(hist.history["loss"])
        results.append(cfg)
        print(f"  {cfg['label']:<26} params {m.count_params():>6}   "
              f"best val MAE ${val_mae:>9,.2f}   epochs {cfg['epochs']}")

    best = min(results, key=lambda c: c["val_mae"])
    print(f"\nBest on validation: {best['label']} (val MAE ${best['val_mae']:,.2f})")
    print("The final model below uses this configuration. The test set was")
    print("not consulted during selection.")
    return results, best


# ----------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_explore()
    plot_eda(df)

    data = preprocess(df)

    experiments, best_cfg = run_experiments(data)

    print("\n" + "=" * 70)
    print("PART 3: FINAL MODEL")
    print("=" * 70)
    keras.utils.set_random_seed(SEED)
    model = build_model(
        data["X_train"].shape[1],
        hidden=best_cfg["hidden"],
        dropout=best_cfg["dropout"],
    )
    print(f"Architecture chosen by the experiment above: {best_cfg['label']}\n")
    model.summary()

    history = train_model(model, data)
    ev = evaluate(model, data, history)
    base = baselines(data)

    plot_history(history, data["y_scaler"])
    plot_predictions(ev)
    plot_model_comparison(ev["test"], base)
    importances = permutation_importance_nn(model, data)

    model.save(f"{MODEL_DIR}/insurance_nn.keras")
    results = {
        "n_records": int(len(df)),
        "n_features_encoded": int(data["X_train"].shape[1]),
        "total_params": int(model.count_params()),
        "epochs_run": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "train": ev["train"],
        "test": ev["test"],
        "verdict": ev["verdict"],
        "baselines": base,
        "experiments": [
            {"config": c["label"], "params": c["params"],
             "val_mae": c["val_mae"], "epochs": c["epochs"]}
            for c in experiments
        ],
        "chosen_architecture": best_cfg["label"],
        "feature_importance": [{"feature": n, "mae_increase": d} for n, d in importances],
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Test MAE  ${ev['test']['mae']:,.2f}")
    print(f"Test RMSE ${ev['test']['rmse']:,.2f}")
    print(f"Test R2   {ev['test']['r2']:.4f}")
    print(f"Verdict:  {ev['verdict']}")
    print("\nWrote results.json, models/insurance_nn.keras, and 5 figures.")


if __name__ == "__main__":
    main()
