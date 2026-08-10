"""
Shared fixtures for reference validation tests.

These fixtures provide deterministic, reproducible test environments for comparing
Explainiverse outputs against canonical reference implementations (Quantus, captum,
SHAP, LIME, etc.).

Design principles:
    1. Every random operation uses a fixed seed for reproducibility.
    2. Models are trained once per session (session-scoped) to save time.
    3. Each fixture is self-documenting — expected shapes and value ranges are asserted.
    4. Three problem types covered: multiclass, binary, regression.
    5. Both sklearn and PyTorch models provided for each problem type.

Datasets:
    - Iris (4 features, 3 classes, 150 samples) — multiclass classification
    - Breast Cancer Wisconsin (30 features, 2 classes, 569 samples) — binary classification
    - Diabetes (10 features, continuous target, 442 samples) — regression
"""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
except ImportError:  # PyTorch is an optional package dependency.
    torch = None
    nn = None

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.3
MLP_EPOCHS = 500  # Enough to get >95% accuracy on Iris/BC
MLP_LR = 0.005
MLP_HIDDEN = 64
TOLERANCE_ATOL = 1e-5  # Absolute tolerance for numerical comparison
TOLERANCE_RTOL = 1e-4  # Relative tolerance for numerical comparison

# Number of test instances to use for batch comparisons
N_TEST_INSTANCES = 5


# ─────────────────────────────────────────────────────────────────────
# PyTorch model definitions
# ─────────────────────────────────────────────────────────────────────
if torch is not None:

    class TabularMLP(nn.Module):
        """Simple MLP for tabular data with a deterministic architecture."""

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:

    class TabularMLP:  # pragma: no cover - instantiated only after the skip guard
        """Placeholder that keeps non-PyTorch reference tests collectable."""


def _require_torch() -> None:
    if torch is None:
        pytest.skip("PyTorch reference fixture requires the optional torch dependency")


def _train_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = MLP_EPOCHS,
    lr: float = MLP_LR,
) -> nn.Module:
    """Train a PyTorch classification model. Returns model in eval mode."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _train_regressor(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = MLP_EPOCHS,
    lr: float = MLP_LR,
) -> nn.Module:
    """Train a PyTorch regression model. Returns model in eval mode."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────
# Dataset fixtures (session-scoped — loaded once)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def iris_data():
    """
    Iris dataset: multiclass classification (3 classes, 4 features).

    Returns dict with keys:
        X_train, X_test, y_train, y_test: scaled arrays
        scaler: fitted StandardScaler
        feature_names: list of 4 feature name strings
        class_names: list of 3 class name strings
        n_classes: 3
        n_features: 4
    """
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    result = {
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(data.feature_names),
        "class_names": list(data.target_names),
        "n_classes": 3,
        "n_features": 4,
    }

    # Sanity checks
    assert X_train_s.shape == (105, 4)
    assert X_test_s.shape == (45, 4)
    assert len(np.unique(y_train)) == 3
    assert len(np.unique(y_test)) == 3

    return result


@pytest.fixture(scope="session")
def breast_cancer_data():
    """
    Breast Cancer Wisconsin: binary classification (2 classes, 30 features).

    Returns dict with same keys as iris_data.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    result = {
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(data.feature_names),
        "class_names": list(data.target_names),
        "n_classes": 2,
        "n_features": 30,
    }

    assert X_train_s.shape[1] == 30
    assert len(np.unique(y_train)) == 2

    return result


@pytest.fixture(scope="session")
def diabetes_data():
    """
    Diabetes dataset: regression (10 features, continuous target).

    Returns dict with keys:
        X_train, X_test, y_train, y_test: scaled arrays (y also scaled)
        scaler_X: fitted StandardScaler for features
        scaler_y: fitted StandardScaler for target
        feature_names: list of 10 feature name strings
        n_features: 10
    """
    data = load_diabetes()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    scaler_X = StandardScaler().fit(X_train)
    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # Scale target too for stable training
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    result = {
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train_s,
        "y_test": y_test_s,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "feature_names": list(data.feature_names),
        "n_features": 10,
    }

    assert X_train_s.shape[1] == 10

    return result


# ─────────────────────────────────────────────────────────────────────
# PyTorch model fixtures (session-scoped — trained once)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def torch_mlp_multiclass(iris_data):
    """
    Trained PyTorch MLP for Iris multiclass classification.

    Returns the raw nn.Module in eval mode.
    Asserts test accuracy >= 90% to ensure the model is meaningful.
    """
    _require_torch()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = TabularMLP(
        input_dim=iris_data["n_features"],
        hidden_dim=MLP_HIDDEN,
        output_dim=iris_data["n_classes"],
    )
    model = _train_classifier(model, iris_data["X_train"], iris_data["y_train"])

    # Enforce the fixed predictive-accuracy precondition for downstream fixtures.
    with torch.no_grad():
        preds = model(torch.FloatTensor(iris_data["X_test"])).argmax(dim=1).numpy()
    acc = (preds == iris_data["y_test"]).mean()
    assert acc >= 0.90, f"Multiclass MLP accuracy too low: {acc:.2%}"

    return model


@pytest.fixture(scope="session")
def torch_mlp_binary(breast_cancer_data):
    """
    Trained PyTorch MLP for Breast Cancer binary classification.

    Returns the raw nn.Module in eval mode.
    Asserts test accuracy >= 90%.
    """
    _require_torch()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = TabularMLP(
        input_dim=breast_cancer_data["n_features"],
        hidden_dim=MLP_HIDDEN,
        output_dim=breast_cancer_data["n_classes"],
    )
    model = _train_classifier(model, breast_cancer_data["X_train"], breast_cancer_data["y_train"])

    with torch.no_grad():
        preds = model(torch.FloatTensor(breast_cancer_data["X_test"])).argmax(dim=1).numpy()
    acc = (preds == breast_cancer_data["y_test"]).mean()
    assert acc >= 0.90, f"Binary MLP accuracy too low: {acc:.2%}"

    return model


@pytest.fixture(scope="session")
def torch_mlp_regression(diabetes_data):
    """
    Trained PyTorch MLP for Diabetes regression.

    Returns the raw nn.Module in eval mode.
    Asserts R^2 >= 0.30 (diabetes is noisy, so moderate R^2 is expected).
    """
    _require_torch()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = TabularMLP(
        input_dim=diabetes_data["n_features"],
        hidden_dim=MLP_HIDDEN,
        output_dim=1,
    )
    model = _train_regressor(model, diabetes_data["X_train"], diabetes_data["y_train"])

    with torch.no_grad():
        preds = model(torch.FloatTensor(diabetes_data["X_test"])).squeeze().numpy()
    ss_res = np.sum((diabetes_data["y_test"] - preds) ** 2)
    ss_tot = np.sum((diabetes_data["y_test"] - diabetes_data["y_test"].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    assert r2 >= 0.30, f"Regression MLP R^2 too low: {r2:.4f}"

    return model


# ─────────────────────────────────────────────────────────────────────
# Explainiverse PyTorchAdapter fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def adapted_mlp_multiclass(torch_mlp_multiclass, iris_data):
    """PyTorchAdapter wrapping the multiclass MLP."""
    from explainiverse.adapters import PyTorchAdapter

    adapter = PyTorchAdapter(
        torch_mlp_multiclass,
        task="classification",
        feature_names=iris_data["feature_names"],
        class_names=iris_data["class_names"],
        device="cpu",
    )

    # Verify adapter produces valid probabilities
    probs = adapter.predict(iris_data["X_test"][:1])
    assert probs.shape == (1, 3), f"Expected (1,3), got {probs.shape}"
    assert np.allclose(probs.sum(), 1.0, atol=1e-5), f"Probs don't sum to 1: {probs.sum()}"

    return adapter


@pytest.fixture(scope="session")
def adapted_mlp_binary(torch_mlp_binary, breast_cancer_data):
    """PyTorchAdapter wrapping the binary MLP."""
    from explainiverse.adapters import PyTorchAdapter

    adapter = PyTorchAdapter(
        torch_mlp_binary,
        task="classification",
        feature_names=breast_cancer_data["feature_names"],
        class_names=breast_cancer_data["class_names"],
        device="cpu",
    )

    probs = adapter.predict(breast_cancer_data["X_test"][:1])
    assert probs.shape == (1, 2), f"Expected (1,2), got {probs.shape}"
    assert np.allclose(probs.sum(), 1.0, atol=1e-5)

    return adapter


@pytest.fixture(scope="session")
def adapted_mlp_regression(torch_mlp_regression, diabetes_data):
    """PyTorchAdapter wrapping the regression MLP."""
    from explainiverse.adapters import PyTorchAdapter

    adapter = PyTorchAdapter(
        torch_mlp_regression,
        task="regression",
        feature_names=diabetes_data["feature_names"],
        device="cpu",
    )

    preds = adapter.predict(diabetes_data["X_test"][:1])
    assert preds.shape[0] == 1, f"Expected batch dim 1, got {preds.shape}"

    return adapter


# ─────────────────────────────────────────────────────────────────────
# sklearn model fixtures (session-scoped)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def rf_multiclass(iris_data):
    """
    RandomForestClassifier trained on Iris.
    Asserts test accuracy >= 90%.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(iris_data["X_train"], iris_data["y_train"])

    acc = model.score(iris_data["X_test"], iris_data["y_test"])
    assert acc >= 0.90, f"RF multiclass accuracy too low: {acc:.2%}"

    return model


@pytest.fixture(scope="session")
def rf_binary(breast_cancer_data):
    """
    RandomForestClassifier trained on Breast Cancer.
    Asserts test accuracy >= 90%.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(breast_cancer_data["X_train"], breast_cancer_data["y_train"])

    acc = model.score(breast_cancer_data["X_test"], breast_cancer_data["y_test"])
    assert acc >= 0.90, f"RF binary accuracy too low: {acc:.2%}"

    return model


@pytest.fixture(scope="session")
def rf_regression(diabetes_data):
    """
    RandomForestRegressor trained on Diabetes.
    Asserts R^2 >= 0.30.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(diabetes_data["X_train"], diabetes_data["y_train"])

    r2 = model.score(diabetes_data["X_test"], diabetes_data["y_test"])
    assert r2 >= 0.30, f"RF regression R^2 too low: {r2:.4f}"

    return model


# ─────────────────────────────────────────────────────────────────────
# XGBoost model fixtures (for TreeSHAP validation)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def xgb_multiclass(iris_data):
    """
    XGBClassifier trained on Iris.
    Asserts test accuracy >= 90%.
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(iris_data["X_train"], iris_data["y_train"])

    acc = model.score(iris_data["X_test"], iris_data["y_test"])
    assert acc >= 0.90, f"XGB multiclass accuracy too low: {acc:.2%}"

    return model


@pytest.fixture(scope="session")
def xgb_binary(breast_cancer_data):
    """
    XGBClassifier trained on Breast Cancer.
    Asserts test accuracy >= 90%.
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(breast_cancer_data["X_train"], breast_cancer_data["y_train"])

    acc = model.score(breast_cancer_data["X_test"], breast_cancer_data["y_test"])
    assert acc >= 0.90, f"XGB binary accuracy too low: {acc:.2%}"

    return model


@pytest.fixture(scope="session")
def xgb_regression(diabetes_data):
    """
    XGBRegressor trained on Diabetes.
    Asserts R^2 >= 0.30.
    """
    model = xgb.XGBRegressor(n_estimators=100, random_state=SEED)
    model.fit(diabetes_data["X_train"], diabetes_data["y_train"])

    r2 = model.score(diabetes_data["X_test"], diabetes_data["y_test"])
    assert r2 >= 0.30, f"XGB regression R^2 too low: {r2:.4f}"

    return model


# ─────────────────────────────────────────────────────────────────────
# Test instance fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def iris_test_instances(iris_data):
    """
    Fixed test instances from Iris test set — one per class guaranteed.

    Returns dict with:
        instances: np.ndarray of shape (N_TEST_INSTANCES, 4)
        labels: np.ndarray of shape (N_TEST_INSTANCES,)
        indices: original indices in X_test
    """
    X_test = iris_data["X_test"]
    y_test = iris_data["y_test"]

    # Pick one instance per class, then fill remaining with class 0
    indices = []
    for cls in range(iris_data["n_classes"]):
        cls_indices = np.where(y_test == cls)[0]
        indices.append(cls_indices[0])

    # Fill to N_TEST_INSTANCES
    remaining = N_TEST_INSTANCES - len(indices)
    all_indices = np.arange(len(y_test))
    unused = [i for i in all_indices if i not in indices]
    indices.extend(unused[:remaining])

    indices = np.array(indices[:N_TEST_INSTANCES])

    return {
        "instances": X_test[indices].astype(np.float32),
        "labels": y_test[indices],
        "indices": indices,
    }


@pytest.fixture(scope="session")
def bc_test_instances(breast_cancer_data):
    """
    Fixed test instances from Breast Cancer test set — both classes.

    Returns dict with instances, labels, indices.
    """
    X_test = breast_cancer_data["X_test"]
    y_test = breast_cancer_data["y_test"]

    indices = []
    for cls in range(breast_cancer_data["n_classes"]):
        cls_indices = np.where(y_test == cls)[0]
        indices.append(cls_indices[0])

    remaining = N_TEST_INSTANCES - len(indices)
    unused = [i for i in range(len(y_test)) if i not in indices]
    indices.extend(unused[:remaining])

    indices = np.array(indices[:N_TEST_INSTANCES])

    return {
        "instances": X_test[indices].astype(np.float32),
        "labels": y_test[indices],
        "indices": indices,
    }


@pytest.fixture(scope="session")
def diabetes_test_instances(diabetes_data):
    """
    Fixed test instances from Diabetes test set.

    Returns dict with instances, targets, indices.
    """
    X_test = diabetes_data["X_test"]
    y_test = diabetes_data["y_test"]

    indices = np.arange(N_TEST_INSTANCES)

    return {
        "instances": X_test[indices].astype(np.float32),
        "targets": y_test[indices],
        "indices": indices,
    }


# ─────────────────────────────────────────────────────────────────────
# Reference attribution fixtures (computed from canonical libraries)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def captum_saliency_iris(torch_mlp_multiclass, iris_test_instances):
    """
    Saliency attributions from captum for Iris test instances.

    Returns np.ndarray of shape (N_TEST_INSTANCES, 4).
    """
    from captum.attr import Saliency

    saliency = Saliency(torch_mlp_multiclass)
    all_attrs = []

    for i in range(len(iris_test_instances["instances"])):
        x = torch.FloatTensor(iris_test_instances["instances"][i : i + 1]).requires_grad_(True)
        label = int(iris_test_instances["labels"][i])
        attr = saliency.attribute(x, target=label)
        all_attrs.append(attr.detach().numpy())

    return np.vstack(all_attrs)


@pytest.fixture(scope="session")
def captum_ig_iris(torch_mlp_multiclass, iris_test_instances):
    """
    Integrated Gradients attributions from captum for Iris test instances.

    Uses baseline of zeros (standard choice).
    Returns np.ndarray of shape (N_TEST_INSTANCES, 4).
    """
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(torch_mlp_multiclass)
    all_attrs = []

    for i in range(len(iris_test_instances["instances"])):
        x = torch.FloatTensor(iris_test_instances["instances"][i : i + 1])
        baseline = torch.zeros_like(x)
        label = int(iris_test_instances["labels"][i])
        attr = ig.attribute(x, baselines=baseline, target=label, n_steps=200)
        all_attrs.append(attr.detach().numpy())

    return np.vstack(all_attrs)


@pytest.fixture(scope="session")
def captum_deeplift_iris(torch_mlp_multiclass, iris_test_instances):
    """
    DeepLIFT attributions from captum for Iris test instances.

    Returns np.ndarray of shape (N_TEST_INSTANCES, 4).
    """
    from captum.attr import DeepLift

    dl = DeepLift(torch_mlp_multiclass)
    all_attrs = []

    for i in range(len(iris_test_instances["instances"])):
        x = torch.FloatTensor(iris_test_instances["instances"][i : i + 1])
        baseline = torch.zeros_like(x)
        label = int(iris_test_instances["labels"][i])
        attr = dl.attribute(x, baselines=baseline, target=label)
        all_attrs.append(attr.detach().numpy())

    return np.vstack(all_attrs)


@pytest.fixture(scope="session")
def shap_kernel_iris(rf_multiclass, iris_data, iris_test_instances):
    """
    KernelSHAP values from shap package for Iris test instances.

    Returns dict with:
        values: np.ndarray of shape (N_TEST_INSTANCES, 4) — for predicted class
        values_all_classes: list of arrays per class
    """
    import shap

    background = shap.sample(iris_data["X_train"], 50, random_state=SEED)
    explainer = shap.KernelExplainer(rf_multiclass.predict_proba, background)

    shap_values = explainer.shap_values(iris_test_instances["instances"])

    # Extract per-predicted-class values
    labels = iris_test_instances["labels"]
    per_class = np.array([shap_values[labels[i]][i] for i in range(len(labels))])

    return {
        "values": per_class,
        "values_all_classes": shap_values,
    }


@pytest.fixture(scope="session")
def shap_tree_iris(xgb_multiclass, iris_test_instances):
    """
    TreeSHAP values from shap package for Iris test instances.

    Returns dict with:
        values: np.ndarray of shape (N_TEST_INSTANCES, 4) — for predicted class
        values_all_classes: full shap_values array
    """
    import shap

    explainer = shap.TreeExplainer(xgb_multiclass)
    shap_values = explainer.shap_values(iris_test_instances["instances"])

    labels = iris_test_instances["labels"]

    # Handle different shap_values formats
    if isinstance(shap_values, list):
        # List of arrays, one per class
        per_class = np.array([shap_values[labels[i]][i] for i in range(len(labels))])
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (n_samples, n_features, n_classes)
        per_class = np.array([shap_values[i, :, labels[i]] for i in range(len(labels))])
    else:
        # 2D: (n_samples, n_features) — binary or regression
        per_class = shap_values

    return {
        "values": per_class,
        "values_all_classes": shap_values,
    }


# ─────────────────────────────────────────────────────────────────────
# Gradient x Input reference (simple manual computation for sanity)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gradient_x_input_iris(torch_mlp_multiclass, iris_test_instances):
    """
    Manually computed gradient x input attributions for Iris.

    This is the simplest possible attribution — useful as a sanity baseline.
    Returns np.ndarray of shape (N_TEST_INSTANCES, 4).
    """
    all_attrs = []

    for i in range(len(iris_test_instances["instances"])):
        x = torch.FloatTensor(iris_test_instances["instances"][i : i + 1]).requires_grad_(True)
        label = int(iris_test_instances["labels"][i])
        out = torch_mlp_multiclass(x)
        out[0, label].backward()
        grad = x.grad.detach().numpy()
        attr = grad * iris_test_instances["instances"][i : i + 1]
        all_attrs.append(attr)

    return np.vstack(all_attrs)


# ─────────────────────────────────────────────────────────────────────
# Quantus metric wrappers (for convenient comparison)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def quantus_metrics():
    """
    Dict of pre-configured Quantus metric instances.

    All configured with disable_warnings=True and consistent parameters
    that match Explainiverse defaults where possible.
    """
    import quantus

    return {
        "monotonicity": quantus.Monotonicity(
            features_in_step=1,
            perturb_baseline="mean",
            disable_warnings=True,
        ),
        "monotonicity_nguyen": quantus.MonotonicityCorrelation(
            nr_samples=10,
            features_in_step=1,
            perturb_baseline="mean",
            disable_warnings=True,
        ),
        "faithfulness_estimate": quantus.FaithfulnessEstimate(
            features_in_step=1,
            perturb_baseline="mean",
            disable_warnings=True,
        ),
        "faithfulness_correlation": quantus.FaithfulnessCorrelation(
            nr_runs=10,
            subset_size=2,
            perturb_baseline="mean",
            disable_warnings=True,
        ),
        "sensitivity_n": quantus.SensitivityN(
            features_in_step=1,
            n_max_percentage=0.8,
            similarity_func=quantus.similarity_func.correlation_pearson,
            perturb_baseline="mean",
            disable_warnings=True,
        ),
        "infidelity": quantus.Infidelity(
            perturb_baseline="mean",
            n_perturb_samples=10,
            disable_warnings=True,
        ),
        "pixel_flipping": quantus.PixelFlipping(
            features_in_step=1,
            perturb_baseline="mean",
            disable_warnings=True,
        ),
        "region_perturbation": quantus.RegionPerturbation(
            patch_size=1,
            regions_evaluation=5,
            perturb_baseline="mean",
            disable_warnings=True,
        ),
        "selectivity": quantus.Selectivity(
            patch_size=1,
            perturb_baseline="mean",
            disable_warnings=True,
        ),
        "irof": quantus.IterativeRemovalOfFeatures(
            segmentation_method="felzenszwalb",
            perturb_baseline="mean",
            disable_warnings=True,
        ),
        "insertion_auc": quantus.InsertionCurve(
            disable_warnings=True,
        ),
        "deletion_auc": quantus.DeletionCurve(
            disable_warnings=True,
        ),
        "road": quantus.ROAD(
            noise=0.01,
            percentages=list(range(1, 100, 2)),
            disable_warnings=True,
        ),
    }


# ─────────────────────────────────────────────────────────────────────
# Helper functions available to all reference tests
# ─────────────────────────────────────────────────────────────────────


def assert_numerical_match(
    explainiverse_value,
    reference_value,
    metric_name: str,
    atol: float = TOLERANCE_ATOL,
    rtol: float = TOLERANCE_RTOL,
):
    """
    Assert two values are numerically close, with a clear error message.

    Works with scalars, 1D arrays, and 2D arrays.
    """
    ev = np.asarray(explainiverse_value, dtype=np.float64)
    rv = np.asarray(reference_value, dtype=np.float64)

    assert ev.shape == rv.shape, (
        f"{metric_name}: shape mismatch — " f"explainiverse={ev.shape}, reference={rv.shape}"
    )

    if not np.allclose(ev, rv, atol=atol, rtol=rtol):
        max_diff = np.max(np.abs(ev - rv))
        mean_diff = np.mean(np.abs(ev - rv))
        raise AssertionError(
            f"{metric_name}: numerical mismatch — "
            f"max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
            f"atol={atol}, rtol={rtol}\n"
            f"  explainiverse: {ev}\n"
            f"  reference:     {rv}"
        )


def assert_rank_correlation(
    explainiverse_attrs: np.ndarray,
    reference_attrs: np.ndarray,
    metric_name: str,
    min_correlation: float = 0.95,
):
    """
    Enforce a configured Spearman threshold on this fixed reference fixture.

    This is a regression criterion for the selected data, seeds, and settings;
    it is not evidence of universal method equivalence.
    """
    from scipy.stats import spearmanr

    for i in range(len(explainiverse_attrs)):
        corr, pval = spearmanr(explainiverse_attrs[i], reference_attrs[i])
        assert corr >= min_correlation, (
            f"{metric_name} instance {i}: rank correlation too low — "
            f"rho={corr:.4f} (min={min_correlation})\n"
            f"  explainiverse: {explainiverse_attrs[i]}\n"
            f"  reference:     {reference_attrs[i]}"
        )
