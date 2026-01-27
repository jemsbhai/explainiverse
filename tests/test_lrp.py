# tests/test_lrp.py
"""
Tests for Layer-wise Relevance Propagation (LRP) explainer.

LRP decomposes network predictions back to input features using a conservation
principle. Unlike gradient-based methods, LRP propagates relevance scores
layer-by-layer through the network using specific propagation rules.

Key Properties:
- Conservation: Sum of relevances at each layer equals the output
- Layer-wise decomposition: Relevance flows backward through layers
- Multiple rules: Different rules for different layer types and use cases

Propagation Rules:
- LRP-0: Basic rule (no stabilization)
- LRP-ε (epsilon): Adds small constant for numerical stability (default)
- LRP-γ (gamma): Enhances positive contributions
- LRP-αβ (alpha-beta): Separates positive/negative contributions
- LRP-z⁺ (z-plus): Only considers positive weights

These tests require PyTorch to be installed. They will be skipped
if torch is not available.

Reference:
    Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K. R., & Samek, W. (2015).
    On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise
    Relevance Propagation. PLOS ONE.
    https://doi.org/10.1371/journal.pone.0130140
"""

import pytest
import numpy as np

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_classifier():
    """Create a simple PyTorch classifier for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 3)
    )
    
    # Initialize with deterministic weights for reproducibility
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    return model


@pytest.fixture
def simple_regressor():
    """Create a simple PyTorch regressor for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    return model


@pytest.fixture
def deep_classifier():
    """Create a deeper network for testing layer-wise propagation."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 3)
    )
    
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    return model


@pytest.fixture
def linear_model():
    """Create a linear model (no nonlinearities) for testing conservation."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    model = nn.Linear(4, 3)
    
    torch.manual_seed(42)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    
    return model


@pytest.fixture
def model_with_batchnorm():
    """Create a model with BatchNorm layers."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.BatchNorm1d(8),
        nn.ReLU(),
        nn.Linear(8, 3)
    )
    
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    # Set to eval mode for consistent BatchNorm behavior
    model.eval()
    
    return model


@pytest.fixture
def model_with_dropout():
    """Create a model with Dropout layers."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(8, 3)
    )
    
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    # Set to eval mode so dropout is disabled
    model.eval()
    
    return model


@pytest.fixture
def sample_data():
    """Create sample input data."""
    np.random.seed(42)
    X = np.random.randn(10, 4).astype(np.float32)
    return X


@pytest.fixture
def positive_sample_data():
    """Create sample data with all positive values (useful for z-plus rule)."""
    np.random.seed(42)
    X = np.abs(np.random.randn(10, 4)).astype(np.float32) + 0.1
    return X


@pytest.fixture
def feature_names():
    return ["feature_0", "feature_1", "feature_2", "feature_3"]


@pytest.fixture
def class_names():
    return ["class_a", "class_b", "class_c"]


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestLRPBasic:
    """Basic functionality tests for LRP."""
    
    def test_lrp_creation_default(self, simple_classifier, feature_names, class_names):
        """LRP explainer can be created with default parameters."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        assert explainer.feature_names == feature_names
        assert explainer.class_names == class_names
        assert explainer.rule == "epsilon"  # default rule
    
    def test_lrp_creation_epsilon_rule(self, simple_classifier, feature_names, class_names):
        """LRP explainer can be created with epsilon rule."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="epsilon",
            epsilon=1e-6
        )
        
        assert explainer.rule == "epsilon"
        assert explainer.epsilon == 1e-6
    
    def test_lrp_creation_gamma_rule(self, simple_classifier, feature_names, class_names):
        """LRP explainer can be created with gamma rule."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="gamma",
            gamma=0.25
        )
        
        assert explainer.rule == "gamma"
        assert explainer.gamma == 0.25
    
    def test_lrp_creation_alpha_beta_rule(self, simple_classifier, feature_names, class_names):
        """LRP explainer can be created with alpha-beta rule."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="alpha_beta",
            alpha=2.0,
            beta=1.0
        )
        
        assert explainer.rule == "alpha_beta"
        assert explainer.alpha == 2.0
        assert explainer.beta == 1.0
    
    def test_lrp_creation_z_plus_rule(self, simple_classifier, feature_names, class_names):
        """LRP explainer can be created with z-plus rule."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="z_plus"
        )
        
        assert explainer.rule == "z_plus"
    
    def test_lrp_rejects_invalid_rule(self, simple_classifier, feature_names, class_names):
        """LRP raises error for invalid rule."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        with pytest.raises(ValueError, match="rule"):
            LRPExplainer(
                model=adapter,
                feature_names=feature_names,
                class_names=class_names,
                rule="invalid_rule"
            )
    
    def test_lrp_rejects_non_pytorch_model(self, feature_names):
        """LRP raises error for non-PyTorch models."""
        from explainiverse.explainers.gradient import LRPExplainer
        from explainiverse.adapters import SklearnAdapter
        from sklearn.linear_model import LogisticRegression
        
        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 4), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)
        
        with pytest.raises(TypeError, match="PyTorch"):
            LRPExplainer(
                model=adapter,
                feature_names=feature_names
            )
    
    def test_lrp_alpha_beta_constraint(self, simple_classifier, feature_names, class_names):
        """LRP alpha-beta rule enforces alpha - beta = 1 constraint."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        # Valid: alpha=2, beta=1 (2-1=1)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="alpha_beta",
            alpha=2.0,
            beta=1.0
        )
        assert explainer.alpha - explainer.beta == 1.0
        
        # Invalid: alpha=2, beta=2 (2-2=0 != 1)
        with pytest.raises(ValueError, match="alpha.*beta"):
            LRPExplainer(
                model=adapter,
                feature_names=feature_names,
                class_names=class_names,
                rule="alpha_beta",
                alpha=2.0,
                beta=2.0
            )


# =============================================================================
# Classification Tests
# =============================================================================

class TestLRPClassification:
    """Tests for LRP on classification models."""
    
    def test_lrp_explain_classification(self, simple_classifier, sample_data, feature_names, class_names):
        """LRP produces valid explanations for classification."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "LRP"
        assert "feature_attributions" in explanation.explanation_data
        assert len(explanation.explanation_data["feature_attributions"]) == 4
    
    def test_lrp_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """LRP respects target_class parameter."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation_0 = explainer.explain(sample_data[0], target_class=0)
        explanation_1 = explainer.explain(sample_data[0], target_class=1)
        
        # Different target classes should produce different attributions
        attr_0 = list(explanation_0.explanation_data["feature_attributions"].values())
        attr_1 = list(explanation_1.explanation_data["feature_attributions"].values())
        
        assert not np.allclose(attr_0, attr_1)
        assert explanation_0.target_class == "class_a"
        assert explanation_1.target_class == "class_b"
    
    def test_lrp_auto_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """LRP uses predicted class when target_class is None."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        # Get predicted class
        predictions = adapter.predict(sample_data[0].reshape(1, -1))
        predicted_class = int(np.argmax(predictions))
        
        explanation = explainer.explain(sample_data[0])
        
        # Should explain the predicted class
        assert explanation.target_class == class_names[predicted_class]
    
    def test_lrp_attribution_shape(self, simple_classifier, sample_data, feature_names, class_names):
        """LRP produces attributions with correct shape."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        attributions_raw = explanation.explanation_data["attributions_raw"]
        assert len(attributions_raw) == len(feature_names)
    
    def test_lrp_attributions_finite(self, simple_classifier, sample_data, feature_names, class_names):
        """LRP attributions are finite (no NaN or Inf)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        for i in range(len(sample_data)):
            explanation = explainer.explain(sample_data[i])
            attributions = list(explanation.explanation_data["feature_attributions"].values())
            assert all(np.isfinite(a) for a in attributions), f"Non-finite attribution at index {i}"


# =============================================================================
# Regression Tests
# =============================================================================

class TestLRPRegression:
    """Tests for LRP on regression models."""
    
    def test_lrp_explain_regression(self, simple_regressor, sample_data, feature_names):
        """LRP produces valid explanations for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert isinstance(explanation, Explanation)
        assert "feature_attributions" in explanation.explanation_data
    
    def test_lrp_regression_no_class_names(self, simple_regressor, sample_data, feature_names):
        """LRP handles regression without class_names."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        # Target class should be "output" for regression
        assert explanation.target_class == "output"
    
    def test_lrp_regression_attributions_finite(self, simple_regressor, sample_data, feature_names):
        """LRP attributions are finite for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)


# =============================================================================
# Conservation Property Tests (Critical for LRP)
# =============================================================================

class TestLRPConservation:
    """Tests for LRP's conservation property.
    
    The conservation property states that the sum of relevances at each layer
    should equal the relevance at the layer above (and ultimately the output).
    This is a fundamental property that distinguishes LRP from gradient methods.
    """
    
    def test_conservation_linear_model(self, linear_model, sample_data, feature_names, class_names):
        """LRP satisfies conservation for linear model (exact)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(linear_model, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="epsilon",
            epsilon=0  # No stabilization for exact conservation
        )
        
        instance = sample_data[0]
        target_class = 0
        
        explanation = explainer.explain(instance, target_class=target_class)
        
        # Get model output for target class
        with torch.no_grad():
            output = linear_model(torch.tensor(instance.reshape(1, -1)))
            target_output = output[0, target_class].item()
        
        # Sum of attributions should equal the output
        attribution_sum = sum(explanation.explanation_data["feature_attributions"].values())
        
        # For linear model with epsilon=0, should be exact
        np.testing.assert_allclose(attribution_sum, target_output, rtol=1e-4)
    
    def test_conservation_approximate_epsilon(self, simple_classifier, sample_data, feature_names, class_names):
        """LRP approximately satisfies conservation with epsilon rule."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="epsilon",
            epsilon=1e-6
        )
        
        instance = sample_data[0]
        target_class = 0
        
        explanation = explainer.explain(instance, target_class=target_class)
        
        # Get model output
        with torch.no_grad():
            simple_classifier.eval()
            output = simple_classifier(torch.tensor(instance.reshape(1, -1)))
            target_output = output[0, target_class].item()
        
        # Sum of attributions should be close to output (approximate due to epsilon)
        attribution_sum = sum(explanation.explanation_data["feature_attributions"].values())
        
        # Allow some tolerance for epsilon stabilization
        relative_error = abs(attribution_sum - target_output) / (abs(target_output) + 1e-10)
        assert relative_error < 0.1, f"Conservation violated: sum={attribution_sum}, output={target_output}"
    
    def test_conservation_return_delta(self, simple_classifier, sample_data, feature_names, class_names):
        """LRP can return convergence delta for conservation check."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(
            sample_data[0],
            target_class=0,
            return_convergence_delta=True
        )
        
        assert "convergence_delta" in explanation.explanation_data
        assert "target_output" in explanation.explanation_data
        assert "attribution_sum" in explanation.explanation_data
        
        # Delta should be small
        delta = explanation.explanation_data["convergence_delta"]
        assert delta < 0.5, f"Convergence delta too large: {delta}"


# =============================================================================
# Propagation Rules Tests
# =============================================================================

class TestLRPRules:
    """Tests for different LRP propagation rules."""
    
    def test_epsilon_rule_stabilizes(self, simple_classifier, sample_data, feature_names, class_names):
        """Epsilon rule produces stable results even for small activations."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        # With very small epsilon
        explainer_small = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="epsilon",
            epsilon=1e-10
        )
        
        # With larger epsilon
        explainer_large = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="epsilon",
            epsilon=0.1
        )
        
        explanation_small = explainer_small.explain(sample_data[0])
        explanation_large = explainer_large.explain(sample_data[0])
        
        # Both should produce finite results
        attr_small = list(explanation_small.explanation_data["feature_attributions"].values())
        attr_large = list(explanation_large.explanation_data["feature_attributions"].values())
        
        assert all(np.isfinite(a) for a in attr_small)
        assert all(np.isfinite(a) for a in attr_large)
    
    def test_gamma_rule_enhances_positive(self, simple_classifier, positive_sample_data, feature_names, class_names):
        """Gamma rule enhances positive contributions."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer_epsilon = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="epsilon"
        )
        
        explainer_gamma = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="gamma",
            gamma=0.25
        )
        
        instance = positive_sample_data[0]
        
        explanation_epsilon = explainer_epsilon.explain(instance, target_class=0)
        explanation_gamma = explainer_gamma.explain(instance, target_class=0)
        
        # Gamma rule should produce different results
        attr_epsilon = np.array(explanation_epsilon.explanation_data["attributions_raw"])
        attr_gamma = np.array(explanation_gamma.explanation_data["attributions_raw"])
        
        assert not np.allclose(attr_epsilon, attr_gamma)
    
    def test_alpha_beta_separates_contributions(self, simple_classifier, sample_data, feature_names, class_names):
        """Alpha-beta rule separates positive and negative contributions."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        # Standard alpha=2, beta=1 (favors positive)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="alpha_beta",
            alpha=2.0,
            beta=1.0
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert "feature_attributions" in explanation.explanation_data
        # Alpha-beta can produce both positive and negative attributions
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_z_plus_non_negative(self, simple_classifier, positive_sample_data, feature_names, class_names):
        """Z-plus rule produces non-negative attributions for positive inputs."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="z_plus"
        )
        
        explanation = explainer.explain(positive_sample_data[0])
        
        # Z-plus with positive inputs should give non-negative attributions
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        # Note: might have small negative values due to numerical precision
        assert all(a >= -1e-6 for a in attributions), f"Unexpected negative attributions: {attributions}"
    
    def test_different_rules_different_results(self, simple_classifier, sample_data, feature_names, class_names):
        """Different rules produce different attribution patterns."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        rules = ["epsilon", "gamma", "alpha_beta", "z_plus"]
        results = {}
        
        for rule in rules:
            kwargs = {"rule": rule}
            if rule == "alpha_beta":
                kwargs["alpha"] = 2.0
                kwargs["beta"] = 1.0
            
            explainer = LRPExplainer(
                model=adapter,
                feature_names=feature_names,
                class_names=class_names,
                **kwargs
            )
            
            explanation = explainer.explain(sample_data[0], target_class=0)
            results[rule] = np.array(explanation.explanation_data["attributions_raw"])
        
        # All rules should produce different results (pairwise comparison)
        for i, rule1 in enumerate(rules):
            for rule2 in rules[i+1:]:
                assert not np.allclose(results[rule1], results[rule2]), \
                    f"Rules {rule1} and {rule2} produced identical results"


# =============================================================================
# Composite Rules Tests
# =============================================================================

class TestLRPComposite:
    """Tests for composite LRP rules (different rules for different layers)."""
    
    def test_composite_rule_creation(self, deep_classifier, feature_names, class_names):
        """Composite rule can be configured with layer-specific rules."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(deep_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="composite"
        )
        
        # Set different rules for different layers
        # Common practice: z-plus for early layers, epsilon for middle, zero for final
        layer_rules = {
            0: "z_plus",    # First linear layer
            2: "epsilon",   # Second linear layer
            4: "epsilon",   # Third linear layer
            6: "epsilon",   # Fourth linear layer
            8: "epsilon"    # Final linear layer
        }
        
        explainer.set_composite_rule(layer_rules)
        
        assert explainer.rule == "composite"
        assert explainer._layer_rules is not None
    
    def test_composite_rule_explain(self, deep_classifier, sample_data, feature_names, class_names):
        """Composite rule produces valid explanations."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(deep_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="composite"
        )
        
        # Simple composite: z_plus for input layer, epsilon for rest
        layer_rules = {0: "z_plus"}  # Others default to epsilon
        explainer.set_composite_rule(layer_rules)
        
        explanation = explainer.explain(sample_data[0])
        
        assert "feature_attributions" in explanation.explanation_data
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestLRPBatch:
    """Tests for batch processing."""
    
    def test_batch_explain(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain processes multiple instances."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanations = explainer.explain_batch(sample_data[:5])
        
        assert len(explanations) == 5
        for exp in explanations:
            assert "feature_attributions" in exp.explanation_data
    
    def test_batch_with_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain respects target_class parameter."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanations = explainer.explain_batch(sample_data[:3], target_class=1)
        
        for exp in explanations:
            assert exp.target_class == "class_b"
    
    def test_batch_single_instance(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain handles single instance (1D input)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanations = explainer.explain_batch(sample_data[0])  # 1D input
        
        assert len(explanations) == 1
    
    def test_batch_consistent_with_individual(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch results match individual explanations."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        # Get batch explanations
        batch_explanations = explainer.explain_batch(sample_data[:3], target_class=0)
        
        # Get individual explanations
        for i in range(3):
            individual = explainer.explain(sample_data[i], target_class=0)
            batch = batch_explanations[i]
            
            ind_attr = np.array(individual.explanation_data["attributions_raw"])
            batch_attr = np.array(batch.explanation_data["attributions_raw"])
            
            np.testing.assert_allclose(ind_attr, batch_attr, rtol=1e-5)


# =============================================================================
# Layer-wise Relevance Tests
# =============================================================================

class TestLRPLayerRelevances:
    """Tests for layer-wise relevance computation."""
    
    def test_layer_relevances_available(self, deep_classifier, sample_data, feature_names, class_names):
        """Layer relevances can be retrieved for analysis."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(deep_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        result = explainer.explain_with_layer_relevances(sample_data[0], target_class=0)
        
        assert "layer_relevances" in result
        assert "input_relevances" in result
        assert len(result["layer_relevances"]) > 0
    
    def test_layer_relevances_conservation(self, deep_classifier, sample_data, feature_names, class_names):
        """Relevances are approximately conserved across layers."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(deep_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="epsilon",
            epsilon=1e-6
        )
        
        result = explainer.explain_with_layer_relevances(sample_data[0], target_class=0)
        
        layer_sums = []
        for layer_name, relevances in result["layer_relevances"].items():
            layer_sums.append(np.sum(relevances))
        
        # All layer sums should be approximately equal (conservation)
        if len(layer_sums) > 1:
            for i in range(1, len(layer_sums)):
                relative_diff = abs(layer_sums[i] - layer_sums[0]) / (abs(layer_sums[0]) + 1e-10)
                assert relative_diff < 0.2, f"Conservation violated between layers: {layer_sums}"


# =============================================================================
# Model Architecture Tests
# =============================================================================

class TestLRPArchitectures:
    """Tests for different model architectures."""
    
    def test_model_with_batchnorm(self, model_with_batchnorm, sample_data, feature_names, class_names):
        """LRP handles models with BatchNorm layers."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(model_with_batchnorm, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_model_with_dropout(self, model_with_dropout, sample_data, feature_names, class_names):
        """LRP handles models with Dropout layers (in eval mode)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(model_with_dropout, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_deep_model(self, deep_classifier, sample_data, feature_names, class_names):
        """LRP handles deep models correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(deep_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)


# =============================================================================
# Registry Integration Tests
# =============================================================================

class TestLRPRegistry:
    """Tests for registry integration."""
    
    def test_lrp_registered(self):
        """LRP is registered in default registry."""
        from explainiverse import default_registry
        
        explainers = default_registry.list_explainers()
        assert "lrp" in explainers
    
    def test_lrp_metadata(self):
        """LRP has correct metadata."""
        from explainiverse import default_registry
        
        meta = default_registry.get_meta("lrp")
        
        assert meta.scope == "local"
        assert "neural" in meta.model_types
        assert "tabular" in meta.data_types
        assert "image" in meta.data_types
        assert "Bach" in meta.paper_reference or "LRP" in meta.description
    
    def test_lrp_filter_neural(self):
        """LRP appears when filtering for neural network explainers."""
        from explainiverse import default_registry
        
        neural_explainers = default_registry.filter(model_type="neural")
        assert "lrp" in neural_explainers
    
    def test_lrp_via_registry(self, simple_classifier, feature_names, class_names):
        """LRP can be created via registry."""
        from explainiverse import default_registry
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = default_registry.create(
            "lrp",
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        assert explainer is not None
        assert explainer.feature_names == feature_names


# =============================================================================
# Comparison Tests
# =============================================================================

class TestLRPComparison:
    """Tests comparing LRP with other methods."""
    
    def test_lrp_differs_from_saliency(self, simple_classifier, sample_data, feature_names, class_names):
        """LRP produces different results than Saliency maps."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer, SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        lrp = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        saliency = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            absolute_value=False
        )
        
        lrp_exp = lrp.explain(sample_data[0], target_class=0)
        saliency_exp = saliency.explain(sample_data[0], target_class=0)
        
        lrp_attr = np.array(lrp_exp.explanation_data["attributions_raw"])
        saliency_attr = np.array(saliency_exp.explanation_data["attributions_raw"])
        
        # LRP and saliency should generally differ
        assert not np.allclose(lrp_attr, saliency_attr)
    
    def test_lrp_correlation_with_deeplift(self, simple_classifier, sample_data, feature_names, class_names):
        """LRP should correlate with DeepLIFT for ReLU networks."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer, DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        lrp = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="epsilon",
            epsilon=1e-6
        )
        
        deeplift = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        lrp_exp = lrp.explain(sample_data[0], target_class=0)
        deeplift_exp = deeplift.explain(sample_data[0], target_class=0)
        
        lrp_attr = np.array(lrp_exp.explanation_data["attributions_raw"])
        deeplift_attr = np.array(deeplift_exp.explanation_data["attributions_raw"])
        
        # Should have positive correlation for ReLU networks
        correlation = np.corrcoef(lrp_attr, deeplift_attr)[0, 1]
        assert correlation > 0.3, f"Low correlation with DeepLIFT: {correlation}"


# =============================================================================
# Determinism and Reproducibility Tests
# =============================================================================

class TestLRPDeterminism:
    """Tests for determinism and reproducibility."""
    
    def test_lrp_deterministic(self, simple_classifier, sample_data, feature_names, class_names):
        """LRP produces identical results for same input."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation_1 = explainer.explain(sample_data[0], target_class=0)
        explanation_2 = explainer.explain(sample_data[0], target_class=0)
        
        attr_1 = np.array(explanation_1.explanation_data["attributions_raw"])
        attr_2 = np.array(explanation_2.explanation_data["attributions_raw"])
        
        np.testing.assert_array_equal(attr_1, attr_2)
    
    def test_lrp_deterministic_across_rules(self, simple_classifier, sample_data, feature_names, class_names):
        """Each LRP rule is individually deterministic."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        for rule in ["epsilon", "gamma", "z_plus"]:
            kwargs = {"rule": rule}
            if rule == "alpha_beta":
                kwargs["alpha"] = 2.0
                kwargs["beta"] = 1.0
            
            explainer = LRPExplainer(
                model=adapter,
                feature_names=feature_names,
                class_names=class_names,
                **kwargs
            )
            
            exp_1 = explainer.explain(sample_data[0], target_class=0)
            exp_2 = explainer.explain(sample_data[0], target_class=0)
            
            attr_1 = np.array(exp_1.explanation_data["attributions_raw"])
            attr_2 = np.array(exp_2.explanation_data["attributions_raw"])
            
            np.testing.assert_array_equal(attr_1, attr_2, err_msg=f"Rule {rule} not deterministic")


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================

class TestLRPEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_feature(self, class_names):
        """LRP handles single feature input."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        model = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=["single_feature"],
            class_names=class_names
        )
        
        instance = np.array([0.5], dtype=np.float32)
        explanation = explainer.explain(instance)
        
        assert len(explanation.explanation_data["feature_attributions"]) == 1
    
    def test_large_feature_space(self, class_names):
        """LRP handles large feature space."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        n_features = 100
        model = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"f_{i}" for i in range(n_features)],
            class_names=class_names
        )
        
        instance = np.random.randn(n_features).astype(np.float32)
        explanation = explainer.explain(instance)
        
        assert len(explanation.explanation_data["feature_attributions"]) == n_features
    
    def test_extreme_input_values(self, simple_classifier, feature_names, class_names):
        """LRP handles extreme input values."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        # Large values
        large_instance = np.array([1000, -1000, 500, -500], dtype=np.float32)
        explanation = explainer.explain(large_instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_zero_input(self, simple_classifier, feature_names, class_names):
        """LRP handles zero input."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        zero_instance = np.zeros(4, dtype=np.float32)
        explanation = explainer.explain(zero_instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_near_zero_activations(self, simple_classifier, feature_names, class_names):
        """LRP handles near-zero activations gracefully (epsilon stabilization)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            rule="epsilon",
            epsilon=1e-6
        )
        
        # Very small values that might cause near-zero activations
        small_instance = np.array([1e-8, 1e-8, 1e-8, 1e-8], dtype=np.float32)
        explanation = explainer.explain(small_instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)


# =============================================================================
# Type Safety Tests
# =============================================================================

class TestLRPTypeSafety:
    """Tests for proper type handling (Python native types)."""
    
    def test_attributions_are_python_floats(self, simple_classifier, sample_data, feature_names, class_names):
        """Feature attributions are Python floats, not numpy types."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        for fname, value in explanation.explanation_data["feature_attributions"].items():
            assert type(value) is float, f"Expected float, got {type(value)}"
    
    def test_attributions_raw_are_python_floats(self, simple_classifier, sample_data, feature_names, class_names):
        """Raw attributions list contains Python floats."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        for value in explanation.explanation_data["attributions_raw"]:
            assert type(value) is float, f"Expected float, got {type(value)}"


# =============================================================================
# CNN Architecture Tests (Conv2d, Pooling, etc.)
# =============================================================================

class TestLRPConv2d:
    """Tests for LRP on convolutional neural networks."""
    
    @pytest.fixture
    def simple_cnn(self):
        """Create a simple CNN for testing."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 3)
        )
        
        torch.manual_seed(42)
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model.eval()
        return model
    
    @pytest.fixture
    def cnn_with_batchnorm(self):
        """Create a CNN with BatchNorm2d layers."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 3)
        )
        
        torch.manual_seed(42)
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model.eval()
        return model
    
    @pytest.fixture
    def image_data(self):
        """Create sample image data (batch, channels, height, width)."""
        np.random.seed(42)
        # 8x8 grayscale images
        return np.random.randn(5, 1, 8, 8).astype(np.float32)
    
    @pytest.fixture
    def image_feature_names(self):
        """Feature names for 8x8 image (64 pixels)."""
        return [f"pixel_{i}" for i in range(64)]
    
    def test_lrp_conv2d_epsilon(self, simple_cnn, image_data, image_feature_names, class_names):
        """LRP epsilon rule works on Conv2d layers."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=image_feature_names,
            class_names=class_names,
            rule="epsilon"
        )
        
        explanation = explainer.explain(image_data[0])
        
        attributions = explanation.explanation_data["attributions_raw"]
        assert len(attributions) == 64  # 8x8 image
        assert all(np.isfinite(a) for a in attributions)
    
    def test_lrp_conv2d_gamma(self, simple_cnn, image_data, image_feature_names, class_names):
        """LRP gamma rule works on Conv2d layers."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=image_feature_names,
            class_names=class_names,
            rule="gamma",
            gamma=0.25
        )
        
        explanation = explainer.explain(image_data[0])
        
        attributions = explanation.explanation_data["attributions_raw"]
        assert all(np.isfinite(a) for a in attributions)
    
    def test_lrp_conv2d_z_plus(self, simple_cnn, image_feature_names, class_names):
        """LRP z-plus rule works on Conv2d layers."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=image_feature_names,
            class_names=class_names,
            rule="z_plus"
        )
        
        # Use positive image data for z_plus
        np.random.seed(42)
        positive_image = np.abs(np.random.randn(1, 8, 8)).astype(np.float32) + 0.1
        
        explanation = explainer.explain(positive_image)
        
        attributions = explanation.explanation_data["attributions_raw"]
        assert all(np.isfinite(a) for a in attributions)
    
    def test_lrp_maxpool2d(self, simple_cnn, image_data, image_feature_names, class_names):
        """LRP handles MaxPool2d layers correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=image_feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(image_data[0])
        
        # Should produce valid attributions through maxpool
        attributions = explanation.explanation_data["attributions_raw"]
        assert all(np.isfinite(a) for a in attributions)
    
    def test_lrp_adaptive_avgpool(self, simple_cnn, image_data, image_feature_names, class_names):
        """LRP handles AdaptiveAvgPool2d layers correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=image_feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(image_data[0])
        
        attributions = explanation.explanation_data["attributions_raw"]
        assert all(np.isfinite(a) for a in attributions)
    
    def test_lrp_cnn_with_batchnorm2d(self, cnn_with_batchnorm, image_data, image_feature_names, class_names):
        """LRP handles CNN with BatchNorm2d layers."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(cnn_with_batchnorm, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=image_feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(image_data[0])
        
        attributions = explanation.explanation_data["attributions_raw"]
        assert all(np.isfinite(a) for a in attributions)
    
    def test_lrp_cnn_batch_processing(self, simple_cnn, image_data, image_feature_names, class_names):
        """LRP batch processing works on CNN models."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=image_feature_names,
            class_names=class_names
        )
        
        explanations = explainer.explain_batch(image_data[:3])
        
        assert len(explanations) == 3
        for exp in explanations:
            attributions = exp.explanation_data["attributions_raw"]
            assert all(np.isfinite(a) for a in attributions)
    
    def test_lrp_cnn_different_rules_different_results(self, simple_cnn, image_data, image_feature_names, class_names):
        """Different LRP rules produce different results on CNN."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        
        results = {}
        for rule in ["epsilon", "gamma", "z_plus"]:
            explainer = LRPExplainer(
                model=adapter,
                feature_names=image_feature_names,
                class_names=class_names,
                rule=rule
            )
            
            explanation = explainer.explain(image_data[0], target_class=0)
            results[rule] = np.array(explanation.explanation_data["attributions_raw"])
        
        # Rules should produce different results
        assert not np.allclose(results["epsilon"], results["gamma"])
        assert not np.allclose(results["epsilon"], results["z_plus"])


# =============================================================================
# Additional Activation Layer Tests
# =============================================================================

class TestLRPActivations:
    """Tests for different activation functions."""
    
    def test_leaky_relu(self, feature_names, class_names):
        """LRP handles LeakyReLU activation."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 3)
        )
        model.eval()
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = np.random.randn(4).astype(np.float32)
        explanation = explainer.explain(instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_elu(self, feature_names, class_names):
        """LRP handles ELU activation."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ELU(),
            nn.Linear(16, 3)
        )
        model.eval()
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = np.random.randn(4).astype(np.float32)
        explanation = explainer.explain(instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_tanh(self, feature_names, class_names):
        """LRP handles Tanh activation."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )
        model.eval()
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = np.random.randn(4).astype(np.float32)
        explanation = explainer.explain(instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_sigmoid(self, feature_names, class_names):
        """LRP handles Sigmoid activation."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.Sigmoid(),
            nn.Linear(16, 3)
        )
        model.eval()
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = np.random.randn(4).astype(np.float32)
        explanation = explainer.explain(instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)


# =============================================================================
# Compare Rules Method Tests
# =============================================================================

class TestLRPCompareRules:
    """Tests for the compare_rules utility method."""
    
    def test_compare_rules_returns_all(self, simple_classifier, sample_data, feature_names, class_names):
        """compare_rules returns results for all rules."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        comparison = explainer.compare_rules(sample_data[0], target_class=0)
        
        assert "epsilon" in comparison
        assert "gamma" in comparison
        assert "alpha_beta" in comparison
        assert "z_plus" in comparison
    
    def test_compare_rules_contains_attributions(self, simple_classifier, sample_data, feature_names, class_names):
        """compare_rules results contain attributions and metadata."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        comparison = explainer.compare_rules(sample_data[0])
        
        for rule, result in comparison.items():
            if "error" not in result:
                assert "attributions" in result
                assert "top_feature" in result
                assert "top_attribution" in result
                assert "attribution_sum" in result
    
    def test_compare_rules_subset(self, simple_classifier, sample_data, feature_names, class_names):
        """compare_rules can compare a subset of rules."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        comparison = explainer.compare_rules(
            sample_data[0],
            rules=["epsilon", "gamma"]
        )
        
        assert len(comparison) == 2
        assert "epsilon" in comparison
        assert "gamma" in comparison
        assert "z_plus" not in comparison


# =============================================================================
# Flatten Layer Tests
# =============================================================================

class TestLRPFlatten:
    """Tests for Flatten layer handling."""
    
    def test_flatten_in_network(self, class_names):
        """LRP handles explicit Flatten layers."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer
        
        # Network with explicit Flatten
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Unflatten(1, (4, 4)),  # Reshape to 4x4
            nn.Flatten(),  # Flatten back
            nn.Linear(16, 3)
        )
        model.eval()
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"f_{i}" for i in range(4)],
            class_names=class_names
        )
        
        instance = np.random.randn(4).astype(np.float32)
        explanation = explainer.explain(instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
