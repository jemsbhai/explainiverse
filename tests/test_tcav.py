# tests/test_tcav.py
"""
Tests for TCAV (Testing with Concept Activation Vectors) explainer.

TCAV provides concept-based explanations by quantifying how much a model's
predictions are influenced by high-level human concepts. Instead of
feature-level attributions, TCAV explains which concepts (e.g., "striped",
"furry") are important for predictions.

These tests require PyTorch, scikit-learn, and scipy to be installed.
They will be skipped if any dependency is not available.

Reference:
    Kim et al., 2018 - "Interpretability Beyond Feature Attribution:
    Quantitative Testing with Concept Activation Vectors" (ICML)
    https://arxiv.org/abs/1711.11279
"""

import pytest
import numpy as np

# Check if dependencies are available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

DEPENDENCIES_AVAILABLE = TORCH_AVAILABLE and SKLEARN_AVAILABLE and SCIPY_AVAILABLE

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason="PyTorch, scikit-learn, and scipy are required for TCAV tests"
)


# =============================================================================
# Helper Functions for Strict Type Checking
# =============================================================================

def assert_python_float(value, name: str):
    """
    Assert that a value is exactly a Python float, not numpy.float64.
    
    Note: isinstance(numpy.float64, float) returns True because numpy.float64
    inherits from float. We use `type(x) is float` for strict checking.
    
    Args:
        value: The value to check
        name: Descriptive name for error messages
    """
    assert type(value) is float, (
        f"{name} should be Python float, got {type(value).__name__} "
        f"(value: {value})"
    )


def assert_python_int(value, name: str):
    """
    Assert that a value is exactly a Python int, not numpy.int64.
    
    Args:
        value: The value to check
        name: Descriptive name for error messages
    """
    assert type(value) is int, (
        f"{name} should be Python int, got {type(value).__name__} "
        f"(value: {value})"
    )


def assert_python_bool(value, name: str):
    """
    Assert that a value is exactly a Python bool, not numpy.bool_.
    
    Note: isinstance(numpy.bool_, bool) returns False because numpy.bool_
    does NOT inherit from Python bool. This is the critical difference
    that caused the original test failure.
    
    Args:
        value: The value to check
        name: Descriptive name for error messages
    """
    assert type(value) is bool, (
        f"{name} should be Python bool, got {type(value).__name__} "
        f"(value: {value})"
    )


def assert_python_list(value, name: str):
    """
    Assert that a value is exactly a Python list.
    
    Args:
        value: The value to check
        name: Descriptive name for error messages
    """
    assert type(value) is list, (
        f"{name} should be Python list, got {type(value).__name__}"
    )


def assert_python_str(value, name: str):
    """
    Assert that a value is exactly a Python str.
    
    Args:
        value: The value to check
        name: Descriptive name for error messages
    """
    assert type(value) is str, (
        f"{name} should be Python str, got {type(value).__name__}"
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
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )
    
    # Initialize with deterministic weights
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    return model


@pytest.fixture
def deeper_network():
    """Create a deeper network with named layers for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    class DeepNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(8, 32)
            self.relu1 = nn.ReLU()
            self.layer2 = nn.Linear(32, 16)
            self.relu2 = nn.ReLU()
            self.layer3 = nn.Linear(16, 8)
            self.relu3 = nn.ReLU()
            self.output = nn.Linear(8, 3)
        
        def forward(self, x):
            x = self.relu1(self.layer1(x))
            x = self.relu2(self.layer2(x))
            x = self.relu3(self.layer3(x))
            return self.output(x)
    
    torch.manual_seed(42)
    model = DeepNet()
    
    return model


@pytest.fixture
def cnn_model():
    """Create a simple CNN for testing with image-like data."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
            self.fc = nn.Linear(4 * 2 * 2, 3)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    torch.manual_seed(42)
    return SimpleCNN()


@pytest.fixture
def sample_data():
    """Create sample input data."""
    np.random.seed(42)
    X = np.random.randn(50, 8).astype(np.float32)
    return X


@pytest.fixture
def concept_data():
    """
    Create synthetic concept data.
    
    Returns data where 'concept A' examples have higher values in features 0-3,
    and 'concept B' examples have higher values in features 4-7.
    """
    np.random.seed(42)
    
    # Concept A: high values in first 4 features
    concept_a = np.random.randn(30, 8).astype(np.float32)
    concept_a[:, :4] += 2.0  # Positive shift in first 4 features
    
    # Concept B: high values in last 4 features
    concept_b = np.random.randn(30, 8).astype(np.float32)
    concept_b[:, 4:] += 2.0  # Positive shift in last 4 features
    
    # Random (negative) examples
    random_examples = np.random.randn(30, 8).astype(np.float32)
    
    return {
        "concept_a": concept_a,
        "concept_b": concept_b,
        "random": random_examples
    }


@pytest.fixture
def class_names():
    return ["class_a", "class_b", "class_c"]


# =============================================================================
# Return Type Verification Tests
# =============================================================================

class TestTCAVReturnTypes:
    """
    Strict type verification tests for all TCAV return values.
    
    These tests ensure that all public API methods return Python native types
    (float, int, bool, list, str) rather than numpy types (numpy.float64,
    numpy.int64, numpy.bool_, numpy.ndarray).
    
    This is critical because:
    1. JSON serialization expects Python native types
    2. Type hints in the API indicate Python types
    3. Downstream code may use strict type checking
    4. numpy.bool_ does NOT inherit from Python bool (isinstance check fails)
    
    The original bug was that `isinstance(numpy.bool_(True), bool)` returns False,
    causing test failures. These tests use `type(x) is T` for strict verification.
    """
    
    def test_cav_accuracy_is_python_float(self, deeper_network, concept_data, class_names):
        """ConceptActivationVector.accuracy must be a Python float."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        cav = explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        assert_python_float(cav.accuracy, "cav.accuracy")
    
    def test_compute_tcav_score_returns_python_float(self, deeper_network, concept_data, sample_data, class_names):
        """compute_tcav_score must return a Python float."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        tcav_score = explainer.compute_tcav_score(
            test_inputs=sample_data,
            target_class=0,
            concept_name="concept_a"
        )
        
        assert_python_float(tcav_score, "tcav_score")
    
    def test_compute_tcav_score_with_derivatives_returns_python_float(self, deeper_network, concept_data, sample_data, class_names):
        """compute_tcav_score with return_derivatives=True must return Python float for score."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        tcav_score, derivatives = explainer.compute_tcav_score(
            test_inputs=sample_data,
            target_class=0,
            concept_name="concept_a",
            return_derivatives=True
        )
        
        assert_python_float(tcav_score, "tcav_score (with derivatives)")
        # derivatives should be numpy array (internal use)
        assert isinstance(derivatives, np.ndarray), "derivatives should be numpy array"
    
    def test_statistical_significance_test_all_types(self, deeper_network, concept_data, sample_data, class_names):
        """statistical_significance_test must return all Python native types."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        result = explainer.statistical_significance_test(
            test_inputs=sample_data,
            target_class=0,
            concept_name="concept_a",
            n_random=5,
            negative_examples=concept_data["random"]
        )
        
        # Check all float fields
        assert_python_float(result["tcav_score"], "result['tcav_score']")
        assert_python_float(result["random_mean"], "result['random_mean']")
        assert_python_float(result["random_std"], "result['random_std']")
        assert_python_float(result["t_statistic"], "result['t_statistic']")
        assert_python_float(result["p_value"], "result['p_value']")
        assert_python_float(result["effect_size"], "result['effect_size']")
        assert_python_float(result["alpha"], "result['alpha']")
        
        # Check bool field (this was the original bug)
        assert_python_bool(result["significant"], "result['significant']")
        
        # Check list field
        assert_python_list(result["random_scores"], "result['random_scores']")
        
        # Check all elements in list are Python floats
        for i, score in enumerate(result["random_scores"]):
            assert_python_float(score, f"result['random_scores'][{i}]")
    
    def test_get_most_influential_concepts_returns_python_floats(self, deeper_network, concept_data, sample_data, class_names):
        """get_most_influential_concepts must return Python floats for scores."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        explainer.learn_concept(
            concept_name="concept_b",
            concept_examples=concept_data["concept_b"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        top_concepts = explainer.get_most_influential_concepts(
            test_inputs=sample_data,
            target_class=0,
            top_k=2
        )
        
        assert_python_list(top_concepts, "top_concepts")
        
        for i, (name, score) in enumerate(top_concepts):
            assert_python_str(name, f"concept name at index {i}")
            assert_python_float(score, f"score at index {i}")
    
    def test_compare_concepts_returns_python_floats(self, deeper_network, concept_data, sample_data, class_names):
        """compare_concepts must return Python floats for all scores."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        comparison = explainer.compare_concepts(
            test_inputs=sample_data,
            target_classes=[0, 1, 2]
        )
        
        for concept_name, class_scores in comparison.items():
            assert_python_str(concept_name, f"concept name")
            for class_idx, score in class_scores.items():
                assert_python_int(class_idx, f"class_idx for {concept_name}")
                assert_python_float(score, f"score for {concept_name}, class {class_idx}")
    
    def test_explain_tcav_scores_types(self, deeper_network, concept_data, sample_data, class_names):
        """explain() tcav_scores must contain Python native types."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        explanation = explainer.explain(
            test_inputs=sample_data,
            target_class=0
        )
        
        tcav_scores = explanation.explanation_data["tcav_scores"]
        
        for concept_name, scores_dict in tcav_scores.items():
            assert_python_str(concept_name, "concept_name in tcav_scores")
            assert_python_float(scores_dict["score"], f"score for {concept_name}")
            assert_python_float(scores_dict["cav_accuracy"], f"cav_accuracy for {concept_name}")
            assert_python_int(scores_dict["positive_count"], f"positive_count for {concept_name}")
            assert_python_int(scores_dict["total_count"], f"total_count for {concept_name}")
        
        # Check other fields
        assert_python_int(explanation.explanation_data["target_class"], "target_class")
        assert_python_int(explanation.explanation_data["n_test_inputs"], "n_test_inputs")
        assert_python_list(explanation.explanation_data["concepts_analyzed"], "concepts_analyzed")
    
    def test_explain_with_significance_test_types(self, deeper_network, concept_data, sample_data, class_names):
        """explain() with significance tests must contain Python native types."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        explanation = explainer.explain(
            test_inputs=sample_data,
            target_class=0,
            run_significance_test=True,
            negative_examples=concept_data["random"],
            n_random=3
        )
        
        sig_tests = explanation.explanation_data["significance_tests"]
        
        for concept_name, result in sig_tests.items():
            assert_python_float(result["tcav_score"], f"sig_test tcav_score for {concept_name}")
            assert_python_float(result["p_value"], f"sig_test p_value for {concept_name}")
            assert_python_bool(result["significant"], f"sig_test significant for {concept_name}")
            assert_python_float(result["effect_size"], f"sig_test effect_size for {concept_name}")
            assert_python_list(result["random_scores"], f"sig_test random_scores for {concept_name}")
            
            for i, rs in enumerate(result["random_scores"]):
                assert_python_float(rs, f"random_score[{i}] for {concept_name}")
    
    def test_cav_metadata_types(self, deeper_network, concept_data, class_names):
        """CAV metadata must contain Python native types."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        cav = explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        assert_python_int(cav.metadata["n_concept_examples"], "n_concept_examples")
        assert_python_int(cav.metadata["n_negative_examples"], "n_negative_examples")
        assert_python_float(cav.metadata["test_size"], "test_size")
    
    def test_random_cav_metadata_types(self, deeper_network, concept_data, class_names):
        """Random CAV metadata must contain Python native types."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        random_cavs = explainer.learn_random_concepts(
            negative_examples=concept_data["random"],
            n_random=3
        )
        
        for i, cav in enumerate(random_cavs):
            assert_python_float(cav.accuracy, f"random_cav[{i}].accuracy")
            assert_python_int(cav.metadata["random_seed"], f"random_cav[{i}].metadata['random_seed']")


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestTCAVBasic:
    """Basic functionality tests for TCAV."""
    
    def test_tcav_creation(self, simple_classifier, class_names):
        """TCAV explainer can be created with valid parameters."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        # Get available layer names
        layers = adapter.list_layers()
        assert len(layers) > 0
        
        # Use first hidden layer
        layer_name = "1"  # ReLU after first linear
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name=layer_name,
            class_names=class_names
        )
        
        assert explainer.layer_name == layer_name
        assert explainer.class_names == class_names
        assert len(explainer.concepts) == 0
    
    def test_tcav_named_layers(self, deeper_network, class_names):
        """TCAV works with named layers."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        assert explainer.layer_name == "layer2"
    
    def test_tcav_invalid_layer(self, simple_classifier, class_names):
        """TCAV raises error for invalid layer name."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        with pytest.raises(ValueError, match="not found"):
            TCAVExplainer(
                model=adapter,
                layer_name="nonexistent_layer",
                class_names=class_names
            )
    
    def test_tcav_rejects_non_layer_model(self, class_names):
        """TCAV raises error for models without layer access."""
        from explainiverse.explainers.gradient import TCAVExplainer
        from explainiverse.adapters import SklearnAdapter
        
        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn not installed")
        
        # SklearnAdapter doesn't have get_layer_output
        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 8), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)
        
        with pytest.raises(TypeError, match="get_layer_output"):
            TCAVExplainer(
                model=adapter,
                layer_name="some_layer",
                class_names=class_names
            )
    
    def test_tcav_custom_classifier(self, deeper_network, class_names):
        """TCAV accepts different CAV classifier types."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        # Test logistic (default)
        explainer_logistic = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names,
            cav_classifier="logistic"
        )
        assert explainer_logistic.cav_classifier == "logistic"
        
        # Test SGD
        explainer_sgd = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names,
            cav_classifier="sgd"
        )
        assert explainer_sgd.cav_classifier == "sgd"


# =============================================================================
# Concept Learning Tests
# =============================================================================

class TestTCAVConceptLearning:
    """Tests for CAV (Concept Activation Vector) learning."""
    
    def test_learn_concept_basic(self, deeper_network, concept_data, class_names):
        """Learn concept successfully creates a CAV."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer, ConceptActivationVector
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        cav = explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5  # Lower threshold for test
        )
        
        assert isinstance(cav, ConceptActivationVector)
        assert cav.concept_name == "concept_a"
        assert cav.layer_name == "layer2"
        assert cav.accuracy >= 0.5
        assert "concept_a" in explainer.concepts
    
    def test_learn_multiple_concepts(self, deeper_network, concept_data, class_names):
        """Can learn multiple concepts."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        # Learn concept A
        cav_a = explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        # Learn concept B
        cav_b = explainer.learn_concept(
            concept_name="concept_b",
            concept_examples=concept_data["concept_b"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        assert len(explainer.concepts) == 2
        assert "concept_a" in explainer.concepts
        assert "concept_b" in explainer.concepts
        
        # CAVs should be different
        assert not np.allclose(cav_a.vector, cav_b.vector)
    
    def test_cav_normalization(self, deeper_network, concept_data, class_names):
        """CAV vectors are normalized to unit length."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        cav = explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        # Check unit norm
        norm = np.linalg.norm(cav.vector)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)
    
    def test_cav_low_accuracy_raises_error(self, simple_classifier, class_names):
        """Low CAV accuracy raises ValueError."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        layers = adapter.list_layers()
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name=layers[1],  # Use a layer
            class_names=class_names
        )
        
        # Create nearly identical data (hard to separate)
        np.random.seed(42)
        concept_examples = np.random.randn(20, 8).astype(np.float32)
        negative_examples = concept_examples + np.random.randn(20, 8).astype(np.float32) * 0.01
        
        with pytest.raises(ValueError, match="accuracy.*below threshold"):
            explainer.learn_concept(
                concept_name="unseparable",
                concept_examples=concept_examples,
                negative_examples=negative_examples,
                min_accuracy=0.95  # Very high threshold
            )
    
    def test_learn_random_concepts(self, deeper_network, concept_data, class_names):
        """Learn random concepts for significance testing."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        random_cavs = explainer.learn_random_concepts(
            negative_examples=concept_data["random"],
            n_random=5
        )
        
        assert len(random_cavs) == 5
        assert len(explainer.random_concepts) > 0
    
    def test_list_concepts(self, deeper_network, concept_data, class_names):
        """list_concepts returns learned concept names."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        assert explainer.list_concepts() == []
        
        explainer.learn_concept(
            concept_name="my_concept",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        assert "my_concept" in explainer.list_concepts()
    
    def test_remove_concept(self, deeper_network, concept_data, class_names):
        """remove_concept removes a learned concept."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="to_remove",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        assert "to_remove" in explainer.list_concepts()
        
        explainer.remove_concept("to_remove")
        
        assert "to_remove" not in explainer.list_concepts()


# =============================================================================
# TCAV Score Computation Tests
# =============================================================================

class TestTCAVScoreComputation:
    """Tests for TCAV score computation."""
    
    def test_compute_tcav_score(self, deeper_network, concept_data, sample_data, class_names):
        """Compute TCAV score for a concept."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        # Learn a concept
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        # Compute TCAV score
        tcav_score = explainer.compute_tcav_score(
            test_inputs=sample_data,
            target_class=0,
            concept_name="concept_a"
        )
        
        # Score should be between 0 and 1
        assert 0.0 <= tcav_score <= 1.0
    
    def test_tcav_score_returns_derivatives(self, deeper_network, concept_data, sample_data, class_names):
        """TCAV score can return directional derivatives."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        tcav_score, derivatives = explainer.compute_tcav_score(
            test_inputs=sample_data,
            target_class=0,
            concept_name="concept_a",
            return_derivatives=True
        )
        
        assert len(derivatives) == len(sample_data)
        # Use float() to ensure comparison works regardless of numpy type in derivatives
        assert tcav_score == float(np.mean(derivatives > 0))
    
    def test_tcav_score_different_classes(self, deeper_network, concept_data, sample_data, class_names):
        """TCAV scores differ for different target classes."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        score_class_0 = explainer.compute_tcav_score(
            test_inputs=sample_data,
            target_class=0,
            concept_name="concept_a"
        )
        
        score_class_1 = explainer.compute_tcav_score(
            test_inputs=sample_data,
            target_class=1,
            concept_name="concept_a"
        )
        
        # Scores for different classes should generally differ
        # (though not always for random networks)
        # Use strict type check instead of isinstance
        assert_python_float(score_class_0, "score_class_0")
        assert_python_float(score_class_1, "score_class_1")
    
    def test_tcav_score_unknown_concept_raises_error(self, deeper_network, sample_data, class_names):
        """TCAV score raises error for unknown concept."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        with pytest.raises(ValueError, match="not found"):
            explainer.compute_tcav_score(
                test_inputs=sample_data,
                target_class=0,
                concept_name="unknown_concept"
            )
    
    def test_directional_derivative(self, deeper_network, concept_data, sample_data, class_names):
        """Directional derivative computation works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        cav = explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        derivatives = explainer.compute_directional_derivative(
            inputs=sample_data[:5],
            cav=cav,
            target_class=0
        )
        
        assert len(derivatives) == 5
        assert all(np.isfinite(d) for d in derivatives)


# =============================================================================
# Statistical Significance Tests
# =============================================================================

class TestTCAVStatistics:
    """Tests for TCAV statistical significance testing."""
    
    def test_statistical_significance_test(self, deeper_network, concept_data, sample_data, class_names):
        """Statistical significance test runs and returns results."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        result = explainer.statistical_significance_test(
            test_inputs=sample_data,
            target_class=0,
            concept_name="concept_a",
            n_random=5,
            negative_examples=concept_data["random"]
        )
        
        # Check result structure
        assert "tcav_score" in result
        assert "random_scores" in result
        assert "p_value" in result
        assert "significant" in result
        assert "effect_size" in result
        
        assert 0.0 <= result["tcav_score"] <= 1.0
        assert 0.0 <= result["p_value"] <= 1.0
        
        # CRITICAL: Use strict type check for bool
        # isinstance(numpy.bool_, bool) returns False, but we want to be explicit
        assert_python_bool(result["significant"], "result['significant']")
    
    def test_significance_with_custom_alpha(self, deeper_network, concept_data, sample_data, class_names):
        """Statistical test respects custom alpha level."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        result = explainer.statistical_significance_test(
            test_inputs=sample_data,
            target_class=0,
            concept_name="concept_a",
            n_random=5,
            negative_examples=concept_data["random"],
            alpha=0.01
        )
        
        assert result["alpha"] == 0.01
        # Significance should be consistent with p-value and alpha
        assert result["significant"] == (result["p_value"] < 0.01)


# =============================================================================
# Explain Method Tests
# =============================================================================

class TestTCAVExplain:
    """Tests for the main explain() method."""
    
    def test_explain_basic(self, deeper_network, concept_data, sample_data, class_names):
        """Basic explain call returns valid Explanation."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        explanation = explainer.explain(
            test_inputs=sample_data,
            target_class=0
        )
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "TCAV"
        assert "tcav_scores" in explanation.explanation_data
        assert "concept_a" in explanation.explanation_data["tcav_scores"]
    
    def test_explain_multiple_concepts(self, deeper_network, concept_data, sample_data, class_names):
        """Explain returns scores for multiple concepts."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        explainer.learn_concept(
            concept_name="concept_b",
            concept_examples=concept_data["concept_b"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        explanation = explainer.explain(
            test_inputs=sample_data,
            target_class=0
        )
        
        tcav_scores = explanation.explanation_data["tcav_scores"]
        assert "concept_a" in tcav_scores
        assert "concept_b" in tcav_scores
    
    def test_explain_with_significance_test(self, deeper_network, concept_data, sample_data, class_names):
        """Explain includes significance tests when requested."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        explanation = explainer.explain(
            test_inputs=sample_data,
            target_class=0,
            run_significance_test=True,
            negative_examples=concept_data["random"],
            n_random=3
        )
        
        assert "significance_tests" in explanation.explanation_data
        assert "concept_a" in explanation.explanation_data["significance_tests"]
    
    def test_explain_auto_target_class(self, deeper_network, concept_data, sample_data, class_names):
        """Explain auto-selects target class when not specified."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        explanation = explainer.explain(
            test_inputs=sample_data,
            target_class=None  # Auto-select
        )
        
        # Should have determined a target class
        assert explanation.target_class in class_names
    
    def test_explain_no_concepts_raises_error(self, deeper_network, sample_data, class_names):
        """Explain raises error when no concepts are learned."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        with pytest.raises(ValueError, match="No concepts learned"):
            explainer.explain(test_inputs=sample_data, target_class=0)
    
    def test_explain_selective_concepts(self, deeper_network, concept_data, sample_data, class_names):
        """Explain can analyze only selected concepts."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        explainer.learn_concept(
            concept_name="concept_b",
            concept_examples=concept_data["concept_b"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        # Only analyze concept_a
        explanation = explainer.explain(
            test_inputs=sample_data,
            target_class=0,
            concept_names=["concept_a"]
        )
        
        tcav_scores = explanation.explanation_data["tcav_scores"]
        assert "concept_a" in tcav_scores
        assert "concept_b" not in tcav_scores


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestTCAVBatch:
    """Tests for batch processing."""
    
    def test_explain_batch(self, deeper_network, concept_data, sample_data, class_names):
        """explain_batch returns single explanation for batch."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        # TCAV typically analyzes batches together
        explanations = explainer.explain_batch(sample_data, target_class=0)
        
        assert len(explanations) == 1
        assert explanations[0].explainer_name == "TCAV"


# =============================================================================
# Comparison Methods Tests
# =============================================================================

class TestTCAVComparison:
    """Tests for concept comparison methods."""
    
    def test_get_most_influential_concepts(self, deeper_network, concept_data, sample_data, class_names):
        """get_most_influential_concepts returns ranked concepts."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        explainer.learn_concept(
            concept_name="concept_b",
            concept_examples=concept_data["concept_b"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        top_concepts = explainer.get_most_influential_concepts(
            test_inputs=sample_data,
            target_class=0,
            top_k=2
        )
        
        assert len(top_concepts) == 2
        assert all(isinstance(c, tuple) and len(c) == 2 for c in top_concepts)
        # Should be sorted by score (descending)
        assert top_concepts[0][1] >= top_concepts[1][1]
    
    def test_compare_concepts(self, deeper_network, concept_data, sample_data, class_names):
        """compare_concepts returns scores across classes."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        comparison = explainer.compare_concepts(
            test_inputs=sample_data,
            target_classes=[0, 1, 2]
        )
        
        assert "concept_a" in comparison
        assert 0 in comparison["concept_a"]
        assert 1 in comparison["concept_a"]
        assert 2 in comparison["concept_a"]


# =============================================================================
# Registry Integration Tests
# =============================================================================

class TestTCAVRegistry:
    """Tests for registry integration."""
    
    def test_tcav_registered(self):
        """TCAV is registered in default registry."""
        from explainiverse import default_registry
        
        explainers = default_registry.list_explainers()
        assert "tcav" in explainers
    
    def test_tcav_metadata(self):
        """TCAV has correct metadata."""
        from explainiverse import default_registry
        
        meta = default_registry.get_meta("tcav")
        
        assert meta.scope == "local"
        assert "neural" in meta.model_types
        assert "tabular" in meta.data_types
        assert "image" in meta.data_types
        assert "classification" in meta.task_types
        assert "Kim" in meta.paper_reference
        assert "ICML" in meta.paper_reference
    
    def test_tcav_filter_neural(self):
        """TCAV appears when filtering for neural network explainers."""
        from explainiverse import default_registry
        
        neural_explainers = default_registry.filter(model_type="neural")
        assert "tcav" in neural_explainers
    
    def test_tcav_via_registry(self, deeper_network, class_names):
        """TCAV can be created via registry."""
        from explainiverse import default_registry
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = default_registry.create(
            "tcav",
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        assert explainer is not None
        assert explainer.layer_name == "layer2"


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================

class TestTCAVEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_test_input(self, deeper_network, concept_data, class_names):
        """TCAV handles single test input."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        # Single test input
        single_input = np.random.randn(1, 8).astype(np.float32)
        
        tcav_score = explainer.compute_tcav_score(
            test_inputs=single_input,
            target_class=0,
            concept_name="concept_a"
        )
        
        # Score should be 0 or 1 for single input
        assert tcav_score in [0.0, 1.0]
    
    def test_large_batch(self, deeper_network, concept_data, class_names):
        """TCAV handles large batch of test inputs."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        # Large batch
        large_batch = np.random.randn(200, 8).astype(np.float32)
        
        tcav_score = explainer.compute_tcav_score(
            test_inputs=large_batch,
            target_class=0,
            concept_name="concept_a"
        )
        
        assert 0.0 <= tcav_score <= 1.0
    
    def test_deterministic_results(self, deeper_network, concept_data, sample_data, class_names):
        """TCAV produces deterministic results with same seed."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        # Create two explainers with same seed
        explainer1 = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names,
            random_seed=42
        )
        
        explainer2 = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names,
            random_seed=42
        )
        
        # Learn same concept
        cav1 = explainer1.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        cav2 = explainer2.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        # CAVs should be identical
        np.testing.assert_array_almost_equal(cav1.vector, cav2.vector)
    
    def test_different_layer_produces_different_cav(self, deeper_network, concept_data, class_names):
        """Different layers produce different CAVs."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer1 = TCAVExplainer(
            model=adapter,
            layer_name="layer1",
            class_names=class_names
        )
        
        explainer2 = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        cav1 = explainer1.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        cav2 = explainer2.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        # Dimensions should be different (different layer sizes)
        # or at least values should differ
        if len(cav1.vector) == len(cav2.vector):
            assert not np.allclose(cav1.vector, cav2.vector)
    
    def test_cav_repr(self, deeper_network, concept_data, class_names):
        """CAV has informative string representation."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        cav = explainer.learn_concept(
            concept_name="test_concept",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        repr_str = repr(cav)
        assert "test_concept" in repr_str
        assert "layer2" in repr_str
        assert "accuracy" in repr_str


# =============================================================================
# CNN-specific Tests
# =============================================================================

class TestTCAVCNN:
    """Tests for TCAV with CNN models."""
    
    def test_tcav_with_cnn(self, cnn_model, class_names):
        """TCAV works with CNN models."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(cnn_model, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="conv1",
            class_names=class_names
        )
        
        # Create image-like concept data
        np.random.seed(42)
        concept_images = np.random.randn(20, 1, 8, 8).astype(np.float32)
        concept_images[:, :, :4, :] += 1.0  # Top half brighter
        
        negative_images = np.random.randn(20, 1, 8, 8).astype(np.float32)
        
        cav = explainer.learn_concept(
            concept_name="bright_top",
            concept_examples=concept_images,
            negative_examples=negative_images,
            min_accuracy=0.5
        )
        
        assert cav is not None
        
        # Compute TCAV score on test images
        test_images = np.random.randn(10, 1, 8, 8).astype(np.float32)
        
        tcav_score = explainer.compute_tcav_score(
            test_inputs=test_images,
            target_class=0,
            concept_name="bright_top"
        )
        
        assert 0.0 <= tcav_score <= 1.0


# =============================================================================
# Performance Tests
# =============================================================================

class TestTCAVPerformance:
    """Tests for performance characteristics."""
    
    def test_tcav_reasonable_speed(self, deeper_network, concept_data, sample_data, class_names):
        """TCAV completes in reasonable time."""
        import time
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import TCAVExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification")
        
        explainer = TCAVExplainer(
            model=adapter,
            layer_name="layer2",
            class_names=class_names
        )
        
        start = time.time()
        
        explainer.learn_concept(
            concept_name="concept_a",
            concept_examples=concept_data["concept_a"],
            negative_examples=concept_data["random"],
            min_accuracy=0.5
        )
        
        tcav_score = explainer.compute_tcav_score(
            test_inputs=sample_data,
            target_class=0,
            concept_name="concept_a"
        )
        
        elapsed = time.time() - start
        
        # Should complete in under 10 seconds for this simple case
        assert elapsed < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
