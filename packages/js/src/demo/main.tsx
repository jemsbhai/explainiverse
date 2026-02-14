import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
// @ts-ignore
import { ExplanationVisualizer } from '../visualizer';
// @ts-ignore
import { Explanation } from '../core/explanation';

// Predefined scenarios
const scenarios = {
    dogClassification: {
        name: 'Dog Classification',
        explainer: 'LIME',
        targetClass: 'Chihuahua',
        features: {
            'ear_shape': 0.45,
            'nose_size': 0.32,
            'tail_curl': -0.12,
            'fur_texture': 0.28,
            'leg_length': -0.08
        }
    },
    loanApproval: {
        name: 'Loan Approval',
        explainer: 'SHAP',
        targetClass: 'Approved',
        features: {
            'credit_score': 0.65,
            'income': 0.52,
            'debt_ratio': -0.38,
            'employment_years': 0.24,
            'age': 0.15
        }
    },
    medicalDiagnosis: {
        name: 'Medical Diagnosis',
        explainer: 'Anchors',
        targetClass: 'Healthy',
        features: {
            'blood_pressure': -0.42,
            'cholesterol': -0.28,
            'exercise_frequency': 0.55,
            'bmi': -0.18,
            'sleep_hours': 0.31
        }
    },
};

function DemoApp() {
    const [selectedScenario, setSelectedScenario] = useState<keyof typeof scenarios>('dogClassification');
    const [customMode, setCustomMode] = useState(false);
    const [explainerName, setExplainerName] = useState('LIME');
    const [targetClass, setTargetClass] = useState('');
    const [features, setFeatures] = useState<Record<string, number>>({});
    const [newFeatureName, setNewFeatureName] = useState('');
    const [newFeatureValue, setNewFeatureValue] = useState('');

    const getCurrentExplanation = () => {
        if (customMode) {
            return new Explanation(
                explainerName,
                targetClass || 'Custom',
                { feature_attributions: features },
                Object.keys(features)
            );
        } else {
            const scenario = scenarios[selectedScenario];
            return new Explanation(
                scenario.explainer,
                scenario.targetClass,
                { feature_attributions: scenario.features },
                Object.keys(scenario.features)
            );
        }
    };

    const addFeature = () => {
        if (newFeatureName && newFeatureValue) {
            setFeatures({ ...features, [newFeatureName]: parseFloat(newFeatureValue) });
            setNewFeatureName('');
            setNewFeatureValue('');
        }
    };

    const removeFeature = (name: string) => {
        const newFeatures = { ...features };
        delete newFeatures[name];
        setFeatures(newFeatures);
    };

    return (
        <div style={styles.container}>
            <header style={styles.header}>
                <h1 style={styles.mainTitle}>Explainiverse Visualizer Demo</h1>
                <p style={styles.subtitle}>Interactive XAI Explanation Explorer</p>
            </header>

            <div style={styles.controls}>
                <div style={styles.modeToggle}>
                    <button
                        onClick={() => setCustomMode(false)}
                        style={{
                            ...styles.modeButton,
                            ...((!customMode) ? styles.modeButtonActive : {})
                        }}
                    >
                        Predefined Scenarios
                    </button>
                    <button
                        onClick={() => setCustomMode(true)}
                        style={{
                            ...styles.modeButton,
                            ...(customMode ? styles.modeButtonActive : {})
                        }}
                    >
                        Custom Input
                    </button>
                </div>

                {!customMode ? (
                    <div style={styles.scenarioSelector}>
                        <label style={styles.label}>Select Scenario:</label>
                        <select
                            value={selectedScenario}
                            onChange={(e) => setSelectedScenario(e.target.value as keyof typeof scenarios)}
                            style={styles.select}
                        >
                            {Object.entries(scenarios).map(([key, scenario]) => (
                                <option key={key} value={key}>{scenario.name}</option>
                            ))}
                        </select>
                    </div>
                ) : (
                    <div style={styles.customInputs}>
                        <div style={styles.inputRow}>
                            <div style={styles.inputGroup}>
                                <label style={styles.label}>Explainer Method:</label>
                                <input
                                    type="text"
                                    value={explainerName}
                                    onChange={(e) => setExplainerName(e.target.value)}
                                    style={styles.input}
                                    placeholder="e.g., LIME, SHAP"
                                />
                            </div>
                            <div style={styles.inputGroup}>
                                <label style={styles.label}>Target Class:</label>
                                <input
                                    type="text"
                                    value={targetClass}
                                    onChange={(e) => setTargetClass(e.target.value)}
                                    style={styles.input}
                                    placeholder="e.g., Approved"
                                />
                            </div>
                        </div>

                        <div style={styles.featureBuilder}>
                            <h4 style={styles.featureBuilderTitle}>Add Features:</h4>
                            <div style={styles.featureInputRow}>
                                <input
                                    type="text"
                                    value={newFeatureName}
                                    onChange={(e) => setNewFeatureName(e.target.value)}
                                    style={styles.input}
                                    placeholder="Feature name"
                                />
                                <input
                                    type="number"
                                    step="0.01"
                                    value={newFeatureValue}
                                    onChange={(e) => setNewFeatureValue(e.target.value)}
                                    style={styles.input}
                                    placeholder="Attribution value"
                                />
                                <button onClick={addFeature} style={styles.addButton}>Add</button>
                            </div>

                            {Object.keys(features).length > 0 && (
                                <div style={styles.featureList}>
                                    {Object.entries(features).map(([name, value]) => (
                                        <div key={name} style={styles.featureTag}>
                                            <span>{name}: {value}</span>
                                            <button onClick={() => removeFeature(name)} style={styles.removeButton}>×</button>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            <div style={styles.visualizerContainer}>
                <ExplanationVisualizer explanation={getCurrentExplanation()} />
            </div>
        </div>
    );
}

const styles: Record<string, React.CSSProperties> = {
    container: {
        minHeight: '100vh',
        backgroundColor: '#f3f4f6',
        padding: '40px 20px',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    },
    header: {
        textAlign: 'center',
        marginBottom: '40px',
    },
    mainTitle: {
        fontSize: '42px',
        fontWeight: '800',
        color: '#111827',
        margin: '0 0 8px 0',
    },
    subtitle: {
        fontSize: '18px',
        color: '#6b7280',
        margin: 0,
    },
    controls: {
        maxWidth: '800px',
        margin: '0 auto 32px',
        backgroundColor: 'white',
        padding: '24px',
        borderRadius: '12px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    },
    modeToggle: {
        display: 'flex',
        gap: '8px',
        marginBottom: '24px',
    },
    modeButton: {
        flex: 1,
        padding: '12px',
        border: '2px solid #e5e7eb',
        backgroundColor: 'white',
        borderRadius: '8px',
        fontSize: '14px',
        fontWeight: '600',
        cursor: 'pointer',
        transition: 'all 0.2s',
    },
    modeButtonActive: {
        backgroundColor: '#3b82f6',
        color: 'white',
        borderColor: '#3b82f6',
    },
    scenarioSelector: {
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
    },
    label: {
        fontSize: '14px',
        fontWeight: '600',
        color: '#374151',
    },
    select: {
        padding: '10px',
        fontSize: '16px',
        border: '2px solid #e5e7eb',
        borderRadius: '8px',
        backgroundColor: 'white',
    },
    customInputs: {
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
    },
    inputRow: {
        display: 'flex',
        gap: '16px',
    },
    inputGroup: {
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
    },
    input: {
        padding: '10px',
        fontSize: '16px',
        border: '2px solid #e5e7eb',
        borderRadius: '8px',
    },
    featureBuilder: {
        padding: '16px',
        backgroundColor: '#f9fafb',
        borderRadius: '8px',
    },
    featureBuilderTitle: {
        margin: '0 0 12px 0',
        fontSize: '16px',
        color: '#374151',
    },
    featureInputRow: {
        display: 'flex',
        gap: '8px',
    },
    addButton: {
        padding: '10px 20px',
        backgroundColor: '#10b981',
        color: 'white',
        border: 'none',
        borderRadius: '8px',
        fontWeight: '600',
        cursor: 'pointer',
    },
    featureList: {
        marginTop: '12px',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '8px',
    },
    featureTag: {
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '6px 12px',
        backgroundColor: '#e0f2fe',
        borderRadius: '6px',
        fontSize: '14px',
    },
    removeButton: {
        backgroundColor: '#ef4444',
        color: 'white',
        border: 'none',
        borderRadius: '50%',
        width: '20px',
        height: '20px',
        cursor: 'pointer',
        fontSize: '16px',
        lineHeight: '1',
    },
    visualizerContainer: {
        maxWidth: '800px',
        margin: '0 auto',
    },
};

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <React.StrictMode>
        <DemoApp />
    </React.StrictMode>
);
