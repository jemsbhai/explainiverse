import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
// @ts-ignore
import { ExplanationVisualizer } from '../visualizer';
// @ts-ignore
import { Explanation } from '../core/explanation';

// Mock Metadata for interactive selection
const MOCK_DATA = {
    'Image Classification': {
        models: ['ResNet50', 'ViT-B/16', 'EfficientNet'],
        classes: ['Chihuahua', 'Persian Cat', 'Golden Retriever', 'Sports Car', 'Teapot'],
        explainers: ['LIME', 'GradCAM', 'IntegratedGradients', 'SHAP'],
        defaultFeatures: {
            'Chihuahua': { 'ear_shape': 0.45, 'nose_size': 0.32, 'tail_curl': -0.12, 'fur_texture': 0.28 },
            'Persian Cat': { 'ear_shape': -0.3, 'nose_size': -0.1, 'fluffiness': 0.8, 'whiskers': 0.4 },
            'Golden Retriever': { 'ear_flop': 0.5, 'snout_length': 0.4, 'gold_color': 0.9, 'friendliness': 0.2 },
            'Sports Car': { 'wheel_rims': 0.6, 'red_paint': 0.4, 'spoiler': 0.3, 'headlights': 0.2 },
            'Teapot': { 'spout': 0.8, 'handle': 0.7, 'lid': 0.4, 'ceramic_texture': 0.2 }
        }
    },
    'Tabular Classification': {
        models: ['XGBoost', 'RandomForest', 'LogisticRegression', 'LightGBM'],
        classes: ['Approved', 'Denied', 'Pending Review'],
        explainers: ['SHAP', 'LIME', 'Anchors', 'TreeExplainer'],
        defaultFeatures: {
            'Approved': { 'credit_score': 0.65, 'income': 0.52, 'debt_ratio': -0.38, 'employment_years': 0.24 },
            'Denied': { 'credit_score': -0.7, 'income': -0.4, 'debt_ratio': 0.8, 'missed_payments': 0.6 },
            'Pending Review': { 'credit_score': 0.1, 'income': 0.2, 'debt_ratio': 0.1, 'documents_missing': 0.5 }
        }
    },
    'Text Classification': {
        models: ['BERT', 'GPT-2', 'RoBERTa', 'DistilBERT'],
        classes: ['Positive Sentiment', 'Negative Sentiment', 'Neutral'],
        explainers: ['LIME', 'IntegratedGradients', 'Attention', 'SHAP'],
        defaultFeatures: {
            'Positive Sentiment': { 'amazing': 0.8, 'love': 0.7, 'great': 0.5, 'recommend': 0.4 },
            'Negative Sentiment': { 'terrible': 0.8, 'hate': 0.7, 'waste': 0.6, 'refund': 0.5 },
            'Neutral': { 'okay': 0.2, 'average': 0.1, 'standard': 0.1, 'feature': 0.05 }
        }
    }
};

type TaskType = keyof typeof MOCK_DATA;

function DemoApp() {
    // Selection state
    const [selectedTask, setSelectedTask] = useState<TaskType>('Image Classification');
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [selectedExplainer, setSelectedExplainer] = useState<string>('');
    const [selectedClass, setSelectedClass] = useState<string>('');

    // Feature state
    const [features, setFeatures] = useState<Record<string, number>>({});
    const [newFeatureName, setNewFeatureName] = useState('');
    const [newFeatureValue, setNewFeatureValue] = useState('');

    // Initialize defaults when task/class changes
    useEffect(() => {
        const taskData = MOCK_DATA[selectedTask];
        setSelectedModel(taskData.models[0]);
        setSelectedExplainer(taskData.explainers[0]);
        setSelectedClass(taskData.classes[0]);
    }, [selectedTask]);

    useEffect(() => {
        if (selectedClass && MOCK_DATA[selectedTask].defaultFeatures[selectedClass as any]) {
            // @ts-ignore
            setFeatures({ ...MOCK_DATA[selectedTask].defaultFeatures[selectedClass] });
        } else {
            setFeatures({});
        }
    }, [selectedClass, selectedTask]);


    const getCurrentExplanation = () => {
        return new Explanation(
            selectedExplainer,
            selectedClass,
            { feature_attributions: features },
            Object.keys(features),
            { model: selectedModel, task: selectedTask } // Storing extra metadata
        );
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

    const taskData = MOCK_DATA[selectedTask];

    return (
        <div style={styles.container}>
            <header style={styles.header}>
                <h1 style={styles.mainTitle}>Explainiverse Visualizer Demo</h1>
                <p style={styles.subtitle}>Interactive XAI Explanation Explorer</p>
            </header>

            <div style={styles.controls}>
                <div style={styles.grid}>
                    {/* Task Selection */}
                    <div style={styles.inputGroup}>
                        <label style={styles.label}>Task Type:</label>
                        <select
                            value={selectedTask}
                            onChange={(e) => setSelectedTask(e.target.value as TaskType)}
                            style={styles.select}
                        >
                            {Object.keys(MOCK_DATA).map(task => (
                                <option key={task} value={task}>{task}</option>
                            ))}
                        </select>
                    </div>

                    {/* Model Selection */}
                    <div style={styles.inputGroup}>
                        <label style={styles.label}>Model:</label>
                        <select
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            style={styles.select}
                        >
                            {taskData.models.map(model => (
                                <option key={model} value={model}>{model}</option>
                            ))}
                        </select>
                    </div>

                    {/* Explainer Selection */}
                    <div style={styles.inputGroup}>
                        <label style={styles.label}>Explainer:</label>
                        <select
                            value={selectedExplainer}
                            onChange={(e) => setSelectedExplainer(e.target.value)}
                            style={styles.select}
                        >
                            {taskData.explainers.map(explainer => (
                                <option key={explainer} value={explainer}>{explainer}</option>
                            ))}
                        </select>
                    </div>

                    {/* Class Selection */}
                    <div style={styles.inputGroup}>
                        <label style={styles.label}>Target Class:</label>
                        <select
                            value={selectedClass}
                            onChange={(e) => setSelectedClass(e.target.value)}
                            style={styles.select}
                        >
                            {taskData.classes.map(cls => (
                                <option key={cls} value={cls}>{cls}</option>
                            ))}
                        </select>
                    </div>
                </div>

                <div style={styles.divider}></div>

                <div style={styles.featureBuilder}>
                    <h4 style={styles.featureBuilderTitle}>Modify Features (Attributions):</h4>
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
                            placeholder="Value"
                        />
                        <button onClick={addFeature} style={styles.addButton}>Add</button>
                    </div>

                    {Object.keys(features).length > 0 ? (
                        <div style={styles.featureList}>
                            {Object.entries(features).map(([name, value]) => (
                                <div key={name} style={styles.featureTag}>
                                    <span>{name}: {value}</span>
                                    <button onClick={() => removeFeature(name)} style={styles.removeButton}>×</button>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p style={styles.emptyState}>No features defined for this class yet.</p>
                    )}
                </div>
            </div>

            <div style={styles.visualizerContainer}>
                {/*  Pass key to force re-render animation when context changes */}
                <ExplanationVisualizer key={`${selectedTask}-${selectedModel}-${selectedClass}`} explanation={getCurrentExplanation()} />
            </div>

            <footer style={styles.footer}>
                <p>Model: <strong>{selectedModel}</strong> | Task: <strong>{selectedTask}</strong></p>
            </footer>
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
        display: 'flex',
        flexDirection: 'column',
        gap: '24px',
    },
    grid: {
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '20px',
    },
    inputGroup: {
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
        width: '100%',
    },
    input: {
        padding: '10px',
        fontSize: '16px',
        border: '2px solid #e5e7eb',
        borderRadius: '8px',
        flex: 1,
    },
    divider: {
        height: '1px',
        backgroundColor: '#e5e7eb',
        width: '100%',
    },
    featureBuilder: {
        // padding: '16px',
        // backgroundColor: '#f9fafb',
        // borderRadius: '8px',
    },
    featureBuilderTitle: {
        margin: '0 0 12px 0',
        fontSize: '16px',
        color: '#374151',
    },
    featureInputRow: {
        display: 'flex',
        gap: '8px',
        marginBottom: '16px',
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
    emptyState: {
        color: '#9ca3af',
        fontStyle: 'italic',
        fontSize: '14px',
    },
    visualizerContainer: {
        maxWidth: '800px',
        margin: '0 auto',
    },
    footer: {
        textAlign: 'center',
        marginTop: '40px',
        color: '#9ca3af',
        fontSize: '14px',
    }

};

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <React.StrictMode>
        <DemoApp />
    </React.StrictMode>
);
