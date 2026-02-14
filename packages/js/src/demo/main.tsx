import React from 'react';
import ReactDOM from 'react-dom/client';
// @ts-ignore
import { ExplanationVisualizer } from '../visualizer';
// @ts-ignore
import { Explanation } from '../core/explanation';

// Create sample data for the demo
const explanation = new Explanation(
    'LIME',
    'Chihuahua',
    {
        feature_attributions: {
            'nose': 0.45,
            'ear': 0.32,
            'tail': -0.12,
            'fur': 0.05,
            'legs': 0.01
        }
    },
    ['nose', 'ear', 'tail', 'fur', 'legs']
);

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <React.StrictMode>
        <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
            <h1>Explainiverse Visualizer Demo</h1>
            <ExplanationVisualizer explanation={explanation} />
        </div>
    </React.StrictMode>
);
