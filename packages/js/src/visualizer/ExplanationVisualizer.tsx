import React from 'react';
// @ts-ignore
import { Explanation } from '../core/explanation';

interface ExplanationVisualizerProps {
    explanation: Explanation;
}

export const ExplanationVisualizer: React.FC<ExplanationVisualizerProps> = ({ explanation }) => {
    const topFeatures = explanation.getTopFeatures();

    return (
        <div className="explanation-visualizer">
            <h3>Explanation: {explanation.explainerName}</h3>
            <p>Target Class: {explanation.targetClass}</p>

            <div className="features-list">
                <h4>Top Features</h4>
                <ul>
                    {topFeatures.map(([feature, attribution]: [string, number]) => (
                        <li key={feature}>
                            {feature}: {attribution.toFixed(4)}
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};
