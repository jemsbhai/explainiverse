import React from 'react';
// @ts-ignore
import { Explanation } from '../core/explanation';

interface ExplanationVisualizerProps {
    explanation: Explanation;
}

export const ExplanationVisualizer: React.FC<ExplanationVisualizerProps> = ({ explanation }) => {
    const topFeatures = explanation.getTopFeatures();

    // Find max absolute value for scaling bars
    const maxValue = Math.max(...topFeatures.map(([_, val]) => Math.abs(val)));

    return (
        <div style={styles.container}>
            <div style={styles.header}>
                <h2 style={styles.title}>Explanation Results</h2>
                <div style={styles.badge}>{explanation.explainerName}</div>
            </div>

            <div style={styles.infoCard}>
                <span style={styles.label}>Target Class:</span>
                <span style={styles.value}>{explanation.targetClass}</span>
            </div>

            <div style={styles.featuresSection}>
                <h3 style={styles.sectionTitle}>Feature Importance</h3>
                <div style={styles.featuresList}>
                    {topFeatures.map(([feature, attribution]: [string, number]) => {
                        const percentage = (Math.abs(attribution) / maxValue) * 100;
                        const isPositive = attribution > 0;

                        return (
                            <div key={feature} style={styles.featureItem}>
                                <div style={styles.featureHeader}>
                                    <span style={styles.featureName}>{feature}</span>
                                    <span style={{
                                        ...styles.featureValue,
                                        color: isPositive ? '#10b981' : '#ef4444'
                                    }}>
                                        {attribution > 0 ? '+' : ''}{attribution.toFixed(4)}
                                    </span>
                                </div>
                                <div style={styles.barContainer}>
                                    <div
                                        style={{
                                            ...styles.bar,
                                            width: `${percentage}%`,
                                            backgroundColor: isPositive ? '#10b981' : '#ef4444',
                                        }}
                                    />
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

const styles: Record<string, React.CSSProperties> = {
    container: {
        maxWidth: '800px',
        margin: '0 auto',
        padding: '24px',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        backgroundColor: '#ffffff',
        borderRadius: '12px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    },
    header: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '24px',
        paddingBottom: '16px',
        borderBottom: '2px solid #e5e7eb',
    },
    title: {
        margin: 0,
        fontSize: '28px',
        fontWeight: '700',
        color: '#111827',
    },
    badge: {
        padding: '6px 16px',
        backgroundColor: '#3b82f6',
        color: 'white',
        borderRadius: '20px',
        fontSize: '14px',
        fontWeight: '600',
    },
    infoCard: {
        padding: '16px',
        backgroundColor: '#f9fafb',
        borderRadius: '8px',
        marginBottom: '24px',
        display: 'flex',
        gap: '12px',
        alignItems: 'center',
    },
    label: {
        fontWeight: '600',
        color: '#6b7280',
        fontSize: '14px',
    },
    value: {
        fontWeight: '700',
        color: '#111827',
        fontSize: '18px',
    },
    featuresSection: {
        marginTop: '24px',
    },
    sectionTitle: {
        fontSize: '20px',
        fontWeight: '600',
        color: '#111827',
        marginBottom: '16px',
    },
    featuresList: {
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
    },
    featureItem: {
        padding: '12px',
        backgroundColor: '#f9fafb',
        borderRadius: '8px',
        border: '1px solid #e5e7eb',
    },
    featureHeader: {
        display: 'flex',
        justifyContent: 'space-between',
        marginBottom: '8px',
    },
    featureName: {
        fontWeight: '600',
        color: '#374151',
        fontSize: '15px',
    },
    featureValue: {
        fontWeight: '700',
        fontSize: '15px',
        fontFamily: 'monospace',
    },
    barContainer: {
        width: '100%',
        height: '8px',
        backgroundColor: '#e5e7eb',
        borderRadius: '4px',
        overflow: 'hidden',
    },
    bar: {
        height: '100%',
        borderRadius: '4px',
        transition: 'width 0.3s ease',
    },
};
