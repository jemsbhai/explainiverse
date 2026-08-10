import { useId, type CSSProperties, type FC } from 'react';

import type { Explanation } from '../core/explanation';

export interface ExplanationVisualizerProps {
  readonly explanation: Explanation;
}

function formatAttribution(value: number): string {
  const normalized = Object.is(value, -0) ? 0 : value;
  return `${normalized > 0 ? '+' : ''}${normalized.toFixed(4)}`;
}

/** Display-only view of caller-supplied, validated feature attributions. */
export const ExplanationVisualizer: FC<ExplanationVisualizerProps> = ({ explanation }) => {
  const titleId = useId();
  const featuresTitleId = useId();
  const topFeatures = explanation.getTopFeatures();
  const maxMagnitude = topFeatures.reduce(
    (maximum, [, value]) => Math.max(maximum, Math.abs(value)),
    0,
  );

  return (
    <section aria-labelledby={titleId} style={styles.container}>
      <header style={styles.header}>
        <h2 id={titleId} style={styles.title}>
          Experimental Attribution View
        </h2>
        <span
          aria-label={`Explainer label: ${explanation.explainerName}`}
          style={styles.badge}
        >
          {explanation.explainerName}
        </span>
      </header>

      <p style={styles.disclosure}>
        Display only: values are supplied by the caller; this component does not run or
        validate an explanation method.
      </p>

      <dl style={styles.infoCard}>
        <dt style={styles.label}>Target class</dt>
        <dd style={styles.value}>{explanation.targetClass}</dd>
      </dl>

      <section aria-labelledby={featuresTitleId} style={styles.featuresSection}>
        <h3 id={featuresTitleId} style={styles.sectionTitle}>
          Top Feature Attributions
        </h3>
        <p style={styles.rankingNote}>Ranked by absolute magnitude; attribution sign is preserved.</p>

        {topFeatures.length === 0 ? (
          <p role="status" style={styles.emptyState}>
            No feature attributions were supplied.
          </p>
        ) : (
          <ul aria-label="Top feature attributions" style={styles.featuresList}>
            {topFeatures.map(([feature, attribution]) => {
              const magnitude = Math.abs(attribution);
              const percentage = maxMagnitude === 0 ? 0 : (magnitude / maxMagnitude) * 100;
              const direction =
                attribution > 0 ? 'positive' : attribution < 0 ? 'negative' : 'zero';
              const color =
                direction === 'positive'
                  ? '#047857'
                  : direction === 'negative'
                    ? '#b91c1c'
                    : '#4b5563';
              const formatted = formatAttribution(attribution);

              return (
                <li key={feature} style={styles.featureItem}>
                  <div style={styles.featureHeader}>
                    <span style={styles.featureName}>{feature}</span>
                    <span
                      aria-label={`${feature}: ${formatted} (${direction})`}
                      style={{ ...styles.featureValue, color }}
                    >
                      {formatted}
                    </span>
                  </div>
                  <div
                    aria-label={`${feature} attribution magnitude`}
                    aria-valuemax={maxMagnitude > 0 ? maxMagnitude : 1}
                    aria-valuemin={0}
                    aria-valuenow={magnitude}
                    aria-valuetext={`${formatted} (${direction})`}
                    role="meter"
                    style={styles.barContainer}
                  >
                    <div
                      aria-hidden="true"
                      style={{
                        ...styles.bar,
                        width: `${percentage}%`,
                        backgroundColor: color,
                      }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </section>
  );
};

const styles: Record<string, CSSProperties> = {
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
    gap: '16px',
    marginBottom: '16px',
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
    backgroundColor: '#1d4ed8',
    color: '#ffffff',
    borderRadius: '20px',
    fontSize: '14px',
    fontWeight: '600',
  },
  disclosure: {
    margin: '0 0 16px',
    color: '#374151',
    lineHeight: 1.5,
  },
  infoCard: {
    padding: '16px',
    backgroundColor: '#f9fafb',
    borderRadius: '8px',
    margin: '0 0 24px',
    display: 'flex',
    gap: '12px',
    alignItems: 'baseline',
  },
  label: {
    fontWeight: '600',
    color: '#4b5563',
    fontSize: '14px',
  },
  value: {
    margin: 0,
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
    margin: '0 0 4px',
  },
  rankingNote: {
    color: '#4b5563',
    fontSize: '14px',
    margin: '0 0 16px',
  },
  featuresList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    listStyle: 'none',
    margin: 0,
    padding: 0,
  },
  featureItem: {
    padding: '12px',
    backgroundColor: '#f9fafb',
    borderRadius: '8px',
    border: '1px solid #d1d5db',
  },
  featureHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '12px',
    marginBottom: '8px',
  },
  featureName: {
    overflowWrap: 'anywhere',
    fontWeight: '600',
    color: '#374151',
    fontSize: '15px',
  },
  featureValue: {
    flexShrink: 0,
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
  },
  emptyState: {
    padding: '12px',
    backgroundColor: '#f9fafb',
    color: '#4b5563',
    fontStyle: 'italic',
    margin: 0,
  },
};
