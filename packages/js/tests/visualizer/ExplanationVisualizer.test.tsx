import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { Explanation } from '../../src/core/explanation';
import { ExplanationVisualizer } from '../../src/visualizer/ExplanationVisualizer';

describe('ExplanationVisualizer', () => {
  it('renders a labelled display-only region with signed attribution semantics', () => {
    const explanation = new Explanation('TestExplainer', 'cat', {
      feature_attributions: { featureA: 0.8, featureB: -0.2 },
    });
    render(<ExplanationVisualizer explanation={explanation} />);

    const region = screen.getByRole('region', {
      name: 'Experimental Attribution View',
    });
    expect(within(region).getByLabelText('Explainer label: TestExplainer')).toBeVisible();
    expect(within(region).getByText('cat')).toBeVisible();
    expect(within(region).getByText(/does not run or validate an explanation method/i)).toBeVisible();
    expect(within(region).getByLabelText('featureA: +0.8000 (positive)')).toBeVisible();
    expect(within(region).getByLabelText('featureB: -0.2000 (negative)')).toBeVisible();

    const positiveMeter = within(region).getByRole('meter', {
      name: 'featureA attribution magnitude',
    });
    const negativeMeter = within(region).getByRole('meter', {
      name: 'featureB attribution magnitude',
    });
    expect(positiveMeter).toHaveAttribute('aria-valuetext', '+0.8000 (positive)');
    expect(negativeMeter).toHaveAttribute('aria-valuetext', '-0.2000 (negative)');
    expect(positiveMeter.firstElementChild).toHaveStyle({ width: '100%' });
    expect(negativeMeter.firstElementChild).toHaveStyle({ width: '25%' });
  });

  it('renders an explicit empty state without invalid scale values', () => {
    render(<ExplanationVisualizer explanation={new Explanation('Test', 'cat', {})} />);
    expect(screen.getByRole('status')).toHaveTextContent('No feature attributions');
    expect(screen.queryAllByRole('meter')).toHaveLength(0);
  });

  it('renders all-zero and negative-zero maps with valid zero-width meters', () => {
    const explanation = new Explanation('Test', 'cat', {
      feature_attributions: { zero: 0, negativeZero: -0 },
    });
    render(<ExplanationVisualizer explanation={explanation} />);

    const meters = screen.getAllByRole('meter');
    expect(meters).toHaveLength(2);
    for (const meter of meters) {
      expect(meter).toHaveAttribute('aria-valuemax', '1');
      expect(meter).toHaveAttribute('aria-valuenow', '0');
      expect(meter.firstElementChild).toHaveStyle({ width: '0%' });
    }
    expect(screen.getByLabelText('zero: 0.0000 (zero)')).toBeVisible();
    expect(screen.getByLabelText('negativeZero: 0.0000 (zero)')).toBeVisible();
  });

  it('shows at most five features and describes magnitude ranking', () => {
    const explanation = new Explanation('Test', 'cat', {
      feature_attributions: { a: 1, b: 2, c: 3, d: 4, e: 5, f: 6 },
    });
    render(<ExplanationVisualizer explanation={explanation} />);

    expect(screen.getByText(/ranked by absolute magnitude/i)).toBeVisible();
    expect(screen.getAllByRole('meter')).toHaveLength(5);
    expect(screen.queryByText('a')).not.toBeInTheDocument();
  });
});
