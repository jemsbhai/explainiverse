import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
// @ts-ignore
import { ExplanationVisualizer } from '../../src/visualizer/ExplanationVisualizer';
// @ts-ignore
import { Explanation } from '../../src/core/explanation';

describe('ExplanationVisualizer', () => {
    it('renders explanation details', () => {
        const data = { feature_attributions: { featureA: 0.8, featureB: -0.2 } };
        const explanation = new Explanation('TestExplainer', 'cat', data, ['featureA', 'featureB']);

        render(<ExplanationVisualizer explanation={explanation} />);

        expect(screen.getByText(/TestExplainer/i)).toBeInTheDocument();
        expect(screen.getByText(/cat/i)).toBeInTheDocument();
        expect(screen.getByText(/featureA/i)).toBeInTheDocument();
    });
});
