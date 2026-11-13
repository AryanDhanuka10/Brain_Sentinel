import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { parseErrorBody } from './api.js';
import { checkFile, OUTCOMES } from './util.js';
import { ErrorBox } from './components.jsx';
import { TraceCard, OutcomeBanner } from './pages/sections.jsx';

describe('parseErrorBody', () => {
  it('handles dict detail with report_id', () => {
    const e = parseErrorBody(422, { detail: { code: 'wrong_size', message: 'bad', report_id: 'r1', job_id: 'j1' } });
    expect(e).toMatchObject({ status: 422, code: 'wrong_size', message: 'bad', reportId: 'r1', jobId: 'j1' });
  });
  it('handles FastAPI validation array', () => {
    const e = parseErrorBody(422, { detail: [{ loc: ['body', 'file'], msg: 'Field required' }] });
    expect(e.code).toBe('validation_error');
    expect(e.message).toContain('file: Field required');
  });
  it('handles string / missing detail', () => {
    expect(parseErrorBody(500, { detail: 'boom' }).message).toBe('boom');
    expect(parseErrorBody(500, null).message).toBe('HTTP 500');
  });
});

describe('checkFile', () => {
  it('rejects non-png and oversize', () => {
    expect(checkFile(new File(['x'], 'a.txt'))).toMatch(/png/);
    const big = new File([new Uint8Array(2 * 1024 * 1024 + 1)], 'a.png');
    expect(checkFile(big)).toMatch(/2 MB/);
    expect(checkFile(new File(['x'], 'A.PNG'))).toBeNull();
  });
});

describe('components', () => {
  it('ErrorBox shows code, message and View report button', () => {
    render(<ErrorBox error={{ status: 422, code: 'corrupt_image', message: 'cannot decode', reportId: 'abc' }} onViewReport={() => {}} />);
    expect(screen.getByRole('alert')).toHaveTextContent('corrupt_image');
    expect(screen.getByRole('button', { name: 'View report' })).toBeInTheDocument();
  });
  it('TraceCard renders status text alongside icon', () => {
    render(<TraceCard trace={[{ node: 'run_detector', status: 'fail', detail: 'x', ms: 1.5 }]} />);
    expect(screen.getByText('fail')).toBeInTheDocument();
  });
  it('outcome wording is never diagnostic', () => {
    const all = JSON.stringify(OUTCOMES).toLowerCase();
    for (const w of ['tumor', 'tumour', 'diagnos', 'cancer', 'probability of']) expect(all).not.toContain(w);
    render(<OutcomeBanner outcome={{ status: 'flagged_for_review' }} />);
    expect(screen.getByText(/flagged for human review/i)).toBeInTheDocument();
  });
  it('shows a plain model verdict, with a low-reliability note', () => {
    render(<OutcomeBanner outcome={{ status: 'flagged_for_review' }} verdict={{ label: 'ANOMALY_DETECTED', reliability: 'low' }} />);
    expect(screen.getByTestId('verdict').textContent).toMatch(/ANOMALY DETECTED/);
    expect(screen.getByTestId('verdict').textContent).toMatch(/low reliability/i);
  });
});
