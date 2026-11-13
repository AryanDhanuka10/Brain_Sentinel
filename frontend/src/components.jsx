import React from 'react';
import { OUTCOMES, humanize } from './util.js';

export function Card({ title, children, actions, id }) {
  const hid = id ? `${id}-h` : undefined;
  return (
    <section className="card" aria-labelledby={hid}>
      <div className="card-head">
        <h2 id={hid}>{title}</h2>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Chip({ status }) {
  const o = OUTCOMES[status];
  const tone = o ? o.tone : status === 'running' ? 'info' : status === 'error' ? 'err' : 'neutral';
  return <span className={`chip chip-${tone}`}>{o ? o.label : humanize(status)}</span>;
}

export function ErrorBox({ error, onViewReport }) {
  if (!error) return null;
  return (
    <div className="alert alert-err" role="alert">
      <strong>{error.offline ? 'Backend offline' : `Error ${error.status || ''}`.trim()}</strong>
      <div>
        <code>{error.code}</code>: {error.message}
      </div>
      {error.reportId && onViewReport && (
        <button className="btn btn-small" onClick={() => onViewReport(error.reportId)}>
          View report
        </button>
      )}
    </div>
  );
}

export function Spinner({ label }) {
  return (
    <p className="muted" role="status">
      <span className="spinner" aria-hidden="true" /> {label}
    </p>
  );
}

export function KV({ items }) {
  return (
    <dl className="kv">
      {items.map(([k, v]) => (
        <React.Fragment key={k}>
          <dt>{k}</dt>
          <dd>{v ?? '-'}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}
