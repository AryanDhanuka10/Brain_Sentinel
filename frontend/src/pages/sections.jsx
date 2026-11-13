import React, { useState } from 'react';
import { Card, Chip, KV } from '../components.jsx';
import { OUTCOMES, fmt, fmtMs, humanize } from '../util.js';
import { imageUrl } from '../api.js';

const VERDICTS = {
  ANOMALY_DETECTED: 'Model verdict: ANOMALY DETECTED',
  NO_ANOMALY_DETECTED: 'Model verdict: NO ANOMALY DETECTED',
  INCONCLUSIVE: 'Model verdict: INCONCLUSIVE',
};

export function OutcomeBanner({ outcome, errors, verdict }) {
  const o = OUTCOMES[outcome?.status] || { label: humanize(outcome?.status), tone: 'neutral', text: '' };
  return (
    <div className={`banner banner-${o.tone}`} role="status">
      {verdict && VERDICTS[verdict.label] && (
        <div className="verdict" data-testid="verdict">
          {VERDICTS[verdict.label]}
          {verdict.reliability === 'low' && <span className="verdict-note"> (low reliability: close to the threshold or not conclusive)</span>}
        </div>
      )}
      <div className="banner-title">{o.label}</div>
      <div>{o.text}</div>
      {errors && errors.length > 0 && (
        <ul className="plain">
          {errors.map((e, i) => <li key={i}><code>{e.code}</code>: {e.message}</li>)}
        </ul>
      )}
    </div>
  );
}

const VIEWS = [
  ['original', 'Original'],
  ['heatmap', 'Heatmap'],
  ['blend', 'Overlay (blend)'],
  ['overlay', 'Server overlay'],
];

function Img({ id, name, label, onMissing }) {
  return <img src={imageUrl(id, name)} alt={label} onError={() => onMissing(name)} draggable="false" />;
}

function Frame({ id, view, opacity, boxes, showBoxes, onMissing, label }) {
  return (
    <div className="frame">
      {view === 'blend' ? (
        <>
          <Img id={id} name="original" label="" onMissing={onMissing} />
          <img className="layer" src={imageUrl(id, 'heatmap')} alt="" style={{ opacity }} onError={() => onMissing('heatmap')} draggable="false" />
          <span className="sr">{label}</span>
        </>
      ) : (
        <Img id={id} name={view} label={label} onMissing={onMissing} />
      )}
      {showBoxes && (view === 'blend' || view === 'overlay') && boxes.length > 0 && (
        <svg className="boxes" viewBox="0 0 240 240" aria-hidden="true" preserveAspectRatio="none">
          {boxes.map((r, i) => {
            const [y0, x0, y1, x1] = r.bbox_yxyx;
            return (
              <g key={i}>
                <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} className="box" />
                <text x={x0 + 2} y={Math.max(y0 - 3, 9)} className="box-label">{i + 1}</text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

export function ImagePanel({ report }) {
  const id = report.report_id;
  const [mode, setMode] = useState('side');
  const [single, setSingle] = useState('blend');
  const [opacity, setOpacity] = useState(0.55);
  const [showBoxes, setShowBoxes] = useState(true);
  const [missing, setMissing] = useState({});
  const onMissing = (n) => setMissing((m) => (m[n] ? m : { ...m, [n]: true }));
  const boxes = (report.regions || []).filter((r) => Array.isArray(r.bbox_yxyx) && r.bbox_yxyx.length === 4);
  const gone = missing.original || missing.heatmap;

  return (
    <Card title="Images" id="img">
      {gone ? (
        <div className="alert alert-warn" role="status">
          No images are stored for this report (images exist only when a model ran).
        </div>
      ) : null}
      {/* Hidden probe so a 404 is detected even before the panels render */}
      <div className={gone ? 'hidden' : ''}>
        <div className="toolbar">
          <div role="group" aria-label="Layout" className="seg">
            <button aria-pressed={mode === 'side'} onClick={() => setMode('side')}>Side by side</button>
            <button aria-pressed={mode === 'single'} onClick={() => setMode('single')}>Single view</button>
          </div>
          {mode === 'single' && (
            <label className="inline">View
              <select value={single} onChange={(e) => setSingle(e.target.value)}>
                {VIEWS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
              </select>
            </label>
          )}
          <label className="inline">
            Heatmap opacity
            <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-valuetext={`${Math.round(opacity * 100)} percent`} />
            <span className="small">{Math.round(opacity * 100)}%</span>
          </label>
          <label className="inline check">
            <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
            Region boxes
          </label>
        </div>
        {mode === 'side' ? (
          <div className="triple">
            {[['original', 'Original'], ['heatmap', 'Heatmap'], ['blend', 'Overlay']].map(([v, l]) => (
              <figure key={v}>
                <Frame id={id} view={v} opacity={opacity} boxes={boxes} showBoxes={showBoxes} onMissing={onMissing} label={`${l} view of the slice`} />
                <figcaption>{l}</figcaption>
              </figure>
            ))}
          </div>
        ) : (
          <figure className="single">
            <Frame id={id} view={single} opacity={opacity} boxes={boxes} showBoxes={showBoxes} onMissing={onMissing} label={`${single} view of the slice`} />
            <figcaption>{VIEWS.find((v) => v[0] === single)[1]}</figcaption>
          </figure>
        )}
        <p className="small">
          <strong>Display scale:</strong> {report.display_scale || '-'}{' '}
          <a href={imageUrl(id, 'heatmap')} download={`heatmap_${id}.png`}>Download heatmap</a>
        </p>
        <p className="muted small">Heatmap colours show relative model deviation, not tissue type or disease.</p>
      </div>
    </Card>
  );
}

export function ScoreCard({ report }) {
  const inf = report.inference;
  const m = report.model;
  return (
    <Card title="Anomaly score" id="score">
      {!inf ? <p className="muted">No model was run for this input.</p> : (
        <>
          <Gauge score={inf.score} threshold={inf.image_threshold} />
          <p className="muted small">Anomaly score, not a probability.</p>
          <KV items={[
            ['Score', fmt(inf.score, 4)],
            ['Image threshold', `${fmt(inf.image_threshold, 4)} (selected on ${inf.threshold_selected_on || '-'})`],
            ['Pixel threshold', fmt(inf.pixel_threshold, 4)],
            ['Latency', fmtMs(inf.latency_ms)],
            ['Attempts', inf.attempts],
            ['Model', m ? `${m.model_id} (${m.kind})` : '-'],
          ]} />
        </>
      )}
    </Card>
  );
}

function Gauge({ score, threshold }) {
  if (typeof score !== 'number' || typeof threshold !== 'number') return null;
  const max = Math.max(score, threshold) * 1.25 || 1;
  const sp = Math.min(100, (score / max) * 100);
  const tp = Math.min(100, (threshold / max) * 100);
  const over = score >= threshold;
  return (
    <div className="gauge" role="img" aria-label={`Anomaly score ${fmt(score, 3)} versus threshold ${fmt(threshold, 3)}; score is ${over ? 'at or above' : 'below'} the threshold`}>
      <div className="gauge-track">
        <div className={`gauge-fill ${over ? 'over' : 'under'}`} style={{ width: `${sp}%` }} />
        <div className="gauge-thr" style={{ left: `${tp}%` }} />
      </div>
      <div className="gauge-labels">
        <span>0</span>
        <span className="gauge-thr-label" style={{ left: `${tp}%` }}>threshold {fmt(threshold, 2)}</span>
        <span>{fmt(max, 1)}</span>
      </div>
    </div>
  );
}

export function EvidenceCard({ report }) {
  const ev = report.evidence;
  const st = report.stability;
  const loc = report.location;
  const regions = report.regions || [];
  return (
    <Card title="Evidence" id="ev">
      {!ev ? <p className="muted">No evidence available (no model was run).</p> : (
        <>
          <KV items={[
            ['Flagged for review', ev.flagged ? 'yes' : 'no'],
            ['Evidence strength (heuristic)', ev.strength],
            ['Margin ratio', fmt(ev.margin_ratio, 3)],
          ]} />
          {ev.reasons && ev.reasons.length > 0 && (
            <>
              <h3>Reasons</h3>
              <ul>{ev.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
            </>
          )}
        </>
      )}
      {st && (
        <>
          <h3>Stability under small perturbations</h3>
          <KV items={[
            ['Variants', st.n_variants],
            ['Mean map correlation', fmt(st.mean_map_corr, 4)],
            ['Min map correlation', fmt(st.min_map_corr, 4)],
            ['Score CV', fmt(st.score_cv, 4)],
            ['Max peak shift', typeof st.max_peak_shift_px === 'number' ? `${fmt(st.max_peak_shift_px, 1)} px` : '-'],
          ]} />
          {st.variants && st.variants.length > 0 && (
            <details>
              <summary>Per-variant results ({st.variants.length})</summary>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Variant</th><th>Score</th><th>Map corr.</th><th>Peak shift (px)</th></tr></thead>
                  <tbody>
                    {st.variants.map((v) => (
                      <tr key={v.name}><td>{v.name}</td><td>{fmt(v.score, 3)}</td><td>{fmt(v.map_corr, 4)}</td><td>{fmt(v.peak_shift_px, 1)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </>
      )}
      {loc && (
        <>
          <h3>Location sanity</h3>
          <KV items={[
            ['Top-scoring pixels', loc.top_pixels],
            ['Outside brain', `${fmt((loc.outside_brain_frac ?? NaN) * 100, 1)} %`],
            ['At image border', `${fmt((loc.border_frac ?? NaN) * 100, 1)} %`],
          ]} />
        </>
      )}
      <h3>Regions flagged for human review ({regions.length})</h3>
      {regions.length === 0 ? <p className="muted">None.</p> : (
        <div className="table-wrap">
          <table>
            <thead><tr><th>#</th><th>Area (px)</th><th>Bounding box (y0, x0, y1, x1)</th><th>Max score</th></tr></thead>
            <tbody>
              {regions.map((r, i) => (
                <tr key={i}><td>{i + 1}</td><td>{r.area_px}</td><td>{Array.isArray(r.bbox_yxyx) ? r.bbox_yxyx.join(', ') : '-'}</td><td>{fmt(r.max_score, 3)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export function QualityDomain({ quality, domain }) {
  return (
    <>
      {quality ? (
        <>
          <p>Quality level: <span className={`chip chip-${quality.level === 'pass' ? 'ok' : quality.level === 'warn' ? 'warn' : 'err'}`}>{quality.level}</span></p>
          {quality.flags && quality.flags.length > 0 ? (
            <ul className="plain">
              {quality.flags.map((f, i) => (
                <li key={i}><span className={`chip chip-${f.level === 'warn' ? 'warn' : 'err'}`}>{f.level}</span> <code>{f.code}</code> {f.detail}</li>
              ))}
            </ul>
          ) : <p className="muted">No quality flags.</p>}
          {quality.metrics && (
            <details>
              <summary>Quality metrics</summary>
              <KV items={Object.entries(quality.metrics).map(([k, v]) => [humanize(k), typeof v === 'number' ? fmt(v, 4) : String(v)])} />
            </details>
          )}
        </>
      ) : <p className="muted">No quality assessment available.</p>}
      <h3>Input domain</h3>
      {domain ? (
        <>
          <KV items={[
            ['Format / mode', `${domain.format || '-'} / ${domain.mode || '-'}`],
            ['Size', domain.width && domain.height ? `${domain.width} x ${domain.height}` : '-'],
            ['Palette conformant', domain.palette_conformant == null ? '-' : domain.palette_conformant ? 'yes' : 'no'],
            ['Pixels outside training palette', domain.pixels_outside_training_palette],
            ['Declared domain', domain.declared_domain],
          ]} />
          {domain.note && <p className="muted small">{domain.note}</p>}
        </>
      ) : <p className="muted">No domain information available.</p>}
    </>
  );
}

export function QualityCard({ report }) {
  return <Card title="Quality & domain" id="qd"><QualityDomain quality={report.quality} domain={report.domain} /></Card>;
}

const TRACE_ICON = { ok: '✓', pass: '✓', fail: '✕', retry: '↻', route: '→', skip: '–', warn: '!' };

export function TraceCard({ trace }) {
  return (
    <Card title="Agent execution trace" id="trace">
      {!trace || trace.length === 0 ? <p className="muted">No trace recorded.</p> : (
        <ol className="timeline">
          {trace.map((t, i) => (
            <li key={i} className={`tl tl-${t.status}`}>
              <span className="tl-dot" aria-hidden="true">{TRACE_ICON[t.status] || '?'}</span>
              <div className="tl-body">
                <div className="tl-head">
                  <strong>{t.node}</strong>
                  <span className={`tl-status tl-status-${t.status}`}>{t.status}</span>
                  <span className="muted small">{t.ms != null ? fmtMs(t.ms) : ''}</span>
                </div>
                {t.detail && <div className="small">{t.detail}</div>}
              </div>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}

export function ReviewReportCard({ report }) {
  return (
    <Card title="Review report" id="rr">
      <ul className="statements">
        {(report.statements || []).map((s, i) => (
          <li key={i} title={(s.refs || []).join(', ')}>
            {s.text}
            {s.refs && s.refs.length > 0 && (
              <details className="refs">
                <summary>Sources</summary>
                <span className="small">{s.refs.map((r) => <code key={r}>{r}</code>)}</span>
              </details>
            )}
          </li>
        ))}
      </ul>
      {report.llm_narrative && (
        <div className="narr">
          <div className="narr-label">Language-model text - not part of the evidence</div>
          <p>{typeof report.llm_narrative === 'string' ? report.llm_narrative : report.llm_narrative.text || JSON.stringify(report.llm_narrative)}</p>
        </div>
      )}
      <h3>Limitations</h3>
      <ul>{(report.limitations || []).map((l, i) => <li key={i}>{l}</li>)}</ul>
    </Card>
  );
}

export { Chip };
