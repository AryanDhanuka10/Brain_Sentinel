import React, { useRef, useState } from 'react';
import { analyze, qualityCheck } from '../api.js';
import { Card, ErrorBox } from '../components.jsx';
import { checkFile, fmt, humanize } from '../util.js';
import { QualityDomain } from './sections.jsx';

const STAGES = ['uploading', 'analysing', 'done'];

export default function Upload({ health, healthErr }) {
  const input = useRef(null);
  const [file, setFile] = useState(null);
  const [clientErr, setClientErr] = useState(null);
  const [drag, setDrag] = useState(false);
  const [stage, setStage] = useState('idle'); // idle|uploading|analysing|done|error
  const [error, setError] = useState(null);
  const [qc, setQc] = useState(null);
  const [qcBusy, setQcBusy] = useState(false);

  const pick = (f) => {
    setQc(null); setError(null); setStage('idle');
    setFile(f || null);
    setClientErr(f ? checkFile(f) : null);
  };

  const busy = stage === 'uploading' || stage === 'analysing';
  const modelDown = health && !health.model_loaded;

  const run = async () => {
    const ce = checkFile(file);
    if (ce) { setClientErr(ce); return; }
    setError(null); setQc(null); setStage('uploading');
    // The request is a single round trip; move to the "analysing" label once the body has been sent.
    const t = setTimeout(() => setStage((s) => (s === 'uploading' ? 'analysing' : s)), 250);
    try {
      const r = await analyze(file);
      clearTimeout(t);
      setStage('done');
      window.location.hash = `#/report/${r.report_id}`;
    } catch (e) {
      clearTimeout(t);
      setError(e); setStage('error');
    }
  };

  const quality = async () => {
    const ce = checkFile(file);
    if (ce) { setClientErr(ce); return; }
    setError(null); setQc(null); setQcBusy(true);
    try { setQc(await qualityCheck(file)); } catch (e) { setError(e); }
    setQcBusy(false);
  };

  const statusText = {
    idle: '', uploading: 'Uploading file...', analysing: 'Analysing slice...', done: 'Done. Opening report...', error: 'Analysis did not complete.',
  }[stage];

  return (
    <>
      <h1>Upload &amp; analyse</h1>
      {healthErr && <div className="alert alert-err" role="alert"><strong>Backend offline.</strong> Uploads cannot be processed until it is reachable.</div>}
      {modelDown && <div className="alert alert-warn" role="status"><strong>No trained model loaded - analysis unavailable.</strong> You can still run a quality check.</div>}
      <Card title="Choose a slice" id="up">
        <div
          className={`drop ${drag ? 'drag' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
        >
          <p>Drag a <strong>.png</strong> slice here (max 2 MB), or</p>
          <label className="btn" htmlFor="file">Choose file</label>
          <input
            id="file" ref={input} type="file" accept=".png,image/png" className="file-input"
            onChange={(e) => pick(e.target.files[0])}
          />
          <p className="small muted" aria-live="polite">
            {file ? `Selected: ${file.name} (${(file.size / 1024).toFixed(0)} KB)` : 'No file selected.'}
          </p>
        </div>
        {clientErr && <div className="alert alert-err" role="alert">{clientErr}</div>}
        <div className="row">
          <button className="btn btn-primary" onClick={run} disabled={!file || busy || qcBusy}>Analyse</button>
          <button className="btn" onClick={quality} disabled={!file || busy || qcBusy}>Quality check only</button>
        </div>
        <ol className="stages" aria-label="Progress">
          {STAGES.map((s) => {
            const idx = STAGES.indexOf(s);
            const cur = STAGES.indexOf(stage);
            const state = stage === 'error' ? (s === 'done' ? 'fail' : 'idle') : cur === idx ? 'current' : cur > idx ? 'done' : 'idle';
            return <li key={s} className={`stage stage-${state}`}>{s === 'done' && stage === 'error' ? 'error' : s}</li>;
          })}
        </ol>
        <p className="status-line" role="status" aria-live="polite">{statusText}{qcBusy ? 'Running quality check...' : ''}</p>
        <ErrorBox error={error} onViewReport={(id) => { window.location.hash = `#/report/${id}`; }} />
      </Card>

      {qc && (
        <Card title={`Quality check: ${qc.filename}`} id="qc">
          <p className="muted small">No model was run. This only checks input format and image quality.</p>
          <QualityDomain quality={qc.quality} domain={qc.domain} />
        </Card>
      )}
    </>
  );
}
