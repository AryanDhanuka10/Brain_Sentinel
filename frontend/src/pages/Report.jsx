import React, { useEffect, useState } from 'react';
import { getReport, submitReview } from '../api.js';
import { Card, ErrorBox, KV, Spinner } from '../components.jsx';
import { fmtDate } from '../util.js';
import { EvidenceCard, ImagePanel, OutcomeBanner, QualityCard, ReviewReportCard, ScoreCard, TraceCard } from './sections.jsx';

function HumanReview({ report, onUpdated }) {
  const hr = report.human_review;
  const [note, setNote] = useState(hr.note || '');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  if (!hr || hr.status === 'not_applicable') return null;

  const send = async (decision) => {
    setBusy(true); setErr(null);
    try { onUpdated(await submitReview(report.report_id, decision, note.trim())); } catch (e) { setErr(e); }
    setBusy(false);
  };
  const options = hr.options || ['accept', 'reject', 'annotate'];
  const labels = { accept: 'Accept', reject: 'Reject', annotate: 'Annotate' };
  return (
    <Card title="Human review" id="hr">
      <KV items={[
        ['Status', hr.status],
        ['Decided at', hr.decided_at ? fmtDate(hr.decided_at) : '-'],
        ['Note', hr.note],
      ]} />
      <label htmlFor="note">Note (optional, max 2000 characters)</label>
      <textarea id="note" rows={3} maxLength={2000} value={note} onChange={(e) => setNote(e.target.value)} />
      <div className="small muted">{note.length} / 2000</div>
      <div className="row">
        {options.map((o) => (
          <button key={o} className={`btn ${o === 'accept' ? 'btn-primary' : ''}`} disabled={busy || (o === 'annotate' && !note.trim())} onClick={() => send(o)}>
            {labels[o] || o}
          </button>
        ))}
      </div>
      <p className="small muted">Accept confirms the flag was reviewed; reject dismisses it. These are review decisions about the automated output, not clinical findings.</p>
      <div role="status" aria-live="polite">{busy ? 'Saving...' : ''}</div>
      <ErrorBox error={err} />
    </Card>
  );
}

export default function Report({ id }) {
  const [report, setReport] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    let live = true;
    getReport(id).then((r) => live && setReport(r), (e) => live && setErr(e));
    return () => { live = false; };
  }, [id]);

  if (err) return (<><h1>Report</h1><ErrorBox error={err} /><a className="btn" href="#/">Back to dashboard</a></>);
  if (!report) return (<><h1>Report</h1><Spinner label="Loading report..." /></>);

  return (
    <>
      <p className="crumbs"><a href="#/">Dashboard</a> / Report {report.report_id}</p>
      <h1>Report</h1>
      <p className="muted small break">
        {report.input?.filename} - created {fmtDate(report.created)} - job {report.job_id || '-'} - sha256 {report.input?.sha256?.slice(0, 12) || '-'}
      </p>
      <OutcomeBanner outcome={report.outcome} errors={report.errors} verdict={report.verdict} />
      <ImagePanel report={report} />
      <div className="grid2">
        <ScoreCard report={report} />
        <QualityCard report={report} />
      </div>
      <EvidenceCard report={report} />
      <TraceCard trace={report.trace} />
      <ReviewReportCard report={report} />
      <HumanReview report={report} onUpdated={setReport} />
    </>
  );
}
