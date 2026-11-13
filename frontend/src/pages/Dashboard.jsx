import React, { useEffect, useState } from 'react';
import { getJobs, getModel } from '../api.js';
import { Card, Chip, ErrorBox, KV, Spinner } from '../components.jsx';
import { fmt, fmtDate, fmtMs, humanize } from '../util.js';

export default function Dashboard({ health, healthErr, onRetry }) {
  const [model, setModel] = useState(null);
  const [modelErr, setModelErr] = useState(null);
  const [jobs, setJobs] = useState(null);
  const [jobsErr, setJobsErr] = useState(null);

  useEffect(() => {
    if (health && health.model_loaded) getModel().then(setModel, setModelErr);
    else setModel(null);
  }, [health]);
  useEffect(() => {
    getJobs().then((j) => { setJobs(j); setJobsErr(null); }, setJobsErr);
  }, [health]);

  return (
    <>
      <h1>Dashboard</h1>
      <div className="grid2">
        <Card title="Backend health" id="health">
          {healthErr && (
            <>
              <ErrorBox error={healthErr} />
              <button className="btn" onClick={onRetry}>Retry</button>
            </>
          )}
          {!health && !healthErr && <Spinner label="Checking backend..." />}
          {health && (
            <KV items={[
              ['Status', health.status],
              ['Version', health.version],
              ['Model loaded', health.model_loaded ? 'yes' : 'no'],
              ['Model id', health.model_id],
              ['Device', health.device],
            ]} />
          )}
          {health && !health.model_loaded && (
            <div className="alert alert-warn" role="status">
              <strong>No trained model loaded - analysis unavailable.</strong> Upload analysis will be refused by the backend until a model artifact is available.
            </div>
          )}
        </Card>

        <Card title="Model" id="model">
          {!health && !healthErr && <Spinner label="Loading..." />}
          {healthErr && <p className="muted">Unavailable while the backend is offline.</p>}
          {health && !health.model_loaded && <p className="muted">No model information: no model is loaded.</p>}
          <ErrorBox error={modelErr} />
          {model && (
            <>
              <KV items={[
                ['Kind', model.kind],
                ['Trained on', model.trained_on],
                ['Checkpoint', model.checkpoint_mb != null ? `${fmt(model.checkpoint_mb, 1)} MB` : '-'],
                ['Image threshold', fmt(model.thresholds?.image_threshold, 4)],
                ['Pixel threshold', fmt(model.thresholds?.pixel_threshold, 4)],
                ['Thresholds selected on', model.thresholds?.selected_on],
              ]} />
              {model.validation_summary && (
                <>
                  <h3>Validation summary</h3>
                  <KV items={Object.entries(model.validation_summary).map(([k, v]) => [humanize(k), typeof v === 'number' ? fmt(v, 4) : v])} />
                </>
              )}
              {model.hparams && (
                <details>
                  <summary>Hyper-parameters</summary>
                  <KV items={Object.entries(model.hparams).map(([k, v]) => [k, typeof v === 'object' ? JSON.stringify(v) : String(v)])} />
                </details>
              )}
            </>
          )}
        </Card>
      </div>

      <Card title="Recent jobs" id="jobs" actions={<a className="btn btn-small" href="#/upload">New analysis</a>}>
        <ErrorBox error={jobsErr} />
        {!jobs && !jobsErr && <Spinner label="Loading jobs..." />}
        {jobs && jobs.length === 0 && <p className="muted">No jobs yet.</p>}
        {jobs && jobs.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Created</th><th>File</th><th>Status</th><th>Latency</th><th><span className="sr">Open</span></th></tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.job_id}>
                    <td>{fmtDate(j.created)}</td>
                    <td className="break">{j.filename}</td>
                    <td><Chip status={j.status} />{j.error_code ? <span className="muted small"> {j.error_code}</span> : null}</td>
                    <td>{fmtMs(j.latency_ms)}</td>
                    <td>
                      {j.report_id ? (
                        <a className="btn btn-small" href={`#/report/${j.report_id}`} aria-label={`Open report for ${j.filename}`}>Open report</a>
                      ) : <span className="muted">no report</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}
