import React, { useCallback, useEffect, useState } from 'react';
import { getHealth } from './api.js';
import Dashboard from './pages/Dashboard.jsx';
import Upload from './pages/Upload.jsx';
import Report from './pages/Report.jsx';

function parseHash() {
  const h = window.location.hash.replace(/^#\/?/, '');
  const [page, id] = h.split('/');
  if (page === 'upload') return { page: 'upload' };
  if (page === 'report' && id) return { page: 'report', id: decodeURIComponent(id) };
  return { page: 'dashboard' };
}

export const go = (path) => {
  window.location.hash = path;
};

export default function App() {
  const [route, setRoute] = useState(parseHash());
  const [health, setHealth] = useState(null);
  const [healthErr, setHealthErr] = useState(null);

  const loadHealth = useCallback(() => {
    getHealth().then(
      (h) => { setHealth(h); setHealthErr(null); },
      (e) => { setHealth(null); setHealthErr(e); },
    );
  }, []);

  useEffect(() => {
    const on = () => { setRoute(parseHash()); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  useEffect(loadHealth, [loadHealth]);

  const nav = (key, label, href) => (
    <a href={href} className={route.page === key ? 'active' : ''} aria-current={route.page === key ? 'page' : undefined}>
      {label}
    </a>
  );

  return (
    <>
      <a className="skip" href="#main">Skip to content</a>
      <div className="disclaimer" role="note" aria-label="Disclaimer">
        <strong>Research prototype - not a diagnostic tool.</strong>{' '}
        {health ? health.disclaimer : healthErr ? 'Backend offline: no analysis output is available. Nothing shown here is a clinical result.' : 'Loading...'}
      </div>
      <header className="topbar">
        <div className="brand"><span className="logo" aria-hidden="true" /> BrainSentinel</div>
        <nav aria-label="Main">
          {nav('dashboard', 'Dashboard', '#/')}
          {nav('upload', 'Upload & analyse', '#/upload')}
        </nav>
        <span className={`pill ${health ? (health.model_loaded ? 'pill-ok' : 'pill-warn') : 'pill-err'}`}>
          {health ? (health.model_loaded ? `Model: ${health.model_id}` : 'No model') : healthErr ? 'Backend offline' : '...'}
        </span>
      </header>
      <main id="main" tabIndex={-1}>
        {route.page === 'dashboard' && <Dashboard health={health} healthErr={healthErr} onRetry={loadHealth} />}
        {route.page === 'upload' && <Upload health={health} healthErr={healthErr} />}
        {route.page === 'report' && <Report key={route.id} id={route.id} />}
      </main>
    </>
  );
}
