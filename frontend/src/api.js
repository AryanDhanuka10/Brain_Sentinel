// Thin API client. All data shown in the UI comes from these calls.
export class ApiError extends Error {
  constructor({ status = 0, code = 'error', message = 'Request failed', jobId, reportId, offline = false }) {
    super(message);
    this.status = status;
    this.code = code;
    this.jobId = jobId;
    this.reportId = reportId;
    this.offline = offline;
  }
}

/** Convert a FastAPI error body (dict detail or validation array) into ApiError fields. */
export function parseErrorBody(status, body) {
  const detail = body && body.detail;
  if (Array.isArray(detail)) {
    const msg = detail
      .map((d) => `${(d.loc || []).filter((x) => x !== 'body').join('.') || 'request'}: ${d.msg}`)
      .join('; ');
    return { status, code: 'validation_error', message: msg || 'Request validation failed' };
  }
  if (detail && typeof detail === 'object') {
    return {
      status,
      code: detail.code || 'http_error',
      message: detail.message || `HTTP ${status}`,
      jobId: detail.job_id,
      reportId: detail.report_id,
    };
  }
  if (typeof detail === 'string') return { status, code: 'http_error', message: detail };
  return { status, code: 'http_error', message: `HTTP ${status}` };
}

async function request(url, opts) {
  let res;
  try {
    res = await fetch(url, opts);
  } catch (e) {
    throw new ApiError({ code: 'backend_offline', message: 'Cannot reach the BrainSentinel backend.', offline: true });
  }
  let body = null;
  const text = await res.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }
  if (!res.ok) {
    // Vite proxy returns 502/504 with empty body when upstream is down.
    if ((res.status === 502 || res.status === 503 || res.status === 504) && !(body && body.detail)) {
      throw new ApiError({ status: res.status, code: 'backend_offline', message: 'Cannot reach the BrainSentinel backend.', offline: true });
    }
    throw new ApiError(parseErrorBody(res.status, body));
  }
  return body;
}

export const getHealth = () => request('/health');
export const getModel = () => request('/api/v1/model');
export const getJobs = () => request('/api/v1/jobs');
export const getReport = (id) => request(`/api/v1/reports/${encodeURIComponent(id)}`);
export const imageUrl = (id, name) => `/api/v1/reports/${encodeURIComponent(id)}/images/${name}`;

function upload(url, file) {
  const fd = new FormData();
  fd.append('file', file);
  return request(url, { method: 'POST', body: fd });
}
export const analyze = (file) => upload('/api/v1/mri/analyze', file);
export const qualityCheck = (file) => upload('/api/v1/mri/quality-check', file);
export const submitReview = (id, decision, note) =>
  request(`/api/v1/reports/${encodeURIComponent(id)}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision, note: note || null }),
  });
