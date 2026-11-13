export const OUTCOMES = {
  no_region_flagged: { label: 'No region flagged', tone: 'ok', text: 'The anomaly score is below the validation threshold. The model misses some abnormal slices, so this is not a guarantee that the slice is normal.' },
  flagged_for_review: { label: 'Region flagged for human review', tone: 'flag', text: 'The anomaly score is at or above the validation threshold. The model also flags some normal slices, so a qualified human should confirm the highlighted region.' },
  uncertain_escalated: { label: 'Uncertain - escalated to human review', tone: 'warn', text: 'The automated checks were not conclusive. This slice needs review by a qualified human.' },
  quality_failed: { label: 'Image quality check failed', tone: 'warn', text: 'The slice did not pass the quality gate, so no result is shown.' },
  rejected_input: { label: 'Input rejected', tone: 'err', text: 'The uploaded file could not be accepted. No model was run.' },
  domain_mismatch: { label: 'Input does not match the training domain', tone: 'warn', text: 'The image format differs from what the model was trained on, so it was not analysed.' },
  no_model_available: { label: 'No model available', tone: 'err', text: 'No trained model was loaded, so analysis was not run.' },
  inference_failed: { label: 'Analysis failed', tone: 'err', text: 'The model could not complete the analysis.' },
};

export const fmt = (v, d = 3) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '-');
export const fmtMs = (v) => (typeof v === 'number' ? `${v.toFixed(1)} ms` : '-');
export const humanize = (s) => (s ? String(s).replace(/_/g, ' ') : '-');

export const MAX_BYTES = 2 * 1024 * 1024;

export function checkFile(file) {
  if (!file) return 'Choose a file first.';
  if (!file.name.toLowerCase().endsWith('.png')) return 'Only .png files are accepted (client-side check).';
  if (file.size > MAX_BYTES) return `File is ${(file.size / 1048576).toFixed(2)} MB; the limit is 2 MB (client-side check).`;
  return null;
}

export const fmtDate = (s) => {
  if (!s) return '-';
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? s : d.toLocaleString();
};
