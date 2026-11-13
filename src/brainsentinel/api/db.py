"""SQLite persistence (jobs, reports, human reviews). One short-lived connection per call."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, created TEXT, filename TEXT, sha256 TEXT, status TEXT,
                                report_id TEXT, error_code TEXT, latency_ms REAL);
CREATE TABLE IF NOT EXISTS reports(report_id TEXT PRIMARY KEY, job_id TEXT, created TEXT, report_json TEXT);
CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY AUTOINCREMENT, report_id TEXT, decision TEXT, note TEXT, decided_at TEXT);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _c(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def new_job(self, job_id, filename):
        with self._c() as c:
            c.execute("INSERT INTO jobs(job_id,created,filename,status) VALUES(?,?,?,?)",
                      (job_id, time.strftime("%Y-%m-%dT%H:%M:%S"), filename, "running"))

    def finish_job(self, job_id, status, report_id=None, sha256=None, error_code=None, latency_ms=None):
        with self._c() as c:
            c.execute("UPDATE jobs SET status=?,report_id=?,sha256=?,error_code=?,latency_ms=? WHERE job_id=?",
                      (status, report_id, sha256, error_code, latency_ms, job_id))

    def save_report(self, job_id, report: dict):
        with self._c() as c:
            c.execute("INSERT INTO reports VALUES(?,?,?,?)", (report["report_id"], job_id, report["created"], json.dumps(report)))

    def job(self, job_id):
        with self._c() as c:
            r = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(r) if r else None

    def jobs(self, limit=50):
        with self._c() as c:
            return [dict(r) for r in c.execute("SELECT * FROM jobs ORDER BY created DESC, rowid DESC LIMIT ?", (limit,))]

    def report(self, report_id):
        with self._c() as c:
            r = c.execute("SELECT report_json FROM reports WHERE report_id=?", (report_id,)).fetchone()
        return json.loads(r["report_json"]) if r else None

    def add_review(self, report_id, decision, note) -> dict:
        rep = self.report(report_id)
        hr = {"status": decision + "ed" if decision != "annotate" else "annotated", "decision": decision, "note": note,
              "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        rep["human_review"] = {**rep["human_review"], **hr}
        with self._c() as c:
            c.execute("INSERT INTO reviews(report_id,decision,note,decided_at) VALUES(?,?,?,?)",
                      (report_id, decision, note, hr["decided_at"]))
            c.execute("UPDATE reports SET report_json=? WHERE report_id=?", (json.dumps(rep), report_id))
        return rep
