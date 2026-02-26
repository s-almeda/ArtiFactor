# jobs/worker.py
# Background worker for processing jobs from the jobs.db queue


import sqlite3
import time
import os
import sys

# Add the app directory (parent of jobs and templates) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Remove top-level import of process_map_job to avoid circular import issues.

DB_PATH = os.path.join(os.path.dirname(__file__), 'jobs.db')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')

# Ensure jobs.db and schema exist

def init_db():
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, 'r') as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()



def fetch_next_job():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1")
    job = cur.fetchone()
    conn.close()
    return job



# This function is now unused; job status is updated by process_map_job via jobs/__init__.py



def worker_loop():
    print("[Worker] Starting job worker loop...")
    from jobs import update_job_status
    from jobs.map_job_processor import process_map_job
    import json
    while True:
        job = fetch_next_job()
        if job:
            job_id = job['job_id'] if 'job_id' in job.keys() else job.get('id')
            print(f"[Worker] Processing job {job_id}")
            try:
                request_params = job['request_params']
                if isinstance(request_params, str):
                    request_params = json.loads(request_params)
                process_map_job(job_id, request_params, update_job_status)
                print(f"[Worker] Job {job_id} completed (process_map_job called).")
            except Exception as inner_e:
                print(f"[Worker] Could not process job {job_id}: {inner_e}")
                try:
                    update_job_status(job_id, 'failed', error_message=str(inner_e))
                except Exception as status_e:
                    print(f"[Worker] Could not update job status for {job_id}: {status_e}")
        else:
            time.sleep(1)

if __name__ == "__main__":
    init_db()
    worker_loop()
