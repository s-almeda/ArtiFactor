import sqlite3
import json
import uuid
import os
from datetime import datetime

# Path to jobs database
JOBS_DB_PATH = os.path.join(os.path.dirname(__file__), 'jobs.db')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')

def init_jobs_db():
    """Initialize the jobs database with schema"""
    try:
        with sqlite3.connect(JOBS_DB_PATH) as conn:
            # Read and execute schema
            with open(SCHEMA_PATH, 'r') as f:
                schema = f.read()
            conn.executescript(schema)
            conn.commit()
        print(f"✓ Jobs database initialized at {JOBS_DB_PATH}")
    except Exception as e:
        print(f"⚠ Failed to initialize jobs database: {e}")
        raise

def create_job(request_params):
    """Create a new job and return job_id"""
    job_id = str(uuid.uuid4())[:8]  # Short UUID
    
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.execute("""
            INSERT INTO jobs (job_id, request_params)
            VALUES (?, ?)
        """, (job_id, json.dumps(request_params)))
        conn.commit()
    
    return job_id

def get_job_status(job_id):
    """Get current status of a job"""
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT job_id, status, progress_message, cache_key, error_message,
                   created_at, started_at, completed_at
            FROM jobs WHERE job_id = ?
        """, (job_id,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

def update_job_status(job_id, status, progress_message=None, cache_key=None, error_message=None):
    """Update job status and progress"""
    updates = ['status = ?']
    params = [status]
    
    if progress_message:
        updates.append('progress_message = ?')
        params.append(progress_message)
    
    if cache_key:
        updates.append('cache_key = ?')
        params.append(cache_key)
    
    if error_message:
        updates.append('error_message = ?')
        params.append(error_message)
    
    # Set timestamps based on status
    if status == 'processing':
        updates.append('started_at = CURRENT_TIMESTAMP')
    elif status in ['completed', 'failed']:
        updates.append('completed_at = CURRENT_TIMESTAMP')
    
    query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
    params.append(job_id)
    
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.execute(query, params)
        conn.commit()

def get_pending_jobs():
    """Get all pending jobs for the worker"""
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT job_id, request_params
            FROM jobs 
            WHERE status = 'pending'
            ORDER BY created_at ASC
        """)
        return [dict(row) for row in cursor.fetchall()]

def cleanup_old_jobs(days_old=7):
    """Clean up completed/failed jobs older than specified days"""
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.execute("""
            DELETE FROM jobs 
            WHERE status IN ('completed', 'failed') 
            AND datetime(created_at) < datetime('now', '-{} days')
        """.format(days_old))
        conn.commit()