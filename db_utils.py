import sqlite3
import datetime

# ✅ Database file
DB_FILE = "token_usage.db"

# ✅ Create table if not exists
def create_table():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            api_name TEXT,
            prompt TEXT,
            response TEXT,
            response_time_ms REAL,
            token_count INTEGER
        )
    """)
    conn.commit()
    conn.close()

# ✅ Function to log token usage
def log_token_usage(api_name, prompt, response, response_time, token_count):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO api_logs (timestamp, api_name, prompt, response, response_time_ms, token_count) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.datetime.utcnow().isoformat(), api_name, prompt, response, response_time, token_count))
    conn.commit()
    conn.close()
