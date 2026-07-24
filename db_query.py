import sqlite3

db_path = r"models\voice_profiles.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("=== TABLES ===")
print(tables)

for t in tables:
    cur.execute(f"SELECT * FROM {t}")
    rows = cur.fetchall()
    print(f"\n=== {t} ({len(rows)} rows) ===")
    if rows:
        cols = list(rows[0].keys())
        print("cols:", cols)
        for r in rows:
            d = dict(r)
            for k in list(d.keys()):
                v = d[k]
                if isinstance(v, (bytes, bytearray)):
                    d[k] = f"<blob {len(v)} bytes>"
            print(d)

conn.close()
