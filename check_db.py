import sqlite3
c = sqlite3.connect('data/pipeline_buffer.db')
print(c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
