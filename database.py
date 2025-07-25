import sqlite3

conn = sqlite3.connect("shortcuts.db")
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS shortcuts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gesture TEXT UNIQUE NOT NULL,
    command TEXT NOT NULL,
    landmarks TEXT NOT NULL,
    image TEXT NOT NULL
            
)''')
conn.commit()
conn.close()

print("Database setup complete!")
