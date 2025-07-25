import sqlite3

conn = sqlite3.connect("shortcuts.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM shortcuts")
rows = cursor.fetchall()
conn.close()

for row in rows:
    print(row)  # This should print (gesture, command, landmarks)
