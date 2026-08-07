import sqlite3

conn = sqlite3.connect("data/matching/database.db")
cursor = conn.cursor()

# Average keypoints per image
cursor.execute("SELECT * FROM keypoints")
print(cursor.fetchall())
keypoints = [row[0] for row in cursor.fetchall()]
print(f"Avg Keypoints/Image: {sum(keypoints)/len(keypoints):.0f}")

# Total matches
cursor.execute("SELECT count(*) FROM matches WHERE rows > 0")
print(f"Valid Matched Image Pairs: {cursor.fetchone()[0]}")