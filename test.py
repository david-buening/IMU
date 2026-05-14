import socket
import csv
import time
import re
import os

HOST = "172.20.10.11"
PORT = 5005
TARGET_HZ = 10
INTERVAL = 1.0 / TARGET_HZ  # 100ms

os.makedirs("data", exist_ok=True)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"Server läuft auf {HOST}:{PORT} – warte auf Uhr...")

conn, addr = server.accept()
hand = conn.recv(1024).decode().strip()
print(f"Verbunden! Uhr '{hand}' von {addr}")

filename = f"data/test_{hand}_{int(time.time())}.csv"
f = open(filename, "w", newline="")
writer = csv.writer(f)
writer.writerow(["timestamp_ms", "GX", "GY", "GZ", "AX", "AY", "AZ", "sync"])

input("Enter drücken um Aufnahme zu starten...")
conn.sendall(b"Start\n")
print(f"Aufnahme läuft @ {TARGET_HZ}Hz → {filename}")
print("Strg+C zum Beenden\n")

last_write = 0
row_count = 0

try:
    buffer = ""
    while True:
        chunk = conn.recv(1024).decode()
        if not chunk:
            break
        buffer += chunk
        packets = re.split(r'(?<=[sn])\n?', buffer)
        buffer = packets.pop()  # letztes unvollständiges Paket zurückhalten

        for p in packets:
            p = p.strip()
            if not p:
                continue
            now = time.time()
            if now - last_write >= INTERVAL:
                row = p.split(", ")
                if len(row) == 8:
                    writer.writerow(row)
                    row_count += 1
                    last_write = now
                    print(f"[{row_count:4d}] {p}")

except KeyboardInterrupt:
    pass

f.close()
conn.sendall(b"End\n")
conn.close()
print(f"\nTest beendet. {row_count} Zeilen gespeichert in {filename}")
