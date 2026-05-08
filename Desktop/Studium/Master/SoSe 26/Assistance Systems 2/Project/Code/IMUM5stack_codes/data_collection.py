import socket
import csv
import time
import re
import os
from threading import Thread, Event

# ── ANPASSEN ──────────────────────────────────────────
DATASET_NAME = "versuch_David_both_hands_small_floor_2"
HOST         = "172.20.10.7"
PORT         = 5005
# ──────────────────────────────────────────────────────

PHASES    = ["Aufheben", "Laufen", "Absetzen"]
COLUMNS   = ["time_s", "server_time_s", "GX", "GY", "GZ", "AX", "AY", "AZ", "phase"]

current_phase = None  # von Threads gelesen, vom Hauptthread gesetzt
stop_event    = Event()
record_start  = None  # Server-Zeitstempel beim Start

os.makedirs("./data", exist_ok=True)

# ── Server starten ────────────────────────────────────
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(2)
print(f"Server läuft auf {HOST}:{PORT}")
print("Warte auf beide Uhren...\n")

# Beide Verbindungen annehmen (Reihenfolge egal)
conn1, addr1 = server.accept()
hand1 = conn1.recv(1024).decode().strip()
print(f"  Uhr '{hand1}' verbunden ({addr1[0]})")

conn2, addr2 = server.accept()
hand2 = conn2.recv(1024).decode().strip()
print(f"  Uhr '{hand2}' verbunden ({addr2[0]})")

# CSV-Dateien anlegen
files   = {}
writers = {}
conns   = {hand1: conn1, hand2: conn2}

for hand in [hand1, hand2]:
    path = f"./data/{DATASET_NAME}_{hand}.csv"
    f = open(path, "w", newline="")
    writers[hand] = csv.writer(f)
    writers[hand].writerow(COLUMNS)
    files[hand] = f
    print(f"  → Speichert nach: {path}")

# ── Daten-Thread pro Uhr ──────────────────────────────
def handle_watch(conn, writer, hand):
    buffer = ""
    while not stop_event.is_set():
        try:
            chunk = conn.recv(1024).decode()
        except:
            break
        if not chunk:
            break
        buffer += chunk
        packets = re.split(r'(?<=[sn])\n?', buffer)
        buffer = packets.pop()  # unvollständiges Paket für nächsten recv

        for p in packets:
            p = p.strip()
            if not p:
                continue
            if current_phase is None:
                continue  # noch nicht gestartet oder schon beendet
            parts = p.split(", ")
            if len(parts) != 8:
                continue
            time_s        = round(int(parts[0]) / 1000.0, 3)
            server_time_s = round(time.time() - record_start, 3)
            row = [time_s, server_time_s] + parts[1:7] + [current_phase]
            writer.writerow(row)

# ── Phasen-Steuerung (Hauptthread) ───────────────────
print("\n" + "="*45)
print(f"  Dataset: {DATASET_NAME}")
print("="*45)
input("\nEnter drücken um Aufnahme zu starten...")

# Start-Signal an beide Uhren
record_start = time.time()
conn1.sendall(b"Start\n")
conn2.sendall(b"Start\n")

# Threads starten
threads = []
for hand, conn in conns.items():
    t = Thread(target=handle_watch, args=(conn, writers[hand], hand), daemon=True)
    t.start()
    threads.append(t)

# Phasen durchlaufen
for phase in PHASES:
    current_phase = phase
    if phase != PHASES[-1]:
        next_phase = PHASES[PHASES.index(phase) + 1]
        input(f"\n▶ Phase: {phase}  –  Enter für '{next_phase}'...")
    else:
        input(f"\n▶ Phase: {phase}  –  Enter zum Beenden...")

# Aufnahme stoppen
current_phase = None
stop_event.set()

conn1.sendall(b"End\n")
conn2.sendall(b"End\n")

for t in threads:
    t.join(timeout=2)

for f in files.values():
    f.close()

conn1.close()
conn2.close()
server.close()

print("\n✓ Aufnahme beendet.")
print(f"  Dateien gespeichert in ./data/{DATASET_NAME}_R.csv  und  ./data/{DATASET_NAME}_L.csv")
