import socket
import csv
import time
from threading import Thread
import re
import os

# Server settings
HOST = "172.20.10.7"  # IP des Laptops im Hotspot-Netz
PORT = 5005

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(2)

CSV_COLUMN_NAMES = ['timestamp_mills_ms', 'GX', 'GY', 'GZ', 'AX', 'AY', 'AZ', 'sync', 'timestamp_listener']

os.makedirs("./data", exist_ok=True)

print("Server started, waiting for connections...")

user_id = input("Enter a user ID: ")
print("Waiting for both watches to connect...")

conn1, addr1 = server.accept()
conn2, addr2 = server.accept()

hand1 = conn1.recv(1024).decode().strip()
hand2 = conn2.recv(1024).decode().strip()

file1 = open(f"./data/{user_id}_{hand1}.csv", "w", newline='')
file2 = open(f"./data/{user_id}_{hand2}.csv", "w", newline='')

writer1 = csv.writer(file1)
writer2 = csv.writer(file2)

writer1.writerow(CSV_COLUMN_NAMES)
writer2.writerow(CSV_COLUMN_NAMES)

print(f"Connected: watch1 ({hand1}) at {addr1}, watch2 ({hand2}) at {addr2}")

input("Press ENTER to start the test")

conn1.sendall(b"Start\n")
conn2.sendall(b"Start\n")

def handle_watch(conn, writer, watch):
    while True:
        data = conn.recv(1024).decode().strip()
        if not data or data == "":
            pass
        elif data == "End":
            print(f"Watch {watch}: connection terminated!")
            break
        else:
            packages = re.split(r'(?<=[sn])', data)
            for i, p in enumerate(packages):
                if len(p) > 1:
                    row = p.split(", ")
                    row.append(str(int(time.time() * 1000)))
                    writer.writerow(row)
                    print(f"Watch {watch}:", row)

thread1 = Thread(target=handle_watch, args=(conn1, writer1, '1'))
thread2 = Thread(target=handle_watch, args=(conn2, writer2, '2'))

thread1.start()
thread2.start()

input("Press ENTER to end the test")

file1.close()
file2.close()

conn1.sendall(b"End\n")
conn2.sendall(b"End\n")

thread1.join()
thread2.join()

conn1.close()
conn2.close()

print("Test ended")
