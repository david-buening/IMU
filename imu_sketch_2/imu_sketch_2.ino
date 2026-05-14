#include <WiFi.h>
#include "M5StickCPlus.h"

// ── ANPASSEN ──────────────────────────────────────────
const char* hand     = "R";               // "R" oder "L"
const char* ssid     = "Davids Handy";
const char* password = "12345679";
const char* host     = "172.20.10.7";    // Laptop-IP (Terminal: ipconfig getifaddr en0)
const uint16_t port  = 5005;
// ──────────────────────────────────────────────────────

WiFiClient client;
unsigned long startTime = 0;
unsigned long sampleCount = 0;
unsigned long lastDisplayUpdate = 0;

void connectToServer() {
  sampleCount = 0;

  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.println(String("Hand: ") + hand);

  // WiFi sicherstellen
  if (WiFi.status() != WL_CONNECTED) {
    M5.Lcd.println("WiFi...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
    }
    M5.Lcd.println("WiFi OK");
  }

  // Server verbinden
  client.stop();
  M5.Lcd.println("Server...");
  while (!client.connect(host, port)) {
    M5.Lcd.println("retry...");
    delay(2000);
  }
  M5.Lcd.println("Server OK");

  // Hand senden
  client.println(hand);

  // Auf Start warten
  M5.Lcd.println("Warte...");
  while (true) {
    if (client.available()) {
      String msg = client.readStringUntil('\n');
      msg.trim();
      if (msg == "Start") break;
    }
  }

  M5.Lcd.fillScreen(GREEN);
  M5.Lcd.println(String("Hand: ") + hand);
  M5.Lcd.println("100 Hz");
  startTime = millis();
  lastDisplayUpdate = millis();
}

void setup() {
  M5.begin();
  M5.IMU.Init();
  M5.Lcd.setTextSize(2);

  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.println("WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  M5.Lcd.println("WiFi OK");
  M5.Lcd.println(WiFi.localIP());
  delay(1000);

  connectToServer();
}

void loop() {
  M5.update();

  float ax, ay, az, gx, gy, gz;
  M5.IMU.getGyroData(&gx, &gy, &gz);
  M5.IMU.getAccelData(&ax, &ay, &az);

  unsigned long t = millis() - startTime;
  char sync = M5.BtnA.wasPressed() ? 's' : 'n';

  String data = String(t) + ", "
              + String(gx) + ", " + String(gy) + ", " + String(gz) + ", "
              + String(ax) + ", " + String(ay) + ", " + String(az) + ", "
              + String(sync);

  if (client.connected()) {
    client.println(data);
    sampleCount++;

    if (millis() - lastDisplayUpdate >= 1000) {
      M5.Lcd.fillScreen(GREEN);
      M5.Lcd.println(String("Hand: ") + hand);
      M5.Lcd.println(String("n=") + sampleCount);
      lastDisplayUpdate = millis();
    }

    if (sync == 's') {
      M5.Lcd.fillScreen(WHITE);
      delay(100);
      M5.Lcd.fillScreen(GREEN);
    }

  } else {
    // Verbindung weg → automatisch neu verbinden
    connectToServer();
  }

  delay(10); // 100 Hz
}
