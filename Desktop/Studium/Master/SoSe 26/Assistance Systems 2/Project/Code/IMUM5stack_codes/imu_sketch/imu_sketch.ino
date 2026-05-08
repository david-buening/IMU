 #include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <NTPClient.h>
#include "M5StickCPlus.h"

// ANPASSEN pro Uhr: "R" = rechte Hand (blaues Band), "L" = linke Hand (weisses Band)
const char* hand = "R";
// ANPASSEN: Name und Passwort deines Handy-Hotspots
const char* ssid = "iPhone von Viktor";
const char* password = "1234567x";
// IP deines Laptops im Hotspot-Netz (Mac: Terminal -> ipconfig getifaddr en0)
const char* host = "172.20.10.11";
const uint16_t port = 5005;



bool is_started = false;
WiFiClient client;
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP);
unsigned long startTime = 0;

int press_count = 0;

void setup() {
  M5.begin();
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.println("Starting up...");
  M5.Axp.ScreenBreath(8);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    M5.Lcd.println("Connecting to WiFi...");
  }
  M5.Lcd.println("WiFi connected");

  // Initialize NTP Client
  timeClient.begin();
  timeClient.update();

  // Setup IMU
  M5.IMU.Init();
  M5.Lcd.println("IMU Initialized");

  // Connect to server
  if (!client.connect(host, port)) {
    M5.Lcd.println("Connection to host failed");
    while (1); // halt if cannot connect to server
  }
  M5.Lcd.println("Connected to server");

  // Send the hand value to server
  sendToServer(hand);

  // Wait for server to send "Start"
  while (!is_started) {
    String response = client.readStringUntil('\n');
    if (response == "Start") {
      M5.Lcd.println("Received 'Start' from server");
      M5.Lcd.println("Experiment ON!");
      is_started = true;
      startTime = millis();
    }
  }
}

void loop() {
  // Do nothing until "Start" is received
  if (!is_started) {
    return;
  }
  M5.update();
  // Obtain gyro data
  float ax, ay, az, gx, gy, gz;
  M5.IMU.getGyroData(&gx, &gy, &gz);

  // Obtain accelerometer data
  M5.IMU.getAccelData(&ax, &ay, &az);

  // Get current timestamp from NTPClient
  timeClient.update();
  unsigned long timestamp = timeClient.getEpochTime();
  // ms since process started
  unsigned long processTime_ms = millis() - startTime;

  if (M5.BtnA.wasPressed()) {
    press_count++;
    // Send gyro data with timestamp and processTime_ms
    String dataToSend = String(processTime_ms) + ", "
                    + String(gx) + ", " + String(gy) + ", " + String(gz) + ", "
                    + String(ax) + ", " + String(ay) + ", " + String(az) + ", "
                    + String('s');

    sendToServer(dataToSend);
    // Flash the light white for 0.5 seconds.
    M5.Axp.ScreenBreath(14);
    M5.Lcd.fillScreen(WHITE);
    delay(500);

    M5.Lcd.fillScreen(BLACK);
    M5.Axp.ScreenBreath(0);

  }
  else {
     // Send gyro data with timestamp and processTime_ms
     String dataToSend = String(processTime_ms) + ", "
                    + String(gx) + ", " + String(gy) + ", " + String(gz) + ", "
                    + String(ax) + ", " + String(ay) + ", " + String(az) + ", "
                    + String('n');

    sendToServer(dataToSend);
    }

  // last press, so we know the watches worked
    if (press_count >= 6){
            M5.Lcd.fillScreen(RED);
            M5.Axp.ScreenBreath(7);
      }
  delay(10);
}

void sendToServer(const String& data) {
  if (client.connected()) {
    client.println(data);
  } else {
    M5.Lcd.println("Connection to host failed");
    M5.Lcd.fillScreen(RED);
    M5.Axp.ScreenBreath(7);
    while (10);
  }
}
