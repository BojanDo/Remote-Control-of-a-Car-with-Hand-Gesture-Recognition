#include <WiFi.h>

void setup() {
    Serial.begin(115200);
    delay(1000);

    String macAddress = WiFi.macAddress();
    Serial.println("ESP32 MAC Address:");
    Serial.println(macAddress);
}

void loop() {
}
