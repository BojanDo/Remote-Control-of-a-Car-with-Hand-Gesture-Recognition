#include "CarESPNow.h"

void (*CarESPNow::userCallback)(const SensorData &) = nullptr;

CarESPNow::CarESPNow() {}

bool CarESPNow::begin(void (*recvCallback)(const SensorData &)) {
    WiFi.mode(WIFI_STA);

    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        return false;
    }

    userCallback = recvCallback;
    esp_now_register_recv_cb(internalCallback);

    return true;
}

void CarESPNow::internalCallback(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (len != sizeof(SensorData)) {
        Serial.println("Received data size mismatch");
        return;
    }

    const SensorData *receivedData = reinterpret_cast<const SensorData *>(data);

    if (userCallback) {
        userCallback(*receivedData);
    }
}
