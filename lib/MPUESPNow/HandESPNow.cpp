#include "HandESPNow.h"
#include <cstring>

HandESPNow::HandESPNow(const uint8_t *peerMAC) {
    memcpy(receiverMAC, peerMAC, 6);
}

bool HandESPNow::begin() {
    WiFi.mode(WIFI_STA);

    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        return false;
    }

    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, receiverMAC, 6);
    peerInfo.channel = 0;
    peerInfo.encrypt = false;

    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        Serial.println("Failed to add peer");
        return false;
    }
    return true;
}

esp_err_t HandESPNow::sendData(const SensorData &data) {
    return esp_now_send(receiverMAC, (uint8_t *)&data, sizeof(data));
}
