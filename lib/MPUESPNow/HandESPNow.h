#ifndef HANDESPNOW_H
#define HANDESPNOW_H

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include "MPUSensorData.h"

class HandESPNow {
public:
    HandESPNow(const uint8_t *peerMAC);

    bool begin();
    esp_err_t sendData(const SensorData &data);

private:
    uint8_t receiverMAC[6];
};

#endif //HANDESPNOW_H
