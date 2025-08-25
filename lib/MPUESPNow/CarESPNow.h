#ifndef CARESPNOW_H
#define CARESPNOW_H



#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include "MPUSensorData.h"

class CarESPNow {
public:
    CarESPNow();

    bool begin(void (*recvCallback)(const SensorData &));

private:
    static void internalCallback(const esp_now_recv_info_t *info, const uint8_t *data, int len);
    static void (*userCallback)(const SensorData &);
};


#endif //CARESPNOW_H
