#ifndef MPUSENSORDATA_H
#define MPUSENSORDATA_H

typedef struct {
    float accel[3];
    float gyro[3];
    float pitch;
    float roll;
    bool isButtonPressed;
} SensorData;

#endif //MPUSENSORDATA_H
