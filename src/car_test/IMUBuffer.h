#ifndef IMUBUFFER_H
#define IMUBUFFER_H

#include <Arduino.h>
#include "MPUSensorData.h"

#define NUM_AXES 6
#define WINDOW_SIZE 500

class IMUBuffer {
public:
    IMUBuffer();
    void addSample(const SensorData& data);
    bool isFull() const;
    void getWindow(float* out);
    void reset();

private:
    float standardize(float value, int index);

    struct Sample {
        float values[NUM_AXES];
    };

    Sample buffer[WINDOW_SIZE];
    int head = 0;
    int count = 0;
};



#endif //IMUBUFFER_H
