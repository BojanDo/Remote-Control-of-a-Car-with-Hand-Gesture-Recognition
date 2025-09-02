#include "IMUBuffer.h"
#include "model_data.h"

IMUBuffer::IMUBuffer() {}

float IMUBuffer::standardize(float value, int index) {
    return (value - imu_mean[index]) / imu_std[index];
}

void IMUBuffer::addSample(const SensorData& data) {
    if (count >= WINDOW_SIZE) return;  // Prevent overflow

    Sample& sample = buffer[count];
    for (int i = 0; i < 3; ++i) {
        sample.values[i] = standardize(data.accel[i], i);
        sample.values[i + 3] = standardize(data.gyro[i], i + 3);
    }

    ++count;
}

bool IMUBuffer::isFull() const {
    return count == WINDOW_SIZE;
}

void IMUBuffer::getWindow(float* out) {
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < NUM_AXES; ++j) {
            *out++ = buffer[i].values[j];
        }
    }
}

void IMUBuffer::reset() {
    count = 0;
}
