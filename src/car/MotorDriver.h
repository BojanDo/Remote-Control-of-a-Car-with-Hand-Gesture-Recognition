#ifndef MOTOR_DRIVER_H
#define MOTOR_DRIVER_H

#include <Arduino.h>

class MotorDriver
{
public:
    enum Direction
    {
        FORWARD,
        BACKWARD,
        STOP
      };

    void init();
    void setSpeedAndDirection(Direction direction_A, uint8_t speed_A,
                              Direction direction_B, uint8_t speed_B,
                              bool enableControl);

private:
    static const uint8_t PIN_Motor_PWMA = 16;  // Right Motor Speed
    static const uint8_t PIN_Motor_PWMB = 27;  // Left Motor Speed
    static const uint8_t PIN_Motor_BIN_1 = 12; // Left Motor Direction
    static const uint8_t PIN_Motor_AIN_1 = 14; // Right Motor Direction
    static const uint8_t PIN_Motor_STBY  = 25; // Motor Driver Standby

    static const uint8_t speed_Max = 255;

    void setMotor(uint8_t pwmPin, uint8_t dirPin, Direction direction, uint8_t speed);
};

#endif
