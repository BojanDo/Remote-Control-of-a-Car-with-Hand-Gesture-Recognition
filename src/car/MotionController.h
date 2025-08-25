#ifndef MOTIONCONTROLLER_H
#define MOTIONCONTROLLER_H

#include <Arduino.h>
#include <Wire.h>
#include <MPU6050_WE.h>
#include "MotorDriver.h"
#include <MadgwickAHRS.h>

class MotionController {
public:
  MotionController(MotorDriver &motor, MPU6050_WE &mpu);
  void init();
  void update(float pitch, float roll);
  void stop();
  void squareLeft();
  void squareRight();
  void circleLeft();
  void circleRight();
  void spinLeft();
  void spinRight();
  void zigzagLeft();
  void zigzagRight();

private:
  MotorDriver &motor;
  MPU6050_WE &mpu;
  uint8_t mapPitchToSpeed(float pitch);
  void calculateSpeeds(float pitch, float roll, uint8_t &speed_A, uint8_t &speed_B,
                       MotorDriver::Direction &dir_A, MotorDriver::Direction &dir_B, bool &enable);
  void waitUntilYawReached(float targetYaw);
  void zigzagTurnLeft(float segmentAngle);
  void zigzagTurnRight(float segmentAngle);
  void zigzagForward();


  const int SPIN_DURATION = 3000;

  const int TURN_RADIUS = 10;
  const int TURN_DISTANCE_WHEELS = 13;
  const int TURN_FAST_SPEED = 180;
  const int TURN_SLOW_SPEED =
    ((TURN_RADIUS - TURN_DISTANCE_WHEELS / 2.0) / (TURN_RADIUS + TURN_DISTANCE_WHEELS / 2.0))
    * TURN_FAST_SPEED;
  const int TURN_AVG_SPEED = (TURN_FAST_SPEED + TURN_SLOW_SPEED) / 2;

  const int ZIGZAG_SEGMENT_TIME = 750;
  const int ZIGZAG_LEFT_ANGLE = 50;
  const int ZIGZAG_RIGHT_ANGLE = 60;
};



#endif  //MOTIONCONTROLLER_H
