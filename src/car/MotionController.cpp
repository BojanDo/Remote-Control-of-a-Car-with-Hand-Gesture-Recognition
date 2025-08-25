#include "MotionController.h"

MotionController::MotionController(MotorDriver &m, MPU6050_WE &mp)
  : motor(m), mpu(mp) {}

void MotionController::init() {
  motor.init();

  if (!mpu.init()) {
    Serial.println("MPU6050 does not respond");
  } else {
    Serial.println("MPU6050 is connected");
  }

  Serial.println("Position you MPU6050 flat and don't move it - calibrating...");
  delay(1000);
  mpu.autoOffsets();
  Serial.println("Done!");

  mpu.setSampleRateDivider(5);
  mpu.setAccRange(MPU6500_ACC_RANGE_2G);
  mpu.enableAccDLPF(true);
  mpu.setAccDLPF(MPU6500_DLPF_3);
}

uint8_t MotionController::mapPitchToSpeed(float pitch) {
  if (pitch >= 60.0) return 255;
  if (pitch >= 10.0) return map(pitch, 10, 60, 0, 255);
  if (pitch <= -60.0) return 255;
  if (pitch <= -10.0) return map(pitch, -10, -60, 0, 255);
  return 0;
}

void MotionController::calculateSpeeds(float pitch, float roll, uint8_t &speed_A, uint8_t &speed_B,
                                       MotorDriver::Direction &dir_A, MotorDriver::Direction &dir_B, bool &enable) {
  int baseSpeed = mapPitchToSpeed(pitch);
  enable = (baseSpeed > 0);

  if (pitch > 10) {
    dir_A = dir_B = MotorDriver::FORWARD;
  } else if (pitch < -10) {
    dir_A = dir_B = MotorDriver::BACKWARD;
  } else {
    dir_A = dir_B = MotorDriver::STOP;
    speed_A = speed_B = 0;
    return;
  }

  if (abs(roll) < 10) roll = 0;

  float ratio = map(roll, -60, 60, -60, 60) / 100.0f;
  int diff = baseSpeed * ratio * 0.3f;

  int leftSpeed = baseSpeed - diff;
  int rightSpeed = baseSpeed + diff;

  leftSpeed = constrain(leftSpeed, 0, 255);
  rightSpeed = constrain(rightSpeed, 0, 255);

  speed_A = leftSpeed;
  speed_B = rightSpeed;
}

void MotionController::update(float pitch, float roll) {
  uint8_t speed_A = 0, speed_B = 0;
  MotorDriver::Direction dir_A = MotorDriver::STOP;
  MotorDriver::Direction dir_B = MotorDriver::STOP;
  bool enable = false;

  calculateSpeeds(pitch, roll, speed_A, speed_B, dir_A, dir_B, enable);
  motor.setSpeedAndDirection(dir_A, speed_A, dir_B, speed_B, enable);
}


void MotionController::stop() {
  motor.setSpeedAndDirection(MotorDriver::STOP, 0, MotorDriver::STOP, 0, true);
}

void MotionController::circleLeft() {
  motor.setSpeedAndDirection(
    MotorDriver::FORWARD, TURN_FAST_SPEED,
    MotorDriver::FORWARD, TURN_SLOW_SPEED,
    true
  );
  waitUntilYawReached(360.0);
}

void MotionController::circleRight() {
  motor.setSpeedAndDirection(
    MotorDriver::FORWARD, TURN_SLOW_SPEED,
    MotorDriver::FORWARD, TURN_FAST_SPEED,
    true
  );
  waitUntilYawReached(-360.0);
}

void MotionController::waitUntilYawReached(float targetYaw) {
  float yaw = 0.0;
  unsigned long lastTime = micros();
  unsigned long startTime = millis();
  const unsigned long timeout = 5000;

  while ((targetYaw > 0 && yaw < targetYaw) || (targetYaw < 0 && yaw > targetYaw)) {
    unsigned long currentTime = micros();
    float dt = (currentTime - lastTime) / 1000000.0;
    lastTime = currentTime;

    xyzFloat gyro = mpu.getGyrValues();
    yaw += gyro.z * dt;

    if (millis() - startTime > timeout) {
      Serial.println("Timeout reached");
      break;
    }
  }
  motor.setSpeedAndDirection(MotorDriver::STOP, 0, MotorDriver::STOP, 0, true);
}


void MotionController::spinLeft() {
  motor.setSpeedAndDirection(MotorDriver::BACKWARD, 255, MotorDriver::FORWARD, 255, true);
  delay(SPIN_DURATION);
  motor.setSpeedAndDirection(MotorDriver::STOP, 0, MotorDriver::STOP, 0, true);
}

void MotionController::spinRight() {
  motor.setSpeedAndDirection(MotorDriver::FORWARD, 255, MotorDriver::BACKWARD, 255, true);
  delay(SPIN_DURATION);
  motor.setSpeedAndDirection(MotorDriver::STOP, 0, MotorDriver::STOP, 0, true);
}



void MotionController::zigzagTurnRight(float segmentAngle) {
  motor.setSpeedAndDirection(MotorDriver::FORWARD, TURN_SLOW_SPEED, MotorDriver::FORWARD, TURN_FAST_SPEED, true);
  waitUntilYawReached(-segmentAngle);
  motor.setSpeedAndDirection(MotorDriver::FORWARD, TURN_AVG_SPEED, MotorDriver::FORWARD, TURN_AVG_SPEED, true);
  zigzagForward();
}

void MotionController::zigzagTurnLeft(float segmentAngle) {
  motor.setSpeedAndDirection(MotorDriver::FORWARD, TURN_FAST_SPEED, MotorDriver::FORWARD, TURN_SLOW_SPEED, true);
  waitUntilYawReached(segmentAngle);
  zigzagForward();
}

void MotionController::zigzagForward() {
  motor.setSpeedAndDirection(MotorDriver::FORWARD, TURN_AVG_SPEED, MotorDriver::FORWARD, TURN_AVG_SPEED, true);
  delay(ZIGZAG_SEGMENT_TIME);
}


void MotionController::zigzagRight() {
  zigzagTurnRight(ZIGZAG_RIGHT_ANGLE/2);
  zigzagTurnLeft(ZIGZAG_LEFT_ANGLE);
  zigzagTurnRight(ZIGZAG_RIGHT_ANGLE);
  zigzagTurnLeft(ZIGZAG_LEFT_ANGLE/2);
  motor.setSpeedAndDirection(MotorDriver::STOP, 0, MotorDriver::STOP, 0, true);
}
void MotionController::zigzagLeft() {
  zigzagTurnLeft(ZIGZAG_LEFT_ANGLE/2);
  zigzagTurnRight(ZIGZAG_RIGHT_ANGLE);
  zigzagTurnLeft(ZIGZAG_LEFT_ANGLE);
  zigzagTurnRight(ZIGZAG_RIGHT_ANGLE/2);
  motor.setSpeedAndDirection(MotorDriver::STOP, 0, MotorDriver::STOP, 0, true);
}
