#include "MotorDriver.h"

void MotorDriver::init() {
  ledcAttach(PIN_Motor_PWMA, 1000, 8);
  ledcAttach(PIN_Motor_PWMB, 1000, 8);
  pinMode(PIN_Motor_AIN_1, OUTPUT);
  pinMode(PIN_Motor_BIN_1, OUTPUT);
  pinMode(PIN_Motor_STBY, OUTPUT);
}

void MotorDriver::setSpeedAndDirection(Direction direction_A, uint8_t speed_A,
                                        Direction direction_B, uint8_t speed_B,
                                        bool enableControl) {
  if (enableControl) {
    digitalWrite(PIN_Motor_STBY, HIGH);

    setMotor(PIN_Motor_PWMA, PIN_Motor_AIN_1, direction_A, speed_A);  // Right
    setMotor(PIN_Motor_PWMB, PIN_Motor_BIN_1, direction_B, speed_B);  // Left
  } else {
    digitalWrite(PIN_Motor_STBY, LOW);
  }
}

void MotorDriver::setMotor(uint8_t pwmPin, uint8_t dirPin, Direction direction,
                            uint8_t speed) {
  switch (direction) {
    case FORWARD:
      digitalWrite(dirPin, HIGH);
      ledcWrite(pwmPin, speed);
      break;

    case BACKWARD:
      digitalWrite(dirPin, LOW);
      ledcWrite(pwmPin, speed);
      break;

    case STOP:
      ledcWrite(pwmPin, 0);
      digitalWrite(PIN_Motor_STBY, LOW);
      break;
  }
}
