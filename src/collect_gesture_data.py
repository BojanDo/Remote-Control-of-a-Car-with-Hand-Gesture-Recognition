import serial
import time
import csv
import os
from pathlib import Path


# === CONFIGURATION ===
PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
SAMPLES_PER_SEQUENCE = 500

NUM_SEQUENCES = 30
NUM_NONE_SEQUENCES = 30

GESTURES = [
    "circle_left",
    "circle_right",
    "spin_left",
    "spin_right",
    "zigzag_left",
    "zigzag_right",
    "none",
]

GESTURE_COLORS = {
    "circle_left": "\033[94m",
    "circle_right": "\033[92m",
    "spin_left": "\033[96m",
    "spin_right": "\033[95m",
    "zigzag_left": "\033[93m",
    "zigzag_right": "\033[91m",
    "none": "\033[90m",
}

RESET_COLOR = "\033[0m"


OUTPUT_DIR = "test_data"

def print_gesture_prompt(gesture: str, index: int):
    color = GESTURE_COLORS.get(gesture, "\033[97m")
    print(f"\n{color}Perform gesture: {gesture.upper()} ({index}/{NUM_SEQUENCES}){RESET_COLOR}")

def print_next_gesture_prompt(gesture: str):
    color = GESTURE_COLORS.get(gesture, "\033[97m")
    print(f"\n{color}Next gesture: {gesture.upper()}{RESET_COLOR}")
    input("Press ENTER to continue...")


# === SETUP ===
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Give time for the ESP32 to reset
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def flush_serial():
    ser.reset_input_buffer()

def read_sequence():
    sequence = []
    while len(sequence) < SAMPLES_PER_SEQUENCE:
        try:
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue
            values = list(map(float, line.split(",")))
            if len(values) != 6:
                continue  # skip malformed lines
            sequence.append(values)
        except Exception as e:
            print(f"Error: {e}")
    return sequence

def save_sequence(data, gesture, index):
    gesture_dir = os.path.join(OUTPUT_DIR, gesture)
    Path(gesture_dir).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(gesture_dir, f"{gesture}_{index:03}.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ax", "ay", "az", "gx", "gy", "gz"])
        writer.writerows(data)
    print(f"Saved: {filename}")

def countdown():
    for i in range(3, 0, -1):
        print(i)
        time.sleep(0.5)
    print('Start...')

# === MAIN LOOP ===

if __name__=="__main__":
    index_tracker = {}

    for gesture in GESTURES + ["none"]:
        gesture_dir = os.path.join(OUTPUT_DIR, gesture)
        Path(gesture_dir).mkdir(parents=True, exist_ok=True)
        existing_files = [f for f in os.listdir(gesture_dir) if f.endswith(".csv")]
        index_tracker[gesture] = len(existing_files)

    print("Starting gesture recording...\n")

    for gesture in GESTURES:
        print_next_gesture_prompt(gesture)
        for i in range(NUM_SEQUENCES):
            print_gesture_prompt(gesture,i)
            countdown()
            flush_serial()
            sequence = read_sequence()
            save_sequence(sequence, gesture, index_tracker[gesture])
            index_tracker[gesture] += 1

    print("\nAll gesture data collected!")
    ser.close()



