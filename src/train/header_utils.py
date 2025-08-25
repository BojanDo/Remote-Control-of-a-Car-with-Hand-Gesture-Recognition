import subprocess
import fileinput
import numpy as np
from config import HEADER_OUTPUT_PATH

def append_normalization_to_header(mean, std):
    with open(HEADER_OUTPUT_PATH, "a") as f:
        f.write("\n// IMU normalization parameters\n")
        f.write(f"#define IMU_FEATURES {len(mean)}\n\n")
        f.write(f"static const float imu_mean[IMU_FEATURES] = {{{', '.join(f'{x:.6f}' for x in mean)}}};\n")
        f.write(f"static const float imu_std[IMU_FEATURES] = {{{', '.join(f'{x:.6f}' for x in std)}}};\n")

def append_gesture_enum_to_header(gestures):
    with open(HEADER_OUTPUT_PATH, "a") as f:
        f.write("\n// Gesture enumeration based on model output indices\n")
        f.write("typedef enum {\n")
        for i, gesture in enumerate(gestures):
            f.write(f"    {gesture} = {i},\n")
        f.write("} Gesture;\n")
        f.write("\nstatic inline Gesture intToGesture(int value) {\n")
        f.write("    switch (value) {\n")
        for i, gesture in enumerate(gestures):
            f.write(f"        case {i}: return {gesture};\n")
        f.write("        default: return none;\n")
        f.write("    }\n}\n")

def create_header_from_model(tflite_model, input_scale, input_zero_point, output_scale, output_zero_point, mean, std, gestures):
    with open("gesture_model.tflite", "wb") as f:
        f.write(tflite_model)

    subprocess.run(["xxd", "-i", "gesture_model.tflite", HEADER_OUTPUT_PATH], check=True)

    for line in fileinput.input(HEADER_OUTPUT_PATH, inplace=True):
        if fileinput.filelineno() == 1:
            print("""#ifndef MODELDATA_H
#define MODELDATA_H

inline const unsigned char gesture_model_tflite[] = {""")
        elif "unsigned int gesture_model_tflite_len" in line:
            print(line.replace("unsigned int", "inline const unsigned int"), end="")
        else:
            print(line, end="")

    with open(HEADER_OUTPUT_PATH, "a") as f:
        f.write(f"\n#define GESTURE_CLASSES {len(gestures)}\n")

        f.write("\n// Quantization parameters\n")
        f.write(f"#define INPUT_SCALE {input_scale:.17g}f\n")
        f.write(f"#define INPUT_ZERO_POINT {input_zero_point}\n")
        f.write(f"#define OUTPUT_SCALE {output_scale:.17g}f\n")
        f.write(f"#define OUTPUT_ZERO_POINT {output_zero_point}\n")

    append_normalization_to_header(mean, std)
    append_gesture_enum_to_header(gestures)


    with open(HEADER_OUTPUT_PATH, "a") as f:
        f.write("#endif //MODELDATA_H")
