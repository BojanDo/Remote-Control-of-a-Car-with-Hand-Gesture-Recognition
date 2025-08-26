import subprocess
import fileinput
import glob
from config import HEADER_OUTPUT_PATH, GESTURES, NUM_MODELS


def add_models(outfile):
    h_files = sorted(glob.glob("generated/model_data_*.h"))

    for h_file in h_files:
        with open(h_file, "r") as infile:
            outfile.write(infile.read())

    outfile.write(f"\n#define NUM_MODELS {NUM_MODELS}\n")

    outfile.write("""\ninline const unsigned char* tflite_models[NUM_MODELS] = {\n""");
    for i in range(NUM_MODELS -1):
        outfile.write(f"    gesture_model_{i}_tflite,\n")
    outfile.write(f"    gesture_model_{NUM_MODELS -1}_tflite\n")
    outfile.write("""};
""")
    outfile.write(f"\n#define GESTURE_CLASSES {len(GESTURES)}\n")


def add_quantization(outfile, scale, zero_point):
    outfile.write("\n// Quantization parameters for each model\n")
    outfile.write(f"#define SCALE {scale:.17g}f\n")
    outfile.write(f"#define ZERO_POINT {zero_point}\n")


def add_standardization(outfile, mean, std):
    outfile.write("\n// IMU standardization parameters\n")
    outfile.write(f"#define IMU_FEATURES {len(mean)}\n\n")
    outfile.write(f"inline const float imu_mean[IMU_FEATURES] = {{{', '.join(f'{x:.6f}' for x in mean)}}};\n")
    outfile.write(f"inline const float imu_std[IMU_FEATURES] = {{{', '.join(f'{x:.6f}' for x in std)}}};\n")


def add_gesture_enum(outfile):
    outfile.write("\n// Gesture enumeration based on model output indices\n")
    outfile.write("typedef enum {\n")
    for i, gesture in enumerate(GESTURES):
        outfile.write(f"    {gesture} = {i},\n")
    outfile.write("} Gesture;\n")
    outfile.write("\nstatic inline Gesture intToGesture(int value) {\n")
    outfile.write("    switch (value) {\n")
    for i, gesture in enumerate(GESTURES):
        outfile.write(f"        case {i}: return {gesture};\n")
    outfile.write("        default: return none;\n")
    outfile.write("    }\n}\n")


def created_header_from_models(tflite_models, scale, zero_point, mean, std):
    for i, tflite_model in enumerate(tflite_models):
        tflite_mode_path = f"generated/gesture_model_{i}.tflite"
        output_path = f"generated/model_data_{i}.h"

        with open(tflite_mode_path, "wb") as f:
            f.write(tflite_model)

        subprocess.run(["xxd", "-i", tflite_mode_path, output_path], check=True)

        for line in fileinput.input(output_path, inplace=True):
            if fileinput.filelineno() == 1:
                print(f"\ninline const unsigned char gesture_model_{i}_tflite[] = {{")
            elif f"unsigned int generated_gesture_model_{i}_tflite_len" in line:
                print(line.replace(f"unsigned int generated_gesture_model_{i}_tflite_len",
                                   f"inline const unsigned int gesture_model_{i}_tflite_len"), end="")
            else:
                print(line, end="")

    with open(HEADER_OUTPUT_PATH, "w") as outfile:
        outfile.write("""#ifndef MODELDATA_H
#define MODELDATA_H""")

        add_models(outfile)
        add_quantization(outfile, scale, zero_point)
        add_standardization(outfile, mean, std)
        add_gesture_enum(outfile)

        outfile.write("#endif //MODELDATA_H")

    print(f"All model headers combined into {HEADER_OUTPUT_PATH}")
