# this runs on device with serial port, parse message from the host and respond to it

# protocol specification:
# H\n : hello message, send back hello and board name and programming language

# C\n : ask for available commands, send back available commands

# M+ModelName\n : set model name, send back model name
# M\n : ask for model name, send back model name

# D+DataName\n : set data name, send back data name
# D\n : ask for data name, send back data name

# S\n : start model, sned banck S when ready to start

# F\n : for device to send this when model is done only, not for host to send

# E\n Error, for device to send this when there is an error
# ">": device ready to receive command

# "R": reset the device, not implemented yet, TODO: implement this

# "A": if device is ready to receive command, send back ">"

# with this code from main.py:

import gc
import time

try:
    from ulab import numpy as np
except:
    import numpy as np


def get_data(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    gc.collect()
    return data


def get_params(file_path, distance_index):
    # get params from a csv file

    open_file = open("best_params.csv", "r")
    data = []
    # read line by line, not enough ram to read all at once

    dataset_name = file_path.split("/")[-1].split(".")[0]
    # print(f"Dataset name: {dataset_name}")
    for line in open_file.readlines():
        line = line.strip().split(",")
        if line[0] == dataset_name and int(line[1]) == distance_index:
            data = line
    open_file.close()

    gc.collect()
    # print(f"Params: {data}")
    return data


model_name = ""
model_set = False

dataset_path = ""
data = []
params = []
model_set = False
data_set = False

start = False


while True:
    print(">")
    # constantly check for incoming message
    command = input()
    if command == "":
        continue
    if command == "H":
        print("hello\nboard: ESP32C6 (RISC-V)\nlanguage: micropython")
    elif command == "C":
        print("H\nA\nM\nD\nS")
    elif command[0] == "M":
        if len(command) > 2 and command[1] == "+":
            model_name = command[2:]
            # if model_name has WATCH and a int, then it is WATCH model

            if "WATCH" in model_name:
                # parse the int from the model name
                try:
                    distance_index = int(model_name.split("WATCH")[-1])
                except:
                    print("E")
                    continue

                import micro_watch as microwatch

                watch = microwatch.microWATCH()
                watch.metric = microwatch.distance_measures[distance_index]
                model_set = True
                print("M:" + model_name)
                model_name = "WATCH"
        else:
            print("model")
    elif command[0] == "D":
        if model_set == False:
            print("E")
            continue
        if len(command) > 2 and command[1] == "+":
            dataset_path = command[2:]
            # load data
            try:
                data = get_data(dataset_path)
            except:
                print("load data error")
                print("E")
                continue
            # load params
            try:
                # params = get_params(dataset_path, distance_index)
                watch.set_params(distance_index, dataset_path)
            except:
                print("load params error")
                print("E")
                continue
            data_set = True
            print("D:" + dataset_path)
        elif data_set == True:
            print("D:" + dataset_path)
        else:
            print("No data")
    elif command == "S":
        if model_set == False or data_set == False:
            print("E")
        elif model_name == "WATCH":
            print("S")
            # 10 time average
            start_time = time.ticks_ms()
            for i in range(10):
                watch.detect(data)
                watch.reinit()
            end_time = time.ticks_ms()
            print("F")
            print(f"Time: {(end_time-start_time)/10} ms")
        else:
            print("E")
    elif command != "A":
        print("E")
    # print(">")

