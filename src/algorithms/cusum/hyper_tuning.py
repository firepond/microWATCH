# hyper paramerter tuning for CUSUM algorithm

import json
import multiprocessing
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import numpy as np

import accuracy
from cusum import CusumMeanDetector


def scoring_function(params):
    data = params["data"]
    detector = CusumMeanDetector(t_warmup=params["t_warmup"], p_limit=params["p_limit"])
    locations = detector.detect(data)
    dataset_name = params["dataset_name"]
    f1, cover = accuracy.scores(locations, dataset_name, data.shape[0])
    # if f1 or cover is not double, set it to 0
    if not isinstance(f1, float):
        f1 = 0
    if not isinstance(cover, float):
        cover = 0
    return f1, cover


def f(params):
    f1, cover = scoring_function(params)
    scoring_metric = params["scoring_metric"]
    if scoring_metric == "f1":
        return {"loss": -1 * f1, "status": STATUS_OK}
    elif scoring_metric == "cover":
        return {"loss": -1 * cover, "status": STATUS_OK}
    else:
        return {"loss": -1 * (f1 + cover), "status": STATUS_OK}


def search(joined_list):
    data_path, scoring_metric = joined_list
    dataset_name = data_path.split("/")[-1].split(".")[0]

    data = np.loadtxt(data_path, delimiter=",")

    # threshold tuning
    param_space = {
        "t_warmup": hp.uniform("t_warmup", 10, 300),
        "p_limit": hp.uniform("p_limit", 0.0001, 1),
        "data": data,
        "dataset_name": dataset_name,
        "scoring_metric": scoring_metric,
    }
    trials = Trials()
    results = None
    loss_threshold = -0.99
    if scoring_metric == "sum":
        loss_threshold = -1.99

    results = fmin(
        f,
        param_space,
        algo=tpe.suggest,
        max_evals=1000,
        trials=trials,
        loss_threshold=loss_threshold,
    )

    # running best model
    detector = CusumMeanDetector(
        t_warmup=results["t_warmup"], p_limit=results["p_limit"]
    )
    # debug information

    locations = detector.detect(data)
    f1, cover = accuracy.scores(locations, dataset_name, data.shape[0])

    twarmup = results["t_warmup"]
    plimit = results["p_limit"]
    print(
        f"Best model for {dataset_name} with f1: {f1} and cover: {cover}, twarmup: {twarmup}, plimit: {plimit}"
    )

    return [
        dataset_name,
        twarmup,
        plimit,
        f1,
        cover,
    ]


def main():
    # Directory containing the files
    directory = "../../../datasets/csv"

    # List all files in the directory
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    # read all files with numpy, only keep the univariate time series
    files = [file for file in files if len(np.loadtxt(file, delimiter=",").shape) == 1]

    global annotations
    with open("./annotations.json", "r") as fp:
        annotations = json.load(fp)

    # Number of processes to use (e.g., number of CPU cores)
    num_processes = multiprocessing.cpu_count() - 4
    # num_processes = 1
    # sort file by name
    files.sort()

    # search for best f1
    # joined_list of files and distance indexes
    joined_list = [(file, "f1") for file in files]
    # map processes to the joined list, then save the results
    with open(f"cusum_f1_results.csv", "w") as fp:
        fp.write("file, twarmup, plimit, f1, cover\n")
    best_results = multiprocessing.Pool(num_processes).map(search, joined_list)
    for result in best_results:
        file = result[0]
        with open(f"cusum_f1_results.csv", "a") as fp:
            # fp.write("file, twarmup, plimit, f1, cover\n")
            fp.write(f"{file}, {result[0]}, {result[1]}, {result[2]}, {result[3]}\n")

    # search for best cover
    # joined_list of files and distance indexes
    joined_list = [(file, "sum") for file in files]
    # map processes to the joined list, then save the results
    with open(f"cusum_sum_results.csv", "w") as fp:
        fp.write("file, twarmup, plimit, f1, cover\n")
    best_results = multiprocessing.Pool(num_processes).map(search, joined_list)
    for result in best_results:
        file = result[0]
        with open(f"cusum_sum_results.csv", "a") as fp:
            # fp.write("file, twarmup, plimit, f1, cover\n")
            fp.write(f"{file}, {result[0]}, {result[1]}, {result[2]}, {result[3]}\n")


if __name__ == "__main__":
    main()
