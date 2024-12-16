#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper for RBOCPDMS in CPDBench.

Author: G.J.J. van den Burg
Date: 2019-10-03
License: See the LICENSE file.
Copyright: 2019, The Alan Turing Institute

"""

bocpd_intensities = [10, 50, 100, 200]
# bocpd_intensities = [10]
bocpd_prior_a = [0.01, 0.1, 1.0, 10, 100]
# bocpd_prior_a = [0.01, 1]
bocpd_prior_b = [0.01, 0.1, 1.0, 10, 100]
# bocpd_prior_b = [0.01]


threshold = 100

import copy
from itertools import product
import json
import multiprocessing
import os
import numpy as np
import time

import accuracy
from cp_probability_model import CpModel
from BVAR_NIG_DPD import BVARNIGDPD

from detector import Detector

from multiprocessing import Process, Manager
import concurrent.futures


def load_dataset(filename):
    """Load a CPDBench dataset"""
    with open(filename, "r") as fp:
        data = json.load(fp)

    if data["time"]["index"] != list(range(0, data["n_obs"])):
        raise NotImplementedError(
            "Time series with non-consecutive time axis are not yet supported."
        )

    mat = np.zeros((data["n_obs"], data["n_dim"]))
    for j, series in enumerate(data["series"]):
        mat[:, j] = series["raw"]

    # We normalize to avoid numerical errors.
    mat = (mat - np.nanmean(mat, axis=0)) / np.sqrt(np.nanvar(mat, axis=0, ddof=1))

    return data, mat


def run_rbocpdms(mat, params):
    """Set up and run RBOCPDMS"""
    S1 = params["S1"]
    S2 = params["S2"]

    # we use "DPD" from the well log example, as that seems to be the robust
    # version.
    model_universe = [
        BVARNIGDPD(
            prior_a=params["prior_a"],
            prior_b=params["prior_b"],
            S1=S1,
            S2=S2,
            alpha_param=params["alpha_param"],
            prior_mean_beta=params["prior_mean_beta"],
            prior_var_beta=params["prior_var_beta"],
            prior_mean_scale=params["prior_mean_scale"],
            prior_var_scale=params["prior_var_scale"],
            general_nbh_sequence=[[[]]] * S1 * S2,
            general_nbh_restriction_sequence=[[0]],
            general_nbh_coupling="weak coupling",
            hyperparameter_optimization="online",
            VB_window_size=params["VB_window_size"],
            full_opt_thinning=params["full_opt_thinning"],
            SGD_batch_size=params["SGD_batch_size"],
            anchor_batch_size_SCSG=params["anchor_batch_size_SCSG"],
            anchor_batch_size_SVRG=None,
            first_full_opt=params["first_full_opt"],
        )
    ]

    model_universe = np.array(model_universe)
    model_prior = np.array([1 / len(model_universe)] * len(model_universe))

    cp_model = CpModel(params["intensity"])

    detector = Detector(
        data=mat,
        model_universe=model_universe,
        model_prior=model_prior,
        cp_model=cp_model,
        S1=params["S1"],
        S2=params["S2"],
        T=mat.shape[0],
        store_rl=True,
        store_mrl=True,
        trim_type="keep_K",
        threshold=params["threshold"],
        save_performance_indicators=True,
        generalized_bayes_rld=params["rld_DPD"],
        alpha_param_learning="individual",
        alpha_param=params["alpha_param"],
        alpha_param_opt_t=100,
        alpha_rld=params["alpha_rld"],
        alpha_rld_learning=True,
        loss_der_rld_learning=params["loss_der_rld_learning"],
    )
    detector.run()

    return detector


def detect(file_name, intensity, prior_a, prior_b, alpha_param=0.5, alpha_rld=0.5):

    # # input_list  = file_name, intensity, prior_a, prior_b, index
    # file_name, intensity, prior_a, prior_b, alpha_param, alpha_rld = input_list

    data, mat = load_dataset(file_name)

    args = {}
    args["intensity"] = intensity
    args["prior_a"] = prior_a
    args["prior_b"] = prior_b
    args["threshold"] = threshold
    args["alpha_param"] = alpha_param
    args["alpha_rld"] = alpha_rld

    # setting S1 as dimensionality follows the 30portfolio_ICML18.py script.
    # other settings mostly taken from the well log example
    defaults = {
        "S1": mat.shape[1],
        "S2": 1,
        "SGD_batch_size": 10,
        "VB_window_size": 360,
        "anchor_batch_size_SCSG": 25,
        "first_full_opt": 10,
        "full_opt_thinning": 20,
        "intercept_grouping": None,
        "loss_der_rld_learning": "absolute_loss",
        "prior_mean_beta": None,
        "prior_mean_scale": 0,  # data has been standardized
        "prior_var_beta": None,
        "prior_var_scale": 1.0,  # data has been standardized
        "rld_DPD": "power_divergence",  # this ensures doubly robust
    }

    # join two dictionaries: defaults and args
    parameters = copy.deepcopy(defaults)
    parameters.update(args)
    print(parameters)

    start_time = time.time()

    detector = run_rbocpdms(mat, parameters)

    stop_time = time.time()
    runtime = stop_time - start_time

    # According to the Nile unit test, the MAP change points are in
    # detector.CPs[-2], with time indices in the first of the two-element
    # vectors.
    locations = [x[0] for x in detector.CPs[-2]]

    # Based on the fact that time_range in plot_raw_TS of the EvaluationTool
    # starts from 1 and the fact that CP_loc that same function is ensured to
    # be in time_range, we assert that the change point locations are 1-based.
    # We want 0-based, so subtract 1 from each point.
    locations = [loc - 1 for loc in locations]

    # convert to Python ints
    locations = [int(loc) for loc in locations]

    return locations, mat.shape[0]


def detect_and_score(file, intensity, prior_a, prior_b):

    locations, length = detect(file, intensity, prior_a, prior_b)
    dataset_name = file.split("/")[-1].split(".")[0]
    f1, cover = accuracy.scores(locations, dataset_name, length)
    score = f1 + cover
    return score, intensity, prior_a, prior_b


def main():

    # Directory containing the files
    directory = "/home/campus.ncl.ac.uk/c4060464/esp32/microWATCH/datasets/json"

    # with open("best_params_rbocpdms.txt", "w") as f:
    # f.write("file, intensity, prior_a, prior_b\n")
    results_file = "best_params_rbocpdms.txt"
    # read existing results, then check if the file has already been processed, if so, skip it
    processed_files = []
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            lines = f.readlines()
            # skip the header
            for line in lines[1:]:
                dataset_name = line.split(",")[0].strip().split("/")[-1].split(".")[0]
                processed_files.append(dataset_name)

    # List all files in the directory
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    for file in files:
        # skip if  in processed files
        cur_dataset_name = file.split("/")[-1].split(".")[0]
        if cur_dataset_name in processed_files:
            print(f"Skipping {file}")
            continue
        print(f"Processing {file}")
        # try all combinations of parameters, save the best one
        best_params = []
        best_score = -1
        bocpd_alpha_param = [0.5]
        bocpd_alpha_rld = [0.5]
        # 准备所有参数组合
        param_combinations = list(
            product(
                bocpd_intensities,
                bocpd_prior_a,
                bocpd_prior_b,
            )
        )

        # 并行执行检测和评分
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 提交所有组合的任务
            futures = {
                executor.submit(detect_and_score, file, intensity, prior_a, prior_b): (
                    intensity,
                    prior_a,
                    prior_b,
                )
                for (intensity, prior_a, prior_b) in param_combinations
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    score, intensity, prior_a, prior_b = future.result()
                    if score > best_score:
                        best_score = score
                        best_params = [intensity, prior_a, prior_b]

                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")

        # 将最佳参数写入文件
        print(f"Best parameters for {file}: {best_params}")
        with open("best_params_rbocpdms.txt", "a") as f:
            if len(best_params) == 3:
                intensity = best_params[0]
                prior_a = best_params[1]
                prior_b = best_params[2]

                f.write(f"{file}, {intensity}, {prior_a}, {prior_b}\n")
            else:
                f.write(f"{file}, None, None, None\n")


if __name__ == "__main__":
    main()
