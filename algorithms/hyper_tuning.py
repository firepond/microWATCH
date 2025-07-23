# hyperparameter tuning for the model

# comapre 3 algorithms: 1. watch_origin.py, watch_stage_1.py, watch_stage_3.py

# compare f1 score and cover on 42 different datasets in "datasets/csv" folder, each dataset is in .csv format

# for watch_stage_1.py and watch_stage_3.py, use chebyshev_min as distance metric, which has been proven to be the best distance metric for these algorithms
#         distance_metric=distance.chebyshev_min,

# tune the following hyperparameters:
#         threshold_ratio=0.51(must be > 0),
#         new_dist_buffer_size=32(must be > 0),
#         batch_size=3(must be > 0),
#         max_dist_size=72(must be > 0)

# save the results in "results/hyper_tuning" folder, each result is in .csv format

# f1 and cover are calculated using the scores function in algorithms/accuracy.py
# use it the following way:  cores(locations, dataset_name, n_obs, annotations_file="../annotations.json"):
# where: locations is the list of locations found by the algorithm
#         dataset_name is the name of the dataset, e.g. "dataset.csv", but without the path or the extension
#         n_obs is the number of samples in the dataset, can be found using the shape of the dataset, e.g. data.shape[0]

# best results are best in terms of sum of f1 and cover, bigger is better


import os
import numpy as np
import pandas as pd
import itertools
import time
import concurrent.futures
import warnings  # For handling warnings from R or other parts

# Import your project's modules
import watch_stage_1
import watch_stage_3
import accuracy
import watch_origin
import watch_stage_list


# --- Configuration Area ---
DATASET_DIR_PATH = "../datasets/csv"
ANNOTATIONS_FILE_PATH = "../annotations.json"
RESULTS_DIR = "../results/hyper_tuning"
# For ProcessPoolExecutor, NUM_WORKERS should ideally be <= os.cpu_count()
NUM_WORKERS = 20


# Define hyperparameter grids
# For watch_stage_1 and watch_stage_3, use chebyshev_min (version=6 assumed)
PARAM_GRID_STAGES = {
    "threshold_ratio": [
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
        1.1,
        1.3,
        1.5,
        1.7,
        1.9,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ],
    "new_dist_buffer_size": [
        3,
        5,
        8,
        10,
        12,
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
    ],  # Wider range for stages
    "batch_size": [3, 5, 7, 10],
    "max_dist_size": [8, 16, 32, 48, 64, 80, 96, 112, 128],  # Wider range for stages
    "version": [6],  # Assuming Chebyshev_min is version 6 for stage1/3
}

# watch_origin does not use 'version' and might have different optimal ranges
PARAM_GRID_ORIGIN = {
    "threshold_ratio": [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
    ],  # Typically higher for Wasserstein
    "new_dist_buffer_size": [3, 5, 8, 10],  # Often smaller for watch_origin
    "batch_size": [3, 5, 7, 10],
    "max_dist_size": [0, 25, 50, 75],  # 0 means infinite for watch_origin
    # No 'version' key here
}


# --- Helper Functions ---
def run_and_score_configuration(
    detector_class,
    params,  # This will be a specific combination from the grid
    data,
    dataset_name_for_scores,
    n_obs,
    annotations_file,
    algorithm_name,
):
    """
    Instantiates detector, runs detection, and calculates F1 and Cover scores.
    This function is designed to be picklable for ProcessPoolExecutor.
    """
    # print(f"PID {os.getpid()}: Running {algorithm_name} with {params} on {dataset_name_for_scores}")
    start_time = time.time()
    current_params_for_detector = params.copy()

    # watch_origin.WATCH __init__ does not take 'version'.
    if detector_class == watch_origin.WATCH:
        if "version" in current_params_for_detector:
            del current_params_for_detector["version"]
    elif (
        "version" not in current_params_for_detector
        and algorithm_name != "WATCH_Origin"
    ):
        # This should ideally not happen if grids are correctly assigned
        warnings.warn(
            f"PID {os.getpid()}: 'version' parameter missing for {algorithm_name} but expected. Params: {params}"
        )
        # Decide on fallback: error out, or use a default. For now, let it try.

    try:
        detector = detector_class(**current_params_for_detector)
    except Exception as e:
        # print(f"PID {os.getpid()}: Error instantiating {algorithm_name} with {current_params_for_detector}: {e}")
        return {
            "params": params,
            "f1_score": np.nan,
            "cover_score": np.nan,
            "combined_score": np.nan,
            "execution_time_s": time.time() - start_time,
            "error": f"Instantiation error: {str(e)}",
            "algorithm_name": algorithm_name,
            "dataset_name": dataset_name_for_scores,
        }

    # reinit might not be necessary if __init__ and detect handle state correctly for multiple calls
    # For process-based parallelism, each run is fresh anyway.
    # if hasattr(detector, "reinit"):
    #     detector.reinit()

    try:
        locations = detector.detect(data.copy())  # data.copy() is good practice
    except Exception as e:
        # print(f"PID {os.getpid()}: Error during {algorithm_name}.detect with {current_params_for_detector} on {dataset_name_for_scores}: {e}")
        return {
            "params": params,
            "f1_score": np.nan,
            "cover_score": np.nan,
            "combined_score": np.nan,
            "execution_time_s": time.time() - start_time,
            "error": f"Detection error: {str(e)}",
            "algorithm_name": algorithm_name,
            "dataset_name": dataset_name_for_scores,
        }
    detection_time = time.time() - start_time

    f1, cover = accuracy.scores(
        locations, dataset_name_for_scores, n_obs, annotations_file=annotations_file
    )

    f1_float = float(f1) if f1 is not None and not np.isnan(f1) else np.nan
    cover_float = float(cover) if cover is not None and not np.isnan(cover) else np.nan

    combined_score = np.nan
    if not np.isnan(f1_float) and not np.isnan(cover_float):
        combined_score = f1_float + cover_float

    # print(f"PID {os.getpid()}: Finished {algorithm_name} with {params} on {dataset_name_for_scores}. Combined: {combined_score}")
    return {
        "params": params,
        "f1_score": f1_float,
        "cover_score": cover_float,
        "combined_score": combined_score,
        "execution_time_s": detection_time,
        "algorithm_name": algorithm_name,
        "dataset_name": dataset_name_for_scores,
    }


def grid_search_for_detector(
    detector_name,
    detector_class,
    param_grid_to_use,  # Specific grid for this detector
    data_to_process,
    dataset_name_for_scores,
    n_observations,
    annotations_file,
):
    print(
        f"\n--- Starting hyperparameter tuning for {detector_name} on {dataset_name_for_scores} ---"
    )
    print(f"Using param grid: {param_grid_to_use}")

    best_combined_score = -np.inf  # Initialize to negative infinity
    best_result_summary = None
    all_run_results = []

    keys, values = zip(*param_grid_to_use.items())
    hyperparameter_combinations = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    total_combinations = len(hyperparameter_combinations)
    if total_combinations == 0:
        print(f"No hyperparameter combinations to test for {detector_name}. Skipping.")
        return [], None

    print(
        f"Total combinations to test for {detector_name}: {total_combinations} using ProcessPoolExecutor with {NUM_WORKERS} workers."
    )

    # Always use ProcessPoolExecutor for rpy2 safety and general parallelism
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_params_map = {
            executor.submit(
                run_and_score_configuration,  # This function must be picklable
                detector_class,
                params_combo,
                data_to_process,  # Ensure data is picklable (numpy arrays are)
                dataset_name_for_scores,
                n_observations,
                annotations_file,
                detector_name,
            ): params_combo
            for params_combo in hyperparameter_combinations
        }

        for i, future in enumerate(
            concurrent.futures.as_completed(future_to_params_map)
        ):
            params_combo_completed = future_to_params_map[future]
            try:
                result = (
                    future.result()
                )  # This can raise exceptions from the child process
                all_run_results.append(result)

                if "error" in result and result["error"]:
                    print(
                        f"  {detector_name} - Combo {i+1}/{total_combinations}: {params_combo_completed} -> ERROR: {result['error']}"
                    )
                elif np.isnan(result["combined_score"]):
                    print(
                        f"  {detector_name} - Combo {i+1}/{total_combinations}: {params_combo_completed} -> "
                        f"F1: {result['f1_score']:.4f}, Cover: {result['cover_score']:.4f}, Combined: NaN (Likely R/logic error)"
                    )
                else:
                    print(
                        f"  {detector_name} - Combo {i+1}/{total_combinations}: {params_combo_completed} -> "
                        f"F1: {result['f1_score']:.4f}, Cover: {result['cover_score']:.4f}, "
                        f"Combined: {result['combined_score']:.4f}"
                    )

                # Check for NaN before comparing, as NaN > anything is False
                if (
                    not np.isnan(result["combined_score"])
                    and result["combined_score"] > best_combined_score
                ):
                    best_combined_score = result["combined_score"]
                    best_result_summary = result.copy()

            except (
                Exception
            ) as exc:  # Errors from executor or future.result() itself (e.g., pickling, process crash)
                print(
                    f"  Critical error processing combo {params_combo_completed} for {detector_name} (Future/Executor level): {exc}"
                )
                all_run_results.append(
                    {
                        "params": params_combo_completed,
                        "f1_score": np.nan,
                        "cover_score": np.nan,
                        "combined_score": np.nan,
                        "execution_time_s": 0.0,
                        "error": f"Executor/Future error: {str(exc)}",
                        "algorithm_name": detector_name,
                        "dataset_name": dataset_name_for_scores,
                    }
                )

    print(f"\n--- {detector_name} tuning on {dataset_name_for_scores} complete ---")
    if best_result_summary and not np.isnan(best_result_summary["combined_score"]):
        print(f"  Best Combined Score: {best_result_summary['combined_score']:.4f}")
        print(
            f"  Best F1: {best_result_summary['f1_score']:.4f}, Best Cover: {best_result_summary['cover_score']:.4f}"
        )
        print(f"  Best Hyperparameters: {best_result_summary['params']}")
    else:
        print(
            f"  No valid (non-NaN) best hyperparameter combination found for {detector_name} on {dataset_name_for_scores}."
        )

    return all_run_results, best_result_summary


def tune_dataset(dataset_path, dataset_name_for_scores):
    """
    Loads a single dataset and tunes all algorithms on it.
    Saves per-dataset summary text file and returns detailed run logs and best params for overall summary.
    """
    print(
        f"\n===== Tuning Dataset: {dataset_name_for_scores} ({os.path.basename(dataset_path)}) ====="
    )
    try:
        # Assuming CSV has no header, adjust if necessary
        data_to_process = np.loadtxt(
            dataset_path, delimiter=",", skiprows=0, dtype=float
        )
        if data_to_process.ndim == 1:
            data_to_process = data_to_process.reshape(-1, 1)
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        return [], []

    n_observations = data_to_process.shape[0]
    if n_observations == 0:
        print(f"Dataset {dataset_path} is empty. Skipping.")
        return [], []

    algorithms_config = [  # Store class and its specific param grid
        ("WATCH_Stage1", watch_stage_1.WATCH, PARAM_GRID_STAGES),
        ("WATCH_Stage3", watch_stage_3.WATCH, PARAM_GRID_STAGES),
        ("WATCH_Stage3_List", watch_stage_list.WATCH, PARAM_GRID_STAGES),
        ("WATCH_Origin", watch_origin.WATCH, PARAM_GRID_ORIGIN),
    ]

    all_runs_for_this_dataset = []
    best_results_for_this_dataset_summary = []

    # Create dataset_summary_text_content here, add to it in the loop
    dataset_summary_text_content = f"Summary for Dataset: {dataset_name_for_scores}\n"
    dataset_summary_text_content += (
        f"Processed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    dataset_summary_text_content += f"Number of observations: {n_observations}\n"
    # Note: PARAM_GRID is not singular anymore, so this line might need adjustment or removal
    # dataset_summary_text_content += f"Base Hyperparameter Grids: STAGES={PARAM_GRID_STAGES}, ORIGIN={PARAM_GRID_ORIGIN}\n"
    dataset_summary_text_content += (
        "---------------------------------------------------\n\n"
    )

    for algo_name, algo_class, algo_param_grid in algorithms_config:
        dataset_summary_text_content += f"Algorithm: {algo_name}\n"
        dataset_summary_text_content += f"  Parameter Grid Used: {algo_param_grid}\n"

        run_logs, best_result = grid_search_for_detector(
            algo_name,
            algo_class,
            algo_param_grid,  # Pass the specific grid
            data_to_process,
            dataset_name_for_scores,
            n_observations,
            ANNOTATIONS_FILE_PATH,
        )
        all_runs_for_this_dataset.extend(run_logs)
        if best_result and not np.isnan(best_result["combined_score"]):
            best_results_for_this_dataset_summary.append(best_result)
            dataset_summary_text_content += (
                f"  Best Combined Score: {best_result['combined_score']:.4f}\n"
            )
            dataset_summary_text_content += (
                f"  Best F1 Score: {best_result['f1_score']:.4f}\n"
            )
            dataset_summary_text_content += (
                f"  Best Cover Score: {best_result['cover_score']:.4f}\n"
            )
            dataset_summary_text_content += (
                f"  Best Parameters: {best_result['params']}\n"
            )
            dataset_summary_text_content += (
                f"  Execution Time for best: {best_result['execution_time_s']:.2f}s\n"
            )
        else:
            dataset_summary_text_content += (
                "  No valid (non-NaN) best parameters found.\n"
            )
        dataset_summary_text_content += "---------------------------------------\n"

    summary_txt_path = os.path.join(
        RESULTS_DIR, f"{dataset_name_for_scores}_summary.txt"
    )
    try:
        with open(summary_txt_path, "w") as f:
            f.write(dataset_summary_text_content)
        print(f"Dataset summary saved to: {summary_txt_path}")
    except Exception as e:
        print(f"Error saving summary text file {summary_txt_path}: {e}")

    return all_runs_for_this_dataset, best_results_for_this_dataset_summary


def main():
    if not os.path.exists(DATASET_DIR_PATH):
        print(f"Error: Dataset directory {DATASET_DIR_PATH} not found.")
        return
    if not os.path.exists(ANNOTATIONS_FILE_PATH):
        print(f"Error: Annotations file {ANNOTATIONS_FILE_PATH} not found.")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Using ProcessPoolExecutor with up to {NUM_WORKERS} worker processes.")

    all_datasets_in_folder = [
        f for f in os.listdir(DATASET_DIR_PATH) if f.endswith(".csv")
    ]

    candidate_datasets_files = [
        "apple.csv",
        "bee_waggle_6.csv",
        "occupancy.csv",
        "run_log.csv",
        "well_log.csv",
        "bitcoin.csv",
        "us_population.csv",
        "measles.csv",
        # Add more datasets if needed, or use all_datasets_in_folder
    ]
    candidate_datasets_files = all_datasets_in_folder
    # datasets_to_process = all_datasets_in_folder # To process all datasets
    datasets_to_process = [
        cf for cf in candidate_datasets_files if cf in all_datasets_in_folder
    ]
    missing_candidates = [
        cf for cf in candidate_datasets_files if cf not in all_datasets_in_folder
    ]

    if not datasets_to_process:
        print(
            "No candidate datasets found in the dataset folder or specified list. Exiting."
        )
        return

    print(f"Found {len(all_datasets_in_folder)} CSV files in dataset directory.")
    print(
        f"Will process {len(datasets_to_process)} specified datasets: {datasets_to_process}"
    )
    if missing_candidates:
        print(
            f"Warning: The following candidate datasets were specified but not found: {missing_candidates}"
        )

    overall_detailed_runs_list = []
    overall_best_summaries_list = []

    for dataset_file_name in datasets_to_process:
        full_dataset_path = os.path.join(DATASET_DIR_PATH, dataset_file_name)
        # Remove .csv for dataset_name_for_scores
        dataset_name_for_scores = os.path.splitext(dataset_file_name)[0]

        detailed_runs, best_summaries = tune_dataset(
            full_dataset_path, dataset_name_for_scores
        )
        overall_detailed_runs_list.extend(detailed_runs)
        overall_best_summaries_list.extend(best_summaries)

        if detailed_runs:
            df_detailed_dataset = pd.DataFrame(detailed_runs)
            # Unpack params dict into separate columns
            if (
                not df_detailed_dataset.empty
                and "params" in df_detailed_dataset.columns
            ):
                # Filter out rows where params might be NaN or not a dict, if any error cases produce that
                valid_params_rows = df_detailed_dataset["params"].apply(
                    lambda x: isinstance(x, dict)
                )
                if valid_params_rows.any():
                    params_df = pd.json_normalize(
                        df_detailed_dataset.loc[valid_params_rows, "params"]
                    )
                    # Align indices for concatenation if only a subset was normalized
                    params_df.index = df_detailed_dataset.loc[valid_params_rows].index
                    df_detailed_dataset = pd.concat(
                        [df_detailed_dataset.drop(columns=["params"]), params_df],
                        axis=1,
                    )
                else:  # No valid params dicts to normalize
                    df_detailed_dataset = df_detailed_dataset.drop(
                        columns=["params"], errors="ignore"
                    )

            detailed_csv_path = os.path.join(
                RESULTS_DIR, f"{dataset_name_for_scores}_all_runs.csv"
            )
            try:
                df_detailed_dataset.to_csv(detailed_csv_path, index=False)
                print(
                    f"Detailed run logs for {dataset_name_for_scores} saved to: {detailed_csv_path}"
                )
            except Exception as e:
                print(f"Error saving detailed CSV {detailed_csv_path}: {e}")
        else:
            print(f"No detailed run logs to save for {dataset_name_for_scores}.")

    if overall_best_summaries_list:
        df_overall_best = pd.DataFrame(overall_best_summaries_list)
        if not df_overall_best.empty and "params" in df_overall_best.columns:
            valid_params_rows_overall = df_overall_best["params"].apply(
                lambda x: isinstance(x, dict)
            )
            if valid_params_rows_overall.any():
                params_df_overall = pd.json_normalize(
                    df_overall_best.loc[valid_params_rows_overall, "params"]
                )
                params_df_overall.index = df_overall_best.loc[
                    valid_params_rows_overall
                ].index
                df_overall_best = pd.concat(
                    [df_overall_best.drop(columns=["params"]), params_df_overall],
                    axis=1,
                )
            else:
                df_overall_best = df_overall_best.drop(
                    columns=["params"], errors="ignore"
                )

        # Define potential parameter columns from all grids
        all_param_keys = set(PARAM_GRID_STAGES.keys()) | set(PARAM_GRID_ORIGIN.keys())

        cols_order_base = [
            "dataset_name",
            "algorithm_name",
            "combined_score",
            "f1_score",
            "cover_score",
        ]
        existing_param_cols_in_df = [
            p for p in all_param_keys if p in df_overall_best.columns
        ]

        # Other columns that might exist (e.g., 'error', 'execution_time_s' if not already handled)
        # 'execution_time_s' is usually part of best_result_summary, so it should be there.
        # 'params' column should have been removed by now.
        other_existing_cols = [
            c
            for c in df_overall_best.columns
            if c not in cols_order_base and c not in existing_param_cols_in_df
        ]

        final_cols_order = (
            cols_order_base
            + sorted(list(existing_param_cols_in_df))
            + sorted(other_existing_cols)
        )

        # Ensure all columns in final_cols_order actually exist in df_overall_best before trying to reindex
        final_cols_order_existing = [
            c for c in final_cols_order if c in df_overall_best.columns
        ]
        df_overall_best = df_overall_best[final_cols_order_existing]

        overall_summary_csv_path = os.path.join(
            RESULTS_DIR, "overall_best_hyperparameters.csv"
        )
        try:
            df_overall_best.to_csv(overall_summary_csv_path, index=False)
            print(
                f"\nOverall best hyperparameters summary saved to: {overall_summary_csv_path}"
            )
        except Exception as e:
            print(f"Error saving overall summary CSV {overall_summary_csv_path}: {e}")
    else:
        print("\nNo best hyperparameter summaries collected to save overall CSV.")

    print("\n===== Hyperparameter Tuning Process Finished =====")


if __name__ == "__main__":
    # This is important for multiprocessing on some platforms (like Windows)
    # It prevents child processes from re-executing the main script's code
    # in a way that leads to infinite recursion of process creation.
    # Should be the first thing in the if __name__ == "__main__": block.
    # multiprocessing.freeze_support() # Uncomment if on Windows or facing related issues
    main()
