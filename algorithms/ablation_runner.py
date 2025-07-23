import math
import numpy as np
import time
import os
import platform  # For path determination
import distance
from micro_watch import distance_measures
import csv


# --- Stage 0: Original WATCH ---
# Assuming watch_origin.py is in algorithms/ and rpy2 is importable
try:
    from watch_origin import WATCH as WATCH_Stage0
except ImportError as e:
    print(
        f"Note: Could not import WATCH_Stage0 (original): {e}. Stage 0 will be skipped."
    )
    WATCH_Stage0 = None  # Placeholder


def distance_function_mean(sample, dist_mean, dist_metric):
    dist_mean_list = np.array([dist_mean] * sample.shape[0])
    distance = dist_metric(sample, dist_mean_list)
    return distance


from watch_stage_1 import WATCH as WATCH_Stage1

from watch_stage_2 import WATCH as WATCH_Stage2

from watch_stage_3 import WATCH as WATCH_Stage3


# --- Helper function to parse parameters ---
def get_params_from_csv(params_path, dataset_filename, target_distance_index=6):
    """
    Loads parameters for a given dataset and distance index from the CSV.
    Returns a dictionary of parameters or None if not found.
    """
    try:
        with open(params_path, "r") as f:
            for line in f:
                # line format: file_name,distance_index,batch_size,threshold,max_dist_size,new_dist_buffer_size
                parts = line.strip().split(",")
                if len(parts) == 6:
                    file_name_csv, dist_idx_csv, bs_csv, thr_csv, mds_csv, ndbs_csv = (
                        parts
                    )
                    if (
                        file_name_csv == dataset_filename
                        and int(dist_idx_csv) == target_distance_index
                    ):
                        return {
                            "batch_size": int(bs_csv),
                            "threshold_ratio": float(
                                thr_csv
                            ),  # 'threshold' in CSV is threshold_ratio
                            "max_dist_size": int(mds_csv),
                            "new_dist_buffer_size": int(ndbs_csv),
                        }
    except FileNotFoundError:
        print(f"Parameter file {params_path} not found.")
    except Exception as e:
        print(f"Error reading parameter file {params_path}: {e}")
    return None


# --- Main execution for ablation ---
def run_ablation_experiment():

    dataset_target_list = [
        "apple.csv",
        "occupancy.csv",
        "run_log.csv",
        "bee_waggle_6.csv",
        "well_log.csv",
        "bitcoin.csv",
        "us_population.csv",
        "measles.csv",
    ]

    csv_folder = "../datasets/csv"
    params_file_path = "../params/params_watch_best.csv"

    print(f"Using CSV folder: {os.path.abspath(csv_folder)}")
    print(f"Using params file: {os.path.abspath(params_file_path)}")

    if not os.path.isdir(csv_folder):
        print(f"Error: CSV folder not found at {os.path.abspath(csv_folder)}")
        return

    dataset_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
    dataset_files = dataset_target_list
    if not dataset_files:
        print(f"No CSV files found in {csv_folder}")
        return

    all_results = {}

    # Default parameters if not found in CSV
    default_params = {
        "threshold_ratio": 3.0,
        "new_dist_buffer_size": 5,
        "batch_size": 3,
        "max_dist_size": 10,  # General max_dist_size
    }
    # Target distance index for Stages 1-3 (Euclidean in micro_watch is 10)
    TARGET_DIST_IDX_FOR_STAGES_1_3 = 6

    for dataset_file in dataset_files:
        dataset_name_no_ext = dataset_file.split(".")[0]
        full_data_path = os.path.join(csv_folder, dataset_file)
        print(f"\n\n--- Processing Dataset: {dataset_file} ---")

        try:
            data = np.loadtxt(full_data_path, delimiter=",")
            if data.ndim == 1:  # Ensure 2D
                data = data.reshape(-1, 1)
            if data.shape[0] == 0:
                print(f"Skipping empty dataset: {dataset_file}")
                continue
            data_width = data.shape[1]
        except Exception as e:
            print(f"Error loading dataset {dataset_file}: {e}")
            continue

        # Try to load specific parameters for this dataset
        # For Stages 1, 2, 3, we try to get params for Euclidean-like behavior
        specific_params_s123 = get_params_from_csv(
            params_file_path, dataset_name_no_ext, TARGET_DIST_IDX_FOR_STAGES_1_3
        )

        # For Stage 0 (Original WATCH), it has different param names and behavior for max_dist_size.
        # We can try to get params for it too, or use defaults.
        # Let's assume we try to get params for a common/first distance index if not specifically for Wasserstein.
        # Or, more simply, adapt the default_params or specific_params_s123.
        # For simplicity, Stage 0 will also try to use adapted params if available, else defaults.

        current_params = default_params.copy()
        if specific_params_s123:
            current_params.update(specific_params_s123)
            print(
                f"Loaded specific parameters for {dataset_name_no_ext} (dist_idx {TARGET_DIST_IDX_FOR_STAGES_1_3}): {specific_params_s123}"
            )
        else:
            print(f"Using default parameters for {dataset_name_no_ext} for Stages 1-3.")

        # Ensure max_dist_size for Stage 3 (buffer capacity) is sensible
        # This logic is from the original ablation runner, adapt as needed.
        # Stage 3's max_dist_size is its buffer capacity.
        stage3_max_dist_size = current_params["max_dist_size"]
        if stage3_max_dist_size <= 0:
            stage3_max_dist_size = current_params["new_dist_buffer_size"] * 2
        if stage3_max_dist_size == 0:
            stage3_max_dist_size = 10  # Absolute fallback
        if stage3_max_dist_size < current_params["new_dist_buffer_size"]:
            stage3_max_dist_size = current_params["new_dist_buffer_size"]

        dataset_results = {}
        num_runs = 100
        # Stage 0: Original WATCH
        print("\n--- Running Stage 0: Original WATCH ---")
        if WATCH_Stage0:
            try:
                # Original WATCH 'threshold' is like 'threshold_ratio'.
                # 'max_dist_size=0' means unlimited list. If CSV provides one, use it, else 0.
                s0_max_dist_size = (
                    current_params["max_dist_size"]
                    if current_params["max_dist_size"] > 0
                    else 0
                )

                watch_s0 = WATCH_Stage0(
                    threshold_ratio=current_params["threshold_ratio"],
                    new_dist_buffer_size=current_params["new_dist_buffer_size"],
                    batch_size=current_params["batch_size"],
                    max_dist_size=s0_max_dist_size,
                )

                total_time_s0 = 0
                locations_s0 = []

                for _ in range(num_runs):
                    start_time_run = time.time()
                    current_locations_s0 = watch_s0.detect(data.copy())
                    end_time_run = time.time()
                    total_time_s0 += end_time_run - start_time_run
                    if hasattr(watch_s0, "reinit"):
                        watch_s0.reinit()  # Reset for the next run
                    if _ == num_runs - 1:  # Keep locations from the last run
                        locations_s0 = current_locations_s0

                avg_time_s0 = total_time_s0 / num_runs

                dataset_results["Stage0"] = {
                    "locations": locations_s0,
                    "time": avg_time_s0,
                }
                print(
                    f"Locations: {locations_s0}, Avg Time ({num_runs} runs): {dataset_results['Stage0']['time']:.6f}s"
                )
            except Exception as e:
                print(f"Could not run Stage 0 for {dataset_file}: {e}")
                dataset_results["Stage0"] = {"locations": "Error", "time": "N/A"}
        else:
            print("Stage 0 skipped due to import error.")
            dataset_results["Stage0"] = {"locations": "Skipped", "time": "N/A"}

        # Stage 1
        print(
            "\n--- Running Stage 1: Python distance (batch mean vs. full dist mean), list-based dist ---"
        )
        try:
            watch_s1 = WATCH_Stage1(
                threshold_ratio=current_params["threshold_ratio"],
                new_dist_buffer_size=current_params["new_dist_buffer_size"],
                batch_size=current_params["batch_size"],
                max_dist_size=current_params["max_dist_size"],  # Cap for list
                version=TARGET_DIST_IDX_FOR_STAGES_1_3,
            )

            total_time_s1 = 0
            locations_s1 = []  # Store locations from the last run, or handle as needed

            for _ in range(num_runs):
                start_time_run = time.time()
                # Re-running detect will reset its internal state as per its implementation
                current_locations_s1 = watch_s1.detect(data.copy())
                end_time_run = time.time()
                total_time_s1 += end_time_run - start_time_run
                watch_s1.reinit()  # Reset for the next run
                if _ == num_runs - 1:  # Keep locations from the last run
                    locations_s1 = current_locations_s1

            avg_time_s1 = total_time_s1 / num_runs

            dataset_results["Stage1"] = {"locations": locations_s1, "time": avg_time_s1}
            print(
                f"Locations: {locations_s1}, Avg Time ({num_runs} runs): {dataset_results['Stage1']['time']:.6f}s"
            )
        except Exception as e:
            print(f"Could not run Stage 1 for {dataset_file}: {e}")
            dataset_results["Stage1"] = {"locations": "Error", "time": "N/A"}

        # Stage 2
        print(
            "\n--- Running Stage 2: Python distance (batch mean vs. pre-calculated dist mean), list-based points ---"
        )
        try:
            watch_s2 = WATCH_Stage2(
                threshold_ratio=current_params["threshold_ratio"],
                new_dist_buffer_size=current_params["new_dist_buffer_size"],
                batch_size=current_params["batch_size"],
                max_dist_size=current_params["max_dist_size"],  # Cap for list
                version=TARGET_DIST_IDX_FOR_STAGES_1_3,
            )

            total_time_s2 = 0
            locations_s2 = []

            for _ in range(num_runs):
                start_time_run = time.time()
                current_locations_s2 = watch_s2.detect(data.copy())
                end_time_run = time.time()
                total_time_s2 += end_time_run - start_time_run
                watch_s2.reinit()  # Reset for the next run
                if _ == num_runs - 1:  # Keep locations from the last run
                    locations_s2 = current_locations_s2

            avg_time_s2 = total_time_s2 / num_runs

            dataset_results["Stage2"] = {"locations": locations_s2, "time": avg_time_s2}
            print(
                f"Locations: {locations_s2}, Avg Time ({num_runs} runs): {dataset_results['Stage2']['time']:.6f}s"
            )
        except Exception as e:
            print(f"Could not run Stage 2 for {dataset_file}: {e}")
            dataset_results["Stage2"] = {"locations": "Error", "time": "N/A"}

        # Stage 3
        print("\n--- Running Stage 3: Pre-allocated memory for distribution ---")
        try:
            watch_s3 = WATCH_Stage3(
                threshold_ratio=current_params["threshold_ratio"],
                new_dist_buffer_size=current_params["new_dist_buffer_size"],
                batch_size=current_params["batch_size"],
                max_dist_size=stage3_max_dist_size,  # Buffer capacity
                version=TARGET_DIST_IDX_FOR_STAGES_1_3,
            )

            total_time_s3 = 0
            locations_s3 = []

            for _ in range(num_runs):
                start_time_run = time.time()
                current_locations_s3 = watch_s3.detect(data.copy())
                end_time_run = time.time()
                total_time_s3 += end_time_run - start_time_run
                watch_s3.reinit()  # Reset for the next run
                if _ == num_runs - 1:  # Keep locations from the last run
                    locations_s3 = current_locations_s3

            avg_time_s3 = total_time_s3 / num_runs

            dataset_results["Stage3"] = {"locations": locations_s3, "time": avg_time_s3}
            print(
                f"Locations: {locations_s3}, Avg Time ({num_runs} runs): {dataset_results['Stage3']['time']:.6f}s"
            )
        except Exception as e:
            print(f"Could not run Stage 3 for {dataset_file}: {e}")
            dataset_results["Stage3"] = {"locations": "Error", "time": "N/A"}

        all_results[dataset_file] = dataset_results

    print("\n\n--- Ablation Summary ---")
    for dataset_name, results_data in all_results.items():
        print(f"\nDataset: {dataset_name}")
        for stage_name, res_val in results_data.items():
            time_val = res_val["time"]
            if isinstance(time_val, float):
                time_str = f"{time_val:.6f}s"
            else:
                time_str = time_val
            print(f"  {stage_name}: Time={time_str}, Locations={res_val['locations']}")

    # 添加分析功能
    analyze_ablation_results(all_results, "ablation_speedup_analysis.csv")


def analyze_ablation_results(all_results, output_csv_path="ablation_analysis.csv"):
    """
    分析消融研究结果，计算每个阶段之间的加速因子
    """
    print("\n--- 消融研究分析 ---")

    # 存储每个数据集的加速因子
    dataset_speedups = {}

    # 阶段转换定义
    stage_transitions = [
        ("Stage0", "Stage1", "Stage0->Stage1"),
        ("Stage1", "Stage2", "Stage1->Stage2"),
        ("Stage2", "Stage3", "Stage2->Stage3"),
        ("Stage0", "Stage3", "Stage0->Stage3"),  # 总体加速
    ]

    # 计算每个数据集的加速因子
    for dataset_name, results in all_results.items():
        print(f"\n数据集: {dataset_name}")
        dataset_speedups[dataset_name] = {}

        for from_stage, to_stage, transition_name in stage_transitions:
            from_time = results.get(from_stage, {}).get("time", "N/A")
            to_time = results.get(to_stage, {}).get("time", "N/A")

            # 检查时间是否为有效数值
            if (
                isinstance(from_time, (int, float))
                and isinstance(to_time, (int, float))
                and from_time > 0
                and to_time > 0
            ):
                speedup = from_time / to_time
                dataset_speedups[dataset_name][transition_name] = speedup
                print(
                    f"  {transition_name}: {speedup:.2f}x ({from_time:.6f}s -> {to_time:.6f}s)"
                )
            else:
                dataset_speedups[dataset_name][transition_name] = "N/A"
                print(f"  {transition_name}: N/A (无效时间数据)")

    # 计算平均加速因子
    print("\n--- 平均加速因子 ---")
    average_speedups = {}

    for _, _, transition_name in stage_transitions:
        valid_speedups = []
        for dataset_name in dataset_speedups:
            speedup = dataset_speedups[dataset_name].get(transition_name, "N/A")
            if isinstance(speedup, (int, float)) and speedup > 0:
                valid_speedups.append(speedup)

        if valid_speedups:
            avg_speedup = sum(valid_speedups) / len(valid_speedups)
            average_speedups[transition_name] = avg_speedup
            print(
                f"{transition_name}: {avg_speedup:.2f}x (基于 {len(valid_speedups)} 个有效数据集)"
            )
        else:
            average_speedups[transition_name] = "N/A"
            print(f"{transition_name}: N/A (无有效数据)")

    # 保存到CSV文件
    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # 写入表头
            header = ["Dataset"] + [
                transition_name for _, _, transition_name in stage_transitions
            ]
            writer.writerow(header)

            # 写入每个数据集的加速因子
            for dataset_name in dataset_speedups:
                row = [dataset_name]
                for _, _, transition_name in stage_transitions:
                    speedup = dataset_speedups[dataset_name].get(transition_name, "N/A")
                    if isinstance(speedup, (int, float)):
                        row.append(f"{speedup:.4f}")
                    else:
                        row.append(speedup)
                writer.writerow(row)

            # 写入平均值行
            avg_row = ["Average"]
            for _, _, transition_name in stage_transitions:
                avg_speedup = average_speedups.get(transition_name, "N/A")
                if isinstance(avg_speedup, (int, float)):
                    avg_row.append(f"{avg_speedup:.4f}")
                else:
                    avg_row.append(avg_speedup)
            writer.writerow(avg_row)

            # 添加空行和时间详情
            writer.writerow([])
            writer.writerow(["=== 详细时间信息 ==="])
            writer.writerow(
                ["Dataset", "Stage0_Time", "Stage1_Time", "Stage2_Time", "Stage3_Time"]
            )

            for dataset_name, results in all_results.items():
                time_row = [dataset_name]
                for stage in ["Stage0", "Stage1", "Stage2", "Stage3"]:
                    time_val = results.get(stage, {}).get("time", "N/A")
                    if isinstance(time_val, (int, float)):
                        time_row.append(f"{time_val:.6f}")
                    else:
                        time_row.append(time_val)
                writer.writerow(time_row)

        print(f"\n分析结果已保存到: {os.path.abspath(output_csv_path)}")

    except Exception as e:
        print(f"保存CSV文件时出错: {e}")

    return dataset_speedups, average_speedups


if __name__ == "__main__":
    run_ablation_experiment()
