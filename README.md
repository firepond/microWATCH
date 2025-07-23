# microCPD
Micro change point detection algorithms for IoT devices.

Codes and experiments results for a research paper: "Energy-Efficient Change Point Detection Algorithm for Resource-Constrained Devices"

## Results

Experiments results can be found in the `results` directory and other specified locations:

1.  **Ablation Study**:
    *   Detailed logs of the ablation study can be found in [results/ablation.txt](results/ablation.txt).
    *   A summary of the ablation speedup analysis is available in [algorithms/ablation_speedup_analysis.csv](algorithms/ablation_speedup_analysis.csv). The analysis is performed by the [`analyze_ablation_results`](algorithms/ablation_runner.py) function in [algorithms/ablation_runner.py](algorithms/ablation_runner.py).
2.  **Hyperparameter Settings**:
    *   Hyperparameter tuning summaries for various datasets are located in the [results/hyper_tuning/](results/hyper_tuning/) directory (e.g., [results/hyper_tuning/apple_summary.txt](results/hyper_tuning/apple_summary.txt), [results/hyper_tuning/measles_summary.txt](results/hyper_tuning/measles_summary.txt)).
3.  **Energy Consumption**:
    *   Energy consumption results for ESP32 devices are in the `results/esp32_energy` directory.
    *   Energy consumption results for Raspberry Pi devices are in the `results/pi_energy` directory.

## Algorithm Implementation
Algorithm implementations for microWATCH and other baseline Change Point Detection (CPD) algorithms (e.g., [`bocpd.py`](algorithms/bocpd.py), [`cusum.py`](algorithms/cusum.py), [`pelt.py`](algorithms/pelt.py), [`micro_watch.py`](algorithms/micro_watch.py)) are in the [algorithms/](algorithms/) directory.

## CPDPerf Implementation
Code for implementing the CPDPerf framework is in the [host/](host/) and [target/](target/) directories (for host and different kinds of devices). For example, detector implementations for target devices can be found in files
