import gc
import math
import os
import time

try:
    from ulab import numpy as np
except:
    import numpy as np

import distance


distance_measures = [
    distance.acc,
    distance.add_chisq,
    distance.bhattacharyya,
    distance.braycurtis,
    distance.canberra,
    distance.chebyshev,
    distance.chebyshev_min,
    distance.clark,
    distance.czekanowski,
    distance.divergence,
    distance.euclidean,
    distance.google,
    distance.gower,
    distance.hellinger,
    distance.jeffreys,
    distance.jensenshannon_divergence,
    distance.jensen_difference,
    distance.k_divergence,
    distance.kl_divergence,
    distance.kulczynski,
    distance.lorentzian,
    distance.manhattan,
    distance.matusita,
    distance.max_symmetric_chisq,
    distance.minkowski,
    distance.motyka,
    distance.neyman_chisq,
    distance.nonintersection,
    distance.pearson_chisq,
    distance.penroseshape,
    distance.soergel,
    distance.squared_chisq,
    distance.squaredchord,
    distance.taneja,
    distance.tanimoto,
    distance.topsoe,
    distance.vicis_symmetric_chisq,
    distance.vicis_wave_hedges,
    distance.wave_hedges,
]
# Note: Cosine distance does not work, dice distance does not work, jaccard distance does not work
# kumarjohnson distance does not work, maryland bridge distance does not work, squared euclidean does not work
# correlation_pearson does not work for most of the datasets


def distance_function(sample, dist, dist_metric):
    try:
        dist_mean = np.mean(dist, axis=0)
        dist_mean_list = np.array([dist_mean] * sample.shape[0])
        distance = dist_metric(sample, dist_mean_list)
        return distance
    except:
        raise Exception("Distance calculation failed")


def iterate_batches(points, batch_size):
    samples_number = math.ceil(points.shape[0] / batch_size)
    for sample_id in range(0, samples_number):
        sample = points[sample_id * batch_size : (sample_id + 1) * batch_size]
        yield sample


class microWATCH:
    def __init__(
        self,
        threshold=0.51,
        new_dist_buffer_size=32,
        batch_size=3,
        max_dist_size=72,
        metric=distance.euclidean,
    ):
        self.threshold_ratio = threshold
        self.max_dist_size = max_dist_size
        self.new_dist_buffer_size = new_dist_buffer_size
        self.batch_size = batch_size
        self.is_creating_new_dist = True
        self.dist = []
        self.dist_values = []
        self.locations = []
        self.metric = metric
        self.values = []
        # self.distance_index = 0

    def reinit(self):
        self.is_creating_new_dist = True
        self.dist = []
        self.dist_values = []
        self.locations = []
        self.values = []

    def set_params(self, distance_index, dataset_path):
        file_name = dataset_path.split("/")[-1]
        params = get_params(file_name, distance_index=distance_index)
        # raise an exception if the params are not found
        if len(params) == 0:
            print(f"Params not found for {file_name} and index {distance_index}")
            return -1
        # print(f"Params: {params}")
        threshold = float(params[3])
        # print(params[4])
        max_dist_size = int(params[4])
        new_dist_buffer_size = int(params[5])
        batch_size = int(params[2])
        self.threshold_ratio = threshold
        self.max_dist_size = max_dist_size
        self.new_dist_buffer_size = new_dist_buffer_size
        self.batch_size = batch_size
        gc.collect()

    def detect(self, data):
        for batch_id, batch in enumerate(
            iterate_batches(data, batch_size=self.batch_size)
        ):
            if self.is_creating_new_dist:
                self.dist.extend(batch)
                if len(self.dist) >= self.new_dist_buffer_size:
                    self.is_creating_new_dist = False
                    values = [
                        distance_function(np.array(s), np.array(self.dist), self.metric)
                        for s in iterate_batches(np.array(self.dist), self.batch_size)
                    ]
                    self.threshold = np.max(values) * self.threshold_ratio
                    # print(self.threshold)
            else:
                value = distance_function(
                    np.array(batch), np.array(self.dist), self.metric
                )

                if value > self.threshold:
                    self.locations.append(batch_id * self.batch_size)
                    self.dist = []
                    self.is_creating_new_dist = True
                if self.max_dist_size == 0 or len(self.dist) < self.max_dist_size:
                    self.dist.extend(batch)
                    values = [
                        distance_function(np.array(s), np.array(self.dist), self.metric)
                        for s in iterate_batches(np.array(self.dist), self.batch_size)
                    ]
                    self.threshold = np.max(values) * self.threshold_ratio

        return self.locations


def get_data(file_name):
    data = np.loadtxt(file_name, delimiter=",")
    gc.collect()
    return data


def get_params(file_name, distance_index):
    # get params from a csv file

    open_file = open("best_params.csv", "r")
    data = []
    # read line by line, not enough ram to read all at once

    file_name = file_name.split(".")[0]
    for line in open_file.readlines():
        line = line.strip().split(",")
        if line[0] == file_name and int(line[1]) == distance_index:
            data = line
    open_file.close()
    # print(f"Params: {data}")
    gc.collect()
    return data


def test_time(file_name, times_count=10, index=0):
    data = get_data("csv/" + file_name)
    params = get_params(file_name, index)
    # raise an exception if the params are not found
    if len(params) == 0:
        print(f"Params not found for {file_name} and index {index}")
        return -1
    # print(f"Params: {params}")
    threshold = float(params[3])
    # print(params[4])
    max_dist_size = int(params[4])
    new_dist_buffer_size = int(params[5])
    batch_size = int(params[2])
    watch = microWATCH(
        threshold,
        new_dist_buffer_size,
        batch_size,
        max_dist_size,
        metric=distance_measures[index],
    )
    print(f"Testing {file_name} with distance {index}")
    gc.collect()

    start = time.ticks_ms()

    for i in range(times_count):
        watch.reinit()
        try:
            locations = watch.detect(data)
            print(f"Locations: {locations}")
        except:
            # get distance name from the index
            name = distance_measures[index].__name__
            print(f"Distance calculation failed for {name}")
            return -1

    end = time.ticks_ms()

    diff_aver = (end - start) / times_count
    print(f"Time: {diff_aver} ms per execution")

    time.sleep(1)
    return diff_aver


def test_all(times_count=10):
    # list all files in "csv folder"
    files = os.listdir("csv")
    fail_log = open("fail_log.txt", "w+")
    # write the time for each file
    distance_indexes = list(range(len(distance_measures)))
    time_log = open("time_log.txt", "w+")
    time_log.write("distance_index, time(ms)\n")
    for index in distance_indexes:
        print(f"Testing distance {distance_measures[index].__name__}, index {index}")
        total_time = 0
        all_success = True
        for file in files:
            runtime = test_time(file, times_count, index)
            gc.collect()
            if runtime == -1:
                all_success = False
                print(f"Failed for {file} and index {index}")
                fail_log.write(f"{file}, {index}\n")
                break
            else:
                # success
                total_time += runtime
        if all_success:
            # write to a file
            print(
                f"Sum time for distance {distance_measures[index].__name__}: {total_time}"
            )
            time_log.write(f"{index}, {total_time}")
    fail_log.close()
    time_log.close()
