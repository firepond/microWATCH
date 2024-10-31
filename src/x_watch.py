import math
import os
import numpy as np
import ot

import accuracy


def wassertein(sample, dist, metric="euclidean", post_process="null"):
    """
    Calculate the Wasserstein distance between two distributions

    Args:
        arr (numpy.ndarray): Input n-dimensional array.

    Returns:
        tuple: A tuple containing:
            - unique_sub_arrays (numpy.ndarray): Array of unique sub-arrays.
            - counts (numpy.ndarray): Array of counts corresponding to unique sub-arrays.
    """
    # sample_list, sample_freq = find_unique_sub_arrays(sample)
    # dist_list, dist_freq = find_unique_sub_arrays(dist)
    sample_list = sample
    sample_freq = np.ones(sample.shape[0])

    dist_list = dist
    dist_freq = np.ones(dist.shape[0])

    sample_dist = sample_freq / sample.shape[0]
    dist_dist = dist_freq / dist.shape[0]

    # loss matrix
    loss_matrix = ot.dist(sample_list, dist_list, metric)

    ot_dis = ot.emd2(sample_dist, dist_dist, loss_matrix)
    # check name of post_process
    if post_process == "sqrt":
        return np.sqrt(ot_dis)
    elif post_process == "square":
        return ot_dis**2
    else:
        return ot_dis


def iterate_batches(points, batch_size):
    # this function is used to iterate over the data points in batches

    samples_number = math.ceil(points.shape[0] / batch_size)
    for sample_id in range(0, samples_number):
        sample = points[sample_id * batch_size : (sample_id + 1) * batch_size]
        yield sample


class WATCH:
    def __init__(
        self,
        threshold=2,
        new_dist_buffer_size=16,
        batch_size=3,
        max_dist_size=100,
    ):
        self.threshold_ratio = threshold
        self.max_dist_size = max_dist_size
        self.new_dist_buffer_size = new_dist_buffer_size
        self.batch_size = batch_size
        self.metric = "euclidean"
        self.post_process = "null"

        self.is_creating_new_dist = True

        self.dist = []
        self.dist_values = []
        self.locations = []

        self.values = []

    def detect(self, data):
        for batch_id, batch in enumerate(
            iterate_batches(data, batch_size=self.batch_size)
        ):
            if self.is_creating_new_dist:
                self.dist.extend(batch)
                if len(self.dist) >= self.new_dist_buffer_size:
                    self.is_creating_new_dist = False
                    values = [
                        wassertein(
                            np.array(s),
                            np.array(self.dist),
                            self.metric,
                            self.post_process,
                        )
                        for s in iterate_batches(np.array(self.dist), self.batch_size)
                    ]
                    self.threshold = np.max(values) * self.threshold_ratio
                    # print(self.threshold)
            else:
                value = wassertein(
                    np.array(batch), np.array(self.dist), self.metric, self.post_process
                )

                if value > self.threshold:
                    self.locations.append(batch_id * self.batch_size)
                    self.dist = []
                    self.is_creating_new_dist = True
                if self.max_dist_size == 0 or len(self.dist) < self.max_dist_size:
                    self.dist.extend(batch)
                    values = [
                        wassertein(
                            np.array(s),
                            np.array(self.dist),
                            self.metric,
                            self.post_process,
                        )
                        for s in iterate_batches(np.array(self.dist), self.batch_size)
                    ]
                    self.threshold = np.max(values) * self.threshold_ratio

        return self.locations


def main():
    dir = "./csv"
    # for all files in dir test the watch algorithm
    files = os.listdir(dir)
    metric = "sqeuclidean"
    post_process = "sqrt"
    for file in files:
        dataset_name = file.split("/")[-1].split(".")[0]
        data_loc = f"./csv/{file}"
        print(data_loc)
        data = np.loadtxt(data_loc, delimiter=",")
        # convert 1 dimensional data to m * 1 np array
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        watch = WATCH()
        watch.metric = metric
        watch.post_process = post_process
        locations = watch.detect(data)
        print(locations)
        f1, cover = accuracy.scores(locations, dataset_name, data.shape[0])
        print(f1, cover)


if __name__ == "__main__":
    main()
