import math
import numpy as np
import ot

from load_dataset import TimeSeries
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Wrapperfor None-detector")
    parser.add_argument("-i", "--input", help="input datasets name", required=True)
    parser.add_argument("-o", "--output", help="path to the output file")
    return parser.parse_args()


def find_unique_sub_arrays(arr):
    """
    Find unique sub-arrays and their counts in an n-dimensional NumPy array.

    Args:
        arr (numpy.ndarray): Input n-dimensional array.

    Returns:
        tuple: A tuple containing:
            - unique_sub_arrays (numpy.ndarray): Array of unique sub-arrays.
            - counts (numpy.ndarray): Array of counts corresponding to unique sub-arrays.
    """
    # Get the number of dimensions in the input array
    num_dims = arr.ndim

    # Initialize a dictionary to track unique sub-arrays and their frequencies
    unique_sub_arrays = {}

    # Calculate the shape for slicing
    shape = arr.shape

    if num_dims == 1:
        unique_sub_arrays = np.unique(arr)
        counts = np.unique(arr, return_counts=True)
        return unique_sub_arrays, counts[1]

    # Generate slices based on the number of dimensions
    if num_dims == 2:
        # For 2D arrays, loop through all possible sub-arrays
        for i in range(shape[0]):
            sub_array = arr[i, :]
            sub_array_tuple = tuple(map(tuple, [sub_array]))  # Make it hashable
            unique_sub_arrays[sub_array_tuple] = (
                unique_sub_arrays.get(sub_array_tuple, 0) + 1
            )

    elif num_dims == 3:
        # For 3D arrays, loop through all possible 2D slices
        for i in range(shape[0]):
            sub_array = arr[i, :, :]
            sub_array_tuple = tuple(map(tuple, sub_array))  # Make it hashable
            unique_sub_arrays[sub_array_tuple] = (
                unique_sub_arrays.get(sub_array_tuple, 0) + 1
            )

    else:
        # For higher dimensions, loop through the first dimension and treat the rest as a sub-array
        for i in range(shape[0]):
            sub_array = arr[i, ...]  # Use Ellipsis for higher dimensions
            sub_array_tuple = tuple(
                map(tuple, map(np.ravel, sub_array))
            )  # Make it hashable
            unique_sub_arrays[sub_array_tuple] = (
                unique_sub_arrays.get(sub_array_tuple, 0) + 1
            )

    # Convert unique sub-arrays and their counts to numpy arrays
    sub_arrays_list = np.array(
        [np.array(sub_array) for sub_array in unique_sub_arrays.keys()]
    )
    counts_list = np.array(list(unique_sub_arrays.values()))
    if num_dims == 2:
        sub_arrays_list = sub_arrays_list[:, 0, :]
    return sub_arrays_list, counts_list


def wassertein(sample, dist):
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
    loss_matrix = ot.dist(sample_list, dist_list, metric="euclidean")

    ot_dis = ot.emd2(sample_dist, dist_dist, loss_matrix)
    return ot_dis


def iterate_batches(points, batch_size):
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
        DEBUG=False,
    ):
        self.threshold_ratio = threshold
        self.max_dist_size = max_dist_size
        self.new_dist_buffer_size = new_dist_buffer_size
        self.batch_size = batch_size
        self.is_creating_new_dist = True
        self.dist = []
        self.dist_values = []
        self.locations = []
        self.DEBUG = DEBUG
        if self.DEBUG:
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
                        wassertein(np.array(s), np.array(self.dist))
                        for s in iterate_batches(np.array(self.dist), self.batch_size)
                    ]
                    self.threshold = np.max(values) * self.threshold_ratio
                    # print(self.threshold)
            else:
                value = wassertein(np.array(batch), np.array(self.dist))

                if value > self.threshold:
                    self.locations.append(batch_id * self.batch_size)
                    self.dist = []
                    self.is_creating_new_dist = True
                if self.max_dist_size == 0 or len(self.dist) < self.max_dist_size:
                    self.dist.extend(batch)
                    values = [
                        wassertein(np.array(s), np.array(self.dist))
                        for s in iterate_batches(np.array(self.dist), self.batch_size)
                    ]
                    self.threshold = np.max(values) * self.threshold_ratio

        return self.locations


def dummy_test():
    dummy_data = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [1000, 1000],
            [5000, 5000],
            [10000, 10000],
            [1000, 1000],
            [5000, 5000],
            [10000, 10000],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ]
    )

    watch = WATCH()
    locations = watch.detect(dummy_data)
    print(locations)


def main():

    args = parse_args()

    dataset_name = args.input
    dataset_prefix = "./datasets/"
    dataset_loc = dataset_prefix + "/" + dataset_name
    dataset_loc = dataset_name
    print(dataset_loc)
    ts = TimeSeries.from_json(dataset_loc)

    data = ts.df.to_numpy()
    data = data[::, 1::]

    print(data.shape)

    watch = WATCH()
    locations = watch.detect(data)

    print(locations)


if __name__ == "__main__":
    main()
