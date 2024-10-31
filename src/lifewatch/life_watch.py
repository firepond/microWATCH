import numpy as np
import ot

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Wrapperfor None-detector")
    parser.add_argument("-i", "--input", help="input datasets name", required=True)
    parser.add_argument("-o", "--output", help="path to the output file")
    return parser.parse_args()


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
    # Ensure sample is a NumPy array
    if not isinstance(sample, np.ndarray):
        raise ValueError("Input sample must be a NumPy array.")
    # Ensure dist is a NumPy array
    if not isinstance(dist, np.ndarray):
        raise ValueError("Input dist must be a NumPy array.")

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


def iterate_batches(points: np.ndarray, batch_size: int):
    """
    Yield batches of data from a NumPy array.

    Args:
        points (np.ndarray): A NumPy array containing the data to be processed.
        batch_size (int): The size of each batch to yield.

    Yields:
        np.ndarray: A slice of the points array representing the current batch.
    """
    # Ensure points is a NumPy array
    if not isinstance(points, np.ndarray):
        raise ValueError("Input points must be a NumPy array.")

    # Initialize the start index
    start_index = 0
    total_samples = points.shape[0]

    while start_index < total_samples:
        # Calculate the end index for the current batch
        end_index = min(start_index + batch_size, total_samples)
        # Yield the current batch
        yield points[start_index:end_index]
        # Move to the next batch
        start_index += batch_size


class LIFE_WATCH:
    def __init__(
        self, threshold=2, new_dist_buffer_size=16, batch_size=3, max_dist_size=100
    ):
        self.threshold_ratio = threshold
        self.max_dist_size = max_dist_size
        self.new_dist_buffer_size = new_dist_buffer_size
        self.batch_size = batch_size
        self.is_creating_new_dist = True
        self.dist_buffer = []
        self.new_locations = []
        self.recurring_locations = []
        self.memory = []  # list [data points, threshold]
        self.cur_dist_id = 0

    def add_batch_to_recurring_dist(self, batch, dist_id):
        current_dist = self.memory[dist_id][0]
        current_dist = np.vstack([current_dist, batch])
        distances = [
            wassertein(s, current_dist)
            for s in iterate_batches(current_dist, self.batch_size)
        ]
        max_dis = np.max(distances)
        self.memory[dist_id][1] = max_dis

    def add_batch_to_new_dist(self, batch):
        self.dist_buffer.extend(batch)
        if len(self.dist_buffer) >= self.new_dist_buffer_size:
            self.is_creating_new_dist = False
            buffer = np.array(self.dist_buffer)
            values = [
                wassertein(s, buffer) for s in iterate_batches(buffer, self.batch_size)
            ]
            new_dist = [
                buffer,
                np.max(values) * self.threshold_ratio,
            ]
            self.cur_dist_id = len(self.memory)
            self.memory.append(new_dist)
            self.dist_buffer = []

    def detect(self, data):
        for batch_id, batch in enumerate(iterate_batches(data, self.batch_size)):
            if self.is_creating_new_dist:
                self.add_batch_to_new_dist(batch)
            else:
                cur_dist = self.memory[self.cur_dist_id][0]
                w_distance = wassertein(batch, cur_dist)

                if w_distance > self.memory[self.cur_dist_id][1]:
                    # w distance > current distribution threshold
                    # check for another distributions in memory pool
                    length = len(self.memory)
                    distances = np.zeros((length, 4))
                    for i in range(0, length):
                        w_distance = wassertein(
                            np.array(batch), np.array(self.memory[i][0])
                        )
                        ratio = w_distance / self.memory[i][1]
                        # id, task threshold, w distance, ratio
                        compare_result = np.array(
                            [i, self.memory[i][1], w_distance, ratio]
                        )
                        distances[i] = compare_result

                    distances = distances[distances[:, 3].argsort()]
                    if distances[0][2] < distances[0][1]:
                        # w distance < distribution threshold for the smallest ratio,
                        # so it is a recurring task
                        # add current point to that distribution
                        # set it to be the current active distribution
                        self.recurring_locations.append(batch_id * self.batch_size)
                        self.cur_dist_id = int(distances[0][0])
                        self.add_batch_to_recurring_dist(batch, self.cur_dist_id)
                    else:
                        # new task
                        self.new_locations.append(batch_id * self.batch_size)
                        self.is_creating_new_dist = True
                        self.add_batch_to_new_dist(batch)
                else:
                    # data not changed, add new point to current distribution
                    self.add_batch_to_recurring_dist(batch, self.cur_dist_id)

        return self.new_locations, self.recurring_locations


def main():

    args = parse_args()

    dataset_name = args.input
    # dataset_name = "./datasets/apple.json"
    data = np.loadtxt(dataset_name, delimiter=",")
    print(data.shape)

    watch = LIFE_WATCH()
    new_locations, recurring_locations = watch.detect(data)

    print(new_locations)
    print(recurring_locations)


if __name__ == "__main__":
    main()
