import math
import numpy as np
from rpy2 import robjects
from rpy2.robjects import numpy2ri

r = robjects.r
x = r["source"]("wasserstein.R")
numpy2ri.activate()

r_wasserstein_dist = robjects.r["WassersteinTest"]


def wassertein(sample, dist):
    print("Calculating Wasserstein distance...")
    try:
        return abs(r_wasserstein_dist(sample, dist)[0])
    except:
        # print debug information
        print(
            "Error calculating Wasserstein distance. Ensure R and the required packages are set up correctly."
        )
        # print rpy2 information
        print("R version:", r.r["version"])
        print("R packages loaded:", r.r["loadedNamespaces"]())
        print("Sample shape:", sample.shape)
        exit()


def iterate_batches(points, batch_size):
    samples_number = math.ceil(points.shape[0] / batch_size)
    for sample_id in range(0, samples_number):
        sample = points[sample_id * batch_size : (sample_id + 1) * batch_size]
        yield sample


class WATCH:
    def __init__(
        self,
        threshold=3,
        new_dist_buffer_size=3,
        batch_size=3,
        max_dist_size=0,
        version=6,
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

        print("Locations found:", self.locations)
        return self.locations


if __name__ == "__main__":
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
