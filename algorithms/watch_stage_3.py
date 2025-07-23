import math
import numpy as np

import distance
from cpd_detector import cpd_detector


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

DISTANCE_COUNT = len(distance_measures)


def distance_function_mean(sample, dist_mean, dist_metric):
    dist_mean_list = np.array([dist_mean] * sample.shape[0])
    distance = dist_metric(sample, dist_mean_list)
    return distance


def iterate_batches(points, batch_size):
    samples_number = math.ceil(points.shape[0] / batch_size)
    for sample_id in range(0, samples_number):
        sample = points[sample_id * batch_size : (sample_id + 1) * batch_size]
        yield sample


class WATCH(cpd_detector):
    def __init__(
        self,
        threshold_ratio=0.51,
        new_dist_buffer_size=32,
        batch_size=3,
        max_dist_size=72,
        version=6,
    ):
        self.threshold_ratio = threshold_ratio
        self.max_dist_size = max_dist_size
        self.new_dist_buffer_size = new_dist_buffer_size
        self.batch_size = batch_size

        self.is_creating_new_dist = True

        self.dist = None
        self.locations = []

        self.metric = distance_measures[version]

        self.sum = None

    def reinit(self):
        self.is_creating_new_dist = True
        self.dist = None
        self.locations = []
        self.sum = None

    def detect(self, data):
        data_width = data.shape[1]
        max_size = max(self.max_dist_size, self.new_dist_buffer_size) + self.batch_size
        if max_size > data.shape[0]:
            max_size = data.shape[0]
        if self.sum is None:
            self.sum = np.zeros(data_width)
        if self.dist is None:
            self.dist = np.zeros((max_size, data_width))
        dist_len = 0
        batch_size = self.batch_size
        for batch_id, batch in enumerate(iterate_batches(data, self.batch_size)):
            if self.is_creating_new_dist:
                current_batch_rows = batch.shape[0]
                self.dist[dist_len : dist_len + current_batch_rows] = batch
                dist_len += current_batch_rows
                self.sum += np.sum(batch, axis=0)
                if dist_len >= self.new_dist_buffer_size:
                    self.is_creating_new_dist = False
                    dist_array = self.dist[:dist_len, ::]
                    dist_mean = self.sum / dist_len
                    max_dist = 0
                    for s in iterate_batches(dist_array, self.batch_size):
                        cur_dist = distance_function_mean(s, dist_mean, self.metric)
                        if cur_dist > max_dist:
                            max_dist = cur_dist
                    self.threshold = max_dist * self.threshold_ratio
            else:
                dist_mean = self.sum / dist_len
                value = distance_function_mean(batch, dist_mean, self.metric)

                if value > self.threshold:
                    self.locations.append(batch_id * self.batch_size)
                    self.dist = np.zeros((max_size, data_width))
                    self.is_creating_new_dist = True

                    dist_len = 0
                    self.sum = np.zeros(data_width)
                    self.dist.fill(0)  # Reset the distance buffer

                if dist_len < self.max_dist_size or self.max_dist_size == 0:
                    self.dist[dist_len : dist_len + len(batch)] = batch
                    dist_len += len(batch)
                    self.sum += np.sum(batch, axis=0)
                    dist_array = self.dist[:dist_len, ::]
                    dist_mean = self.sum / dist_len
                    max_dist = 0
                    for s in iterate_batches(dist_array, self.batch_size):
                        cur_dist = distance_function_mean(s, dist_mean, self.metric)
                        if cur_dist > max_dist:
                            max_dist = cur_dist
                    self.threshold = max_dist * self.threshold_ratio

        return self.locations
