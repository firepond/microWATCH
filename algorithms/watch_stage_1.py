import gc
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


class WATCH:
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

        self.dist = []
        self.locations = []

        self.metric = distance_measures[version]

    def reinit(self):
        self.is_creating_new_dist = True
        self.dist = []
        self.locations = []
        gc.collect()

    def detect(self, data):
        for batch_id, batch in enumerate(
            iterate_batches(data, batch_size=self.batch_size)
        ):
            if self.is_creating_new_dist:
                self.dist.extend(batch)
                if len(self.dist) >= self.new_dist_buffer_size:
                    self.is_creating_new_dist = False
                    dist_mean = np.mean(self.dist, axis=0)
                    values = [
                        distance_function_mean(np.array(s), dist_mean, self.metric)
                        for s in iterate_batches(np.array(self.dist), self.batch_size)
                    ]
                    self.threshold = np.max(values) * self.threshold_ratio
                    # print(self.threshold)
            else:
                dist_mean = np.mean(self.dist, axis=0)
                value = distance_function_mean(np.array(batch), dist_mean, self.metric)

                if value > self.threshold:
                    self.locations.append(batch_id * self.batch_size)
                    self.dist = []
                    self.is_creating_new_dist = True

                if self.max_dist_size == 0 or len(self.dist) < self.max_dist_size:
                    self.dist.extend(batch)
                    values = [
                        distance_function_mean(np.array(s), dist_mean, self.metric)
                        for s in iterate_batches(np.array(self.dist), self.batch_size)
                    ]
                    self.threshold = np.max(values) * self.threshold_ratio

        return self.locations
