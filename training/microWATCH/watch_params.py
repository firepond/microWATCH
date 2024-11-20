import json
import multiprocessing
import os
import accuracy
import distance
import math
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def distance_function(sample, dist, dist_metric):
    try:
        dist_mean = np.mean(dist, axis=0)
        dist_mean_list = np.tile(dist_mean, (sample.shape[0], 1))
        return dist_metric(sample, dist_mean_list)
    except:
        # raise an exception if the distance calculation fails
        raise Exception("Distance calculation failed")


def iterate_batches(points, batch_size):
    if batch_size == 0:
        # print debug information
        print(f"{points.shape[0]}, {batch_size}")
        raise Exception("Batch size cannot be zero")
    samples_count = math.ceil(points.shape[0] / batch_size)
    for sample_id in range(0, samples_count):
        sample = points[sample_id * batch_size : (sample_id + 1) * batch_size]
        yield sample


class ModularDetector:
    def __init__(
        self,
        threshold=3,
        new_dist_buffer_size=3,
        batch_size=3,
        max_dist_size=0,
        dist_metric=distance.euclidean,
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
        self.dist_metric = dist_metric
        self.DEBUG = DEBUG
        if self.DEBUG:
            self.values = []

    def detect(self, data):
        if self.batch_size == 0:
            raise Exception("Batch size cannot be zero")
        for batch_id, batch in enumerate(
            iterate_batches(data, batch_size=self.batch_size)
        ):
            if self.is_creating_new_dist:
                self.dist.extend(batch)

                if len(self.dist) >= self.new_dist_buffer_size:
                    self.is_creating_new_dist = False
                    values = [
                        distance_function(
                            np.array(s), np.array(self.dist), self.dist_metric
                        )
                        for s in iterate_batches(np.array(self.dist), self.batch_size)
                    ]
                    self.threshold = np.max(values) * self.threshold_ratio

            else:
                value = distance_function(
                    np.array(batch), np.array(self.dist), self.dist_metric
                )

                if value > self.threshold:
                    self.locations.append(batch_id * self.batch_size)
                    self.dist = []
                    self.is_creating_new_dist = True

                if self.max_dist_size == 0 or len(self.dist) < self.max_dist_size:
                    self.dist.extend(batch)
                    values = [
                        distance_function(
                            np.array(s), np.array(self.dist), self.dist_metric
                        )
                        for s in iterate_batches(np.array(self.dist), self.batch_size)
                    ]
                    self.threshold = np.max(values) * self.threshold_ratio

        return self.locations


def true_positives(T, X, margin=5):
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP


def partition_from_cps(locations, n_obs):
    """Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.
    """
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def overlap(A, B):
    """
    Return the overlap (i.e. Jaccard index) of two sets
    """
    return len(A.intersection(B)) / len(A.union(B))


def cover_single(S, Sprime):
    """
    Compute the covering of a segmentation S by a segmentation Sprime.

    This follows equation (8) in Arbaleaz, 2010.
    """
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(annotations, predictions, n_obs):
    """
    Compute the average segmentation covering against the human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted Cp locations
    n_obs : number of observations in the series

    """
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)

    Cs = [cover_single(Ak[k], pX) for k in Ak]
    return sum(Cs) / len(Cs)


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """
    Compute the F-measure based on human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted CP locations
    alpha : value for the F-measure, alpha=0.5 gives the F1-measure
    return_PR : whether to return precision and recall too
    """
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F


def clean_cps(locations, n_obs):
    # Filter change points to ensure they fall within the valid range [1, n_obs - 2]
    valid_cps = set([cp for cp in locations if 1 <= cp < n_obs - 1])
    # Return the sorted list of valid change points
    return sorted(valid_cps)


def scores(locations, dataset_name, n_obs):
    # Initialize the attribute if it doesn't exist
    if not hasattr(scores, "annotations"):
        scores.annotations = annotations[dataset_name]

    locations = clean_cps(locations, n_obs)
    f1, precision, recall = f_measure(scores.annotations, locations, return_PR=True)
    cover = covering(scores.annotations, locations, n_obs)
    return f1, cover


def scoring_function(params):
    data = params["data"]
    detector = ModularDetector(
        batch_size=params["batch_size"],
        threshold=params["threshold"],
        max_dist_size=params["max_dist_size"],
        dist_metric=params["metric"],
        new_dist_buffer_size=params["new_dist_buffer_size"],
    )
    if detector.batch_size == 0:
        raise Exception("Batch size cannot be zero")
    locations = detector.detect(data)
    f1, cover = scores(locations, params["dataset_name"], data.shape[0])
    return f1, cover


def f(params):
    f1, cover = scoring_function(params)
    scoring_metric = params["scoring_metric"]
    if scoring_metric == "f1":
        return {"loss": -1 * f1, "status": STATUS_OK}
    elif scoring_metric == "cover":
        return {"loss": -1 * cover, "status": STATUS_OK}
    else:
        return {"loss": -1 * (f1 + cover), "status": STATUS_OK}


distance_measures = [
    distance.acc,
    distance.add_chisq,
    distance.bhattacharyya,
    distance.braycurtis,
    distance.canberra,
    distance.chebyshev,
    distance.chebyshev_min,
    distance.clark,  # does not work for most of the datasets
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


def search(joined_list):
    if len(joined_list) == 3:
        data_path, distance_index, scoring_metric = joined_list
    elif len(joined_list) == 2:
        data_path, distance_index = joined_list
        scoring_metric = "sum"
    else:
        raise Exception("Invalid joined list")
    metric = distance_measures[distance_index]
    batch_size = 3
    dataset_name = data_path.split("/")[-1].split(".")[0]
    data = np.loadtxt(data_path, delimiter=",")

    # threshold tuning
    param_space = {
        "threshold": hp.uniform("threshold", 0.1, 4),
        "metric": metric,
        "batch_size": hp.uniformint("batch_size", 3, 6),
        "max_dist_size": hp.uniformint("max_dist_size", 10, 100),
        "new_dist_buffer_size": hp.uniformint("new_dist_buffer_size", 3, 50),
        "data": data,
        "dataset_name": dataset_name,
        "scoring_metric": scoring_metric,
    }
    trials = Trials()
    results = None
    loss_threshold = -0.99
    if scoring_metric == "sum":
        loss_threshold = -1.99
    try:
        results = fmin(
            f,
            param_space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials,
            loss_threshold=loss_threshold,
        )
    except:
        # if fails, test the results so far
        # if evals runned is greater than 0
        if len(trials.results) > 0:
            # get the best result
            results = trials.best_trial["result"]["params"]
        else:
            # return False and index
            return [False, distance_index]

    batch_size = int(results["batch_size"])
    max_dist_size = int(results["max_dist_size"])
    new_dist_buffer_size = int(results["new_dist_buffer_size"])
    threshold = results["threshold"]

    # running best model
    detector = ModularDetector(
        batch_size=batch_size,
        threshold=threshold,
        max_dist_size=max_dist_size,
        dist_metric=metric,
        new_dist_buffer_size=new_dist_buffer_size,
    )
    # debug information

    locations = detector.detect(data)
    f1, cover = accuracy.scores(locations, dataset_name, data.shape[0])
    print(
        f"batch: {batch_size}, threshold: {results['threshold']}, max_dist: {max_dist_size}, new_dist: {new_dist_buffer_size}, f1:{f1}, cover:{cover}"
    )
    return [
        dataset_name,
        distance_index,
        batch_size,
        threshold,
        max_dist_size,
        new_dist_buffer_size,
        f1,
        cover,
    ]


def main():
    # Directory containing the files
    directory = "./csv"

    # List all files in the directory
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    global annotations
    with open("./annotations.json", "r") as fp:
        annotations = json.load(fp)

    # Number of processes to use (e.g., number of CPU cores)
    num_processes = multiprocessing.cpu_count() - 4
    # num_processes = 1
    # sort file by name
    files.sort()

    for file in files:
        print(f"Processing {file}")
        # search for best f1
        # joined_list of files and distance indexes
        joined_list = [(file, i, "f1") for i in range(len(distance_measures))]
        # map processes to the joined list, then save the results
        print("processing f1")
        best_f1 = multiprocessing.Pool(num_processes).map(search, joined_list)
        dataset_name = file.split("/")[-1].split(".")[0]
        # save results to a csv file, named as {dataset_name}_f1_results.csv
        with open(f"./results/{dataset_name}_f1_results.csv", "w") as fp:
            fp.write(
                "file, distance_index, batch_size, threshold, max_dist_size, new_dist_buffer_size, f1, cover\n"
            )
            for result in best_f1:
                if result[0]:
                    # format result to a string
                    formatted = f"{result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]}, {result[5]}, {result[6]}, {result[7]}\n"
                    fp.write(formatted)
            fp.close()

        # search for best cover, same as f1
        joined_list = [(file, i, "cover") for i in range(len(distance_measures))]
        print("processing cover")
        best_cover = multiprocessing.Pool(num_processes).map(search, joined_list)
        # save results to a csv file, named as {dataset_name}_cover_results.csv
        with open(f"./results/{dataset_name}_cover_results.csv", "w") as fp:
            fp.write(
                "file, distance_index, batch_size, threshold, max_dist_size, new_dist_buffer_size, f1, cover\n"
            )
            for result in best_cover:
                if result[0]:
                    # format result to a string
                    formatted = f"{result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]}, {result[5]}, {result[6]}, {result[7]}\n"
                    fp.write(formatted)
            fp.close()
        print(f"{dataset_name} done")
        # percentage completed
        print(f"{files.index(file) / len(files) * 100}% completed")


if __name__ == "__main__":
    main()
