import json
from utils.metrics import covering
from utils.metrics import f_measure


def load_annotations(filename, dataset):
    with open(filename, "r") as fp:
        data = json.load(fp)
    return data[dataset]


def clean_cps(locations, n_obs):
    # Filter change points to ensure they fall within the valid range [1, n_obs - 2]
    valid_cps = set([cp for cp in locations if 1 <= cp < n_obs - 1])
    # Return the sorted list of valid change points
    return sorted(valid_cps)


def scores(locations, dataset_name, n_obs):
    # Initialize the attribute if it doesn't exist
    if not hasattr(scores, "annotations"):
        scores.annotations = load_annotations("./annotations.json", dataset_name)

    locations = clean_cps(locations, n_obs)
    f1, precision, recall = f_measure(scores.annotations, locations, return_PR=True)
    cover = covering(scores.annotations, locations, n_obs)
    return f1, cover
