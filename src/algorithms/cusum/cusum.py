import math
import gc

try:
    from ulab import numpy as np
except:
    import numpy as np


class CusumMeanDetector:

    def __init__(self, t_warmup=30, p_limit=0.01):
        self._t_warmup = t_warmup
        self._p_limit = p_limit
        self.statring = True

        self._reset()

    def reinit(self):
        self._reset()

    def get_params(self, file_name):
        # get params from a csv file

        open_file = open("cusum_best_params.csv", "r")
        data = []
        # read line by line, not enough ram to read all at once

        file_name = file_name.split(".")[0]
        for line in open_file.readlines():
            line = line.strip().split(",")
            if line[0] == file_name:
                data = line
        open_file.close()
        # print(f"Params: {data}")
        gc.collect()
        return data

    def set_params(self, dataset_path):
        file_name = dataset_path.split("/")[-1]
        params = self.get_params(file_name)
        # raise an exception if the params are not found
        if len(params) == 0:
            print(f"Params not found for {file_name}")
            return -1

        self.t_warmup = float(params[2])
        self.p_limit = float(params[3])
        gc.collect()

    def detect(self, data):
        changepoints = []
        for i in range(len(data)):
            prob, is_changepoint = self.predict_next(data[i])
            if is_changepoint:
                changepoints.append(i)
        return changepoints

    def predict_next(self, y):
        self._update_data(y)

        if self.current_t == self._t_warmup:
            self._init_params()

        if self.current_t >= self._t_warmup:
            prob, is_changepoint = self._check_for_changepoint()
            if is_changepoint:
                self._reset()

            return (1 - prob), is_changepoint

        else:
            return 0, False

    def _reset(self):
        self.current_t = 0

        self.current_obs = []

        self.current_mean = 0
        self.current_std = 0.1

    def _update_data(self, y):
        self.current_t += 1
        # convert ndarray y to list
        list_y = y.tolist()
        self.current_obs.append(list_y)

    def _init_params(self):
        self.current_mean = np.mean(np.array(self.current_obs))
        self.current_std = np.std(np.array(self.current_obs))

    def _check_for_changepoint(self):
        standardized_sum = np.sum(np.array(self.current_obs) - self.current_mean) / (
            self.current_std * self.current_t**0.5
        )
        prob = float(self._get_prob(standardized_sum))

        return prob, prob < self._p_limit

    def _get_prob(self, y):

        p = 0.5 * (1 + math.erf(abs(y) / math.sqrt(2)))
        prob = 2 * (1 - p)

        return prob
