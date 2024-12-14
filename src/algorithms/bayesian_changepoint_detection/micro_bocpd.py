import os

try:
    from ulab import numpy as np
except:
    import numpy as np

try:
    from ulab import scipy as spy
except:
    import scipy as spy


gammaln = spy.special.gammaln


def gamma(z):
    return spy.special.gamma(z)


def constant_hazard(r):
    lam = 250
    """
    Hazard function for bayesian online learning
    Arguments:
        lam - inital prob
        r - R matrix
    """
    return 1 / lam * np.ones(r.shape)


def hazard_function(r):
    return constant_hazard(r)


def online_changepoint_detection(data, hazard_function, log_likelihood_class):
    """
    Use online bayesian changepoint detection
    https://scientya.com/bayesian-online-change-point-detection-an-intuitive-understanding-b2d2b9dc165b

    Parameters:
    data    -- the time series data

    Outputs:
        R  -- is the probability at time step t that the last sequence is already s time steps long
        maxes -- the argmax on column axis of matrix R (growth probability value) for each time step
    """
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = log_likelihood_class.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_function(np.array(range(t + 1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])

        # Update the parameter sets for each possible run length.
        log_likelihood_class.update_theta(x, t=t)

        maxes[t] = np.argmax(R[:, t])

    return R, maxes


def t_pdf(x, df, loc=0, scale=1):
    # 如果是 loc=0, scale=1，则退化为标准 t 分布

    # 标准化
    z = (x - loc) / scale

    # 标准 t 分布的 PDF 系数
    numerator = gamma((df + 1) / 2.0)
    denominator = np.sqrt(df * np.pi) * gamma(df / 2.0)
    coefficient = numerator / denominator

    # (1 + z^2/df)^(-(df+1)/2)
    exponent = -(df + 1) / 2.0
    pdf_val_standard = coefficient * (1 + (z**2 / df)) ** exponent

    # loc-scale变换后的 PDF: 对标准 PDF 再除以 scale
    return pdf_val_standard / scale


class StudentT:
    def __init__(
        self, alpha: float = 0.1, beta: float = 0.1, kappa: float = 1, mu: float = 0
    ):
        """
        StudentT distribution except normal distribution is replaced with the student T distribution
        https://en.wikipedia.org/wiki/Normal-gamma_distribution

        Parameters:
            alpha - alpha in gamma distribution prior
            beta - beta inn gamma distribution prior
            mu - mean from normal distribution
            kappa - variance from normal distribution
        """

        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data: np.array):
        """
        Return the pdf function of the t distribution

        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        return t_pdf(
            x=data,
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa)),
        )

    def update_theta(self, data: np.array, **kwargs):
        """
        Performs a bayesian update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (
                self.beta0,
                self.beta
                + (self.kappa * (data - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0


def isnan(num):
    return (num - num) != 0


def logaddexp(x1, x2):
    # impelment logaddexp
    return np.log(np.exp(x1) + np.exp(x2))


def var(x):
    # implement variance without using numpy.var
    return np.mean((x - np.mean(x)) ** 2)


def simple_logsumexp(x):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def const_prior(t, p: float = 0.25):
    return np.log(p)


# class IndepentFeaturesLikelihood:
#     """
#     Return the pdf for an independent features model discussed in xuan et al

#     Parmeters:
#         data - the datapoints to be evaluated (shape: 1 x D vector)
#         t - start of data segment
#         s - end of data segment
#     """

#     def pdf(self, data: np.array, t: int, s: int):
#         s += 1
#         n = s - t
#         x = data[t:s]
#         if len(x.shape) == 2:
#             d = x.shape[1]
#         else:
#             d = 1
#             x = np.atleast_2d(x).T

#         N0 = d  # weakest prior we can use to retain proper prior
#         V0 = var(x)
#         Vn = V0 + np.sum(x**2, axis=0)

#         # sum over dimension and return (section 3.1 from Xuan paper):
#         return d * (
#             -(n / 2) * np.log(np.pi)
#             + (N0 / 2) * np.log(V0)
#             - gammaln(N0 / 2)
#             + gammaln((N0 + n) / 2)
#         ) - np.sum(((N0 + n) / 2) * np.log(Vn), axis=0)


# def offline_changepoint_detection(data, log_likelihood_class, truncate: int = -40):
#     """
#     Compute the likelihood of changepoints on data.

#     Parameters:
#     data    -- the time series data
#     truncate  -- the cutoff probability 10^truncate to stop computation for that changepoint log likelihood

#     Outputs:
#         P  -- the log-likelihood of a datasequence [t, s], given there is no changepoint between t and s
#         Q -- the log-likelihood of data
#         Pcp --  the log-likelihood that the i-th changepoint is at time step t. To actually get the probility of a changepoint at time step t sum the probabilities.
#     """
#     p = 1 / (len(data) + 1)

#     # Set up the placeholders for each parameter
#     n = len(data)
#     Q = np.zeros((n,))
#     g = np.zeros((n,))
#     G = np.zeros((n,))
#     P = np.ones((n, n)) * -np.inf

#     # save everything in log representation
#     for t in range(n):
#         g[t] = const_prior(t, p)
#         if t == 0:
#             G[t] = g[t]
#         else:
#             G[t] = logaddexp(G[t - 1], g[t])

#     P[n - 1, n - 1] = log_likelihood_class.pdf(data, t=n - 1, s=n)
#     Q[n - 1] = P[n - 1, n - 1]

#     for t in reversed(range(n - 1)):
#         P_next_cp = -np.inf  # == log(0)
#         for s in range(t, n - 1):
#             P[t, s] = log_likelihood_class.pdf(data, t=t, s=s + 1)

#             # compute recursion
#             summand = P[t, s] + Q[s + 1] + g[s + 1 - t]
#             P_next_cp = logaddexp(P_next_cp, summand)

#             # truncate sum to become approx. linear in time (see
#             # Fearnhead, 2006, eq. (3))
#             if summand - P_next_cp < truncate:
#                 break

#         P[t, n - 1] = log_likelihood_class.pdf(data, t=t, s=n)

#         # (1 - G) is numerical stable until G becomes numerically 1
#         if G[n - 1 - t] < -1e-15:  # exp(-1e-15) = .99999...
#             antiG = np.log(1 - np.exp(G[n - 1 - t]))
#         else:
#             # (1 - G) is approx. -log(G) for G close to 1
#             antiG = np.log(-G[n - 1 - t])

#         Q[t] = logaddexp(P_next_cp, P[t, n - 1] + antiG)

#     Pcp = np.ones((n - 1, n - 1)) * -np.inf
#     for t in range(n - 1):
#         Pcp[0, t] = P[0, t] + Q[t + 1] + g[t] - Q[0]
#         if isnan(Pcp[0, t]):
#             Pcp[0, t] = -np.inf
#     for j in range(1, n - 1):
#         for t in range(j, n - 1):
#             tmp_cond = (
#                 Pcp[j - 1, j - 1 : t]
#                 + P[j : t + 1, t]
#                 + Q[t + 1]
#                 + g[0 : t - j + 1]
#                 - Q[j : t + 1]
#             )
#             Pcp[j, t] = simple_logsumexp(tmp_cond)
#             if isnan(Pcp[j, t]):
#                 Pcp[j, t] = -np.inf

#     return Q, P, Pcp


# def bcpd(data):
#     Q_ifm, P_ifm, Pcp_ifm = offline_changepoint_detection(
#         data,
#         IndepentFeaturesLikelihood(),
#         truncate=-20,
#     )


class BOCPD:
    def __init__(self):
        pass

    def reinit(self):
        pass

    def detect(self, data):
        R, maxes = online_changepoint_detection(
            data, hazard_function, StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
        )
        Nw = 10
        cp_probs = R[Nw, Nw:-1]
        # cps = np.where(cp_probs > 0.3)
        # find index of the changepoints, use p_probs > 0.3, but do not use np where
        cps = []
        for i in range(1, len(cp_probs)):
            if cp_probs[i] > 0.3:
                cps.append(i)

        return cps


# csv_folder = "/home/campus.ncl.ac.uk/c4060464/esp32/microWATCH/datasets/csv/"

# # for all files in the folder

# for file_name in os.listdir(csv_folder):
#     print("Reading file: ", file_name)
#     file_path = csv_folder + file_name
#     data = np.loadtxt(file_path, delimiter=",")

#     bcpd(data)
