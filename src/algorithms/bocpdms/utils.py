# # implement of special functions from scipy and numpy but bot available in micropython

# try:
#     from ulab import numpy as np
#     from ulab import scipy as spy
# except:
#     import numpy as np
#     import scipy as spy

import scipy
import numpy as np

logsumexp = scipy.special.logsumexp

# def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
#     """Compute the log of the sum of exponentials of input elements."""

#     if b is not None:
#         a, b = np.broadcast_arrays(a, b)
#         if np.any(b == 0):
#             a = a + 0.0  # promote to at least float
#             a[b == 0] = -np.inf

#     # Scale by real part for complex inputs, because this affects
#     # the magnitude of the exponential.
#     initial_value = -np.inf if np.size(a) == 0 else None
#     a_max = np.amax(a.real, axis=axis, keepdims=True, initial=initial_value)

#     if a_max.ndim > 0:
#         a_max[~np.isfinite(a_max)] = 0
#     elif not np.isfinite(a_max):
#         a_max = 0

#     if b is not None:
#         b = np.asarray(b)
#         tmp = b * np.exp(a - a_max)
#     else:
#         tmp = np.exp(a - a_max)

#     s = np.sum(tmp, axis=axis, keepdims=keepdims)
#     if return_sign:
#         # For complex, use the numpy>=2.0 convention for sign.
#         if np.issubdtype(s.dtype, np.complexfloating):
#             sgn = s / np.where(s == 0, 1, abs(s))
#         else:
#             sgn = np.sign(s)
#         s = abs(s)
#     out = np.log(s)

#     if not keepdims:
#         a_max = np.squeeze(a_max, axis=axis)
#     out += a_max

#     if return_sign:
#         return out, sgn
#     else:
#         return out

psi = scipy.special.psi

# def psi(z):
#     try:
#         import cmath
#         import math
#     except ImportError:
#         raise ImportError("The 'cmath' module is required for complex number support.")

#     # Helper function to compute psi for a single value
#     def psi_single(z):
#         # Euler-Mascheroni constant
#         gamma_const = 0.57721566490153286060651209

#         # Handle negative values using the reflection formula
#         if z.real < 0:
#             return psi_single(1 - z) - cmath.pi / cmath.tan(cmath.pi * z)

#         # For small z, use recurrence to shift to a larger value
#         result = 0.0 + 0.0j
#         while z.real < 10:
#             result -= 1.0 / z
#             z += 1.0

#         # Use the asymptotic expansion for large z
#         inv_z = 1.0 / z
#         inv_z2 = inv_z * inv_z

#         # Coefficients of the asymptotic expansion (Bernoulli numbers)
#         coef = [
#             -1 / 12,
#             1 / 120,
#             -1 / 252,
#             1 / 240,
#             -1 / 132,
#             691 / 32760,
#             -1 / 12,
#         ]

#         # Compute the series sum
#         series = 0.0 + 0.0j
#         inv_z_pow = inv_z2
#         for c in coef:
#             series += c * inv_z_pow
#             inv_z_pow *= inv_z2

#         result += cmath.log(z) - (0.5 * inv_z) + series
#         # if no imaginary part, return real part
#         if result.imag == 0:
#             return result.real
#         return result

#     # Check if z is array-like (list or tuple)
#     if isinstance(z, (list, tuple, np.ndarray)):
#         return np.array([psi_single(zi) for zi in z])
#     else:
#         return psi_single(z)

slogdet = np.linalg.slogdet

# def slogdet(a):
#     import math

#     # Compute the determinant using the available det function
#     d = np.linalg.det(a)

#     # Handle zero determinant
#     if d == 0:
#         sign = 0
#         logabsdet = float("-inf")  # Logarithm of zero is negative infinity
#     else:
#         # Compute the sign of the determinant
#         sign = d / abs(d)
#         # Compute the logarithm of the absolute value of the determinant
#         logabsdet = math.log(abs(d))

#     return (sign, logabsdet)


# def main():
#     z = [-3 + 4j, -1 + 2j]
#     print(psi(z))
#     # print(psi(z + 1) - 1 / z)
#     print(spy.special.psi(z))
#     # print(scipy.special.psi(z + 1) - 1 / z)


# if __name__ == "__main__":
#     main()
