"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:

    import numpy as np
    from transforms import DFTAnalyzer, FFTTransformer
    x = np.random.randn(64) + 1j * np.random.randn(64)
    d, f = DFTAnalyzer(), FFTTransformer()
    assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
    assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
"""

import numpy as np


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    # TODO: implement this function

    power = 1
    while power < n:
        power *= 2
    return power


class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        # TODO: implement this method

        x = np.asarray(x, dtype=np.complex128)
        N = len(x)

        X = np.zeros(N, dtype=np.complex128)

        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

        return X

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        # TODO: implement this method

        spectrum = np.asarray(spectrum, dtype=np.complex128)
        N = len(spectrum)

        x = np.zeros(N, dtype=np.complex128)

        for n in range(N):
            for k in range(N):
                x[n] += spectrum[k] * np.exp(2j * np.pi * k * n / N)

            x[n] /= N

        return x


class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    name = "fft"

    def transform(self, x):
        """Forward FFT. Same contract as DFTAnalyzer.transform."""
        # TODO: implement this method

        x = np.asarray(x, dtype=np.complex128)
        N = len(x)

        if N == 0 or (N & (N - 1)) != 0:
            raise ValueError("FFT length must be a power of two")

        j = 0
        for i in range(1, N):
            bit = N >> 1

            while j & bit:
                j ^= bit
                bit >>= 1

            j ^= bit

            if i < j:
                x[i], x[j] = x[j], x[i]

        size = 2

        while size <= N:
            half = size // 2

            twiddles = np.exp(-2j * np.pi * np.arange(half) / size)

            for start in range(0, N, size):
                for k in range(half):
                    even = start + k
                    odd = even + half

                    t = twiddles[k] * x[odd]

                    x[odd] = x[even] - t
                    x[even] = x[even] + t

            size *= 2

        return x

    def inverse(self, spectrum):
        """Inverse FFT, including the 1/N factor."""
        # TODO: implement this method

        spectrum = np.asarray(spectrum, dtype=np.complex128)

        x = np.conjugate(spectrum)
        x = self.transform(x)
        x = np.conjugate(x)

        return x / len(x)


# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"

    def transform(self, x):
        # TODO (bonus): implement this method

        x = np.asarray(x, dtype=np.complex128)
        N = len(x)

        if N == 0:
            return np.array([], dtype=np.complex128)

        M = next_power_of_two(2 * N - 1)

        n = np.arange(N)

        chirp = np.exp(-1j * np.pi * n**2 / N)

        a = x * chirp

        b = np.zeros(M, dtype=np.complex128)

        b[:N] = np.exp(1j * np.pi * n**2 / N)

        for k in range(1, N):
            b[M - k] = b[k]

        fft = FFTTransformer()
        A = fft.transform(np.pad(a, (0, M - N)))
        B = fft.transform(b)

        convolution = fft.inverse(A * B)

        result = convolution[:N] * chirp

        return result

    def inverse(self, spectrum):
        # TODO (bonus): implement this method

        spectrum = np.asarray(spectrum, dtype=np.complex128)

        if len(spectrum) == 0:
            return np.array([], dtype=np.complex128)

        return np.conjugate(
            self.transform(np.conjugate(spectrum))
        ) / len(spectrum)


class NTTTransformer(FFTTransformer):
    """
    Radix-2 Number Theoretic Transform.

    Uses:
        p = 998244353
        primitive root = 3
    """

    name = "ntt"

    MOD = 998244353
    ROOT = 3

    def _is_power_of_two(self, n):
        return n > 0 and (n & (n - 1)) == 0

    def _bit_reverse(self, values):
        """Perform an in-place bit-reversal permutation."""
        n = len(values)
        j = 0

        for i in range(1, n):
            bit = n >> 1

            while j & bit:
                j ^= bit
                bit >>= 1

            j ^= bit

            if i < j:
                values[i], values[j] = values[j], values[i]

    def _ntt(self, values, inverse=False):
        """
        Radix-2 NTT butterfly machinery.

        The same function is used for both forward and inverse transforms.
        """
        n = len(values)

        self._bit_reverse(values)

        if inverse:
            root = pow(self.ROOT, self.MOD - 2, self.MOD)
        else:
            root = self.ROOT

        length = 2

        while length <= n:
            half = length // 2

            # Compute this stage's twiddle factors once.
            wlen = pow(
                root,
                (self.MOD - 1) // length,
                self.MOD
            )

            twiddles = np.empty(half, dtype=np.int64)
            twiddles[0] = 1

            for k in range(1, half):
                twiddles[k] = (
                    twiddles[k - 1] * wlen
                ) % self.MOD

            # Apply the butterflies.
            for start in range(0, n, length):
                for k in range(half):
                    u = values[start + k]

                    v = (
                        values[start + k + half]
                        * twiddles[k]
                    ) % self.MOD

                    values[start + k] = (
                        u + v
                    ) % self.MOD

                    values[start + k + half] = (
                        u - v
                    ) % self.MOD

            length *= 2

        if inverse:
            inv_n = pow(n, self.MOD - 2, self.MOD)

            for i in range(n):
                values[i] = (
                    values[i] * inv_n
                ) % self.MOD

        return values

    def transform(self, x):
        """Forward NTT."""
        x = np.asarray(x, dtype=np.int64)

        n = len(x)

        if not self._is_power_of_two(n):
            raise ValueError(
                "NTT length must be a power of two"
            )

        values = np.array(
            x % self.MOD,
            dtype=np.int64,
            copy=True
        )

        return self._ntt(values, inverse=False)

    def inverse(self, spectrum):
        """Inverse NTT, including the 1/N factor."""
        spectrum = np.asarray(
            spectrum,
            dtype=np.int64
        )

        n = len(spectrum)

        if not self._is_power_of_two(n):
            raise ValueError(
                "NTT length must be a power of two"
            )

        values = np.array(
            spectrum % self.MOD,
            dtype=np.int64,
            copy=True
        )

        return self._ntt(values, inverse=True)