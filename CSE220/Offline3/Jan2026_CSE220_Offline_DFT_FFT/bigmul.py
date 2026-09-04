"""
bigmul.py  --  TASK A: multiplying huge integers with your own transform.

YOUR CODE GOES HERE. The provided modules io_utils.py (file reading, random
operands) and bench_utils.py (timing, runtime plots) are already imported;
everything mathematical is yours.

Usage (the command line is already wired up for you):

    python3 bigmul.py inputs/1.txt --engine fft --out-dir outputs/1
    python3 bigmul.py inputs/3.txt --engine dft --out-dir outputs/3
    python3 bigmul.py --benchmark --out-dir outputs/benchmark

Restrictions: no numpy.fft / scipy.fft / numpy.convolve / scipy.signal, and
no Python big-integer multiplication of the operands themselves. Python's
own integers may be used ONLY to check your answer at the end.
"""

import argparse
import os
import sys

import numpy as np

from bench_utils import plot_runtime_curve, time_best, timing_table_lines
from io_utils import random_decimal, read_operands, write_report, write_text
from transforms import DFTAnalyzer, FFTTransformer, next_power_of_two, ArbitraryLengthFFT

# Python 3.11+ refuses to print integers longer than 4300 digits unless this
# limit is raised, and the verification step below prints one.
sys.set_int_max_str_digits(2_000_000)

# Number of decimal digits packed into one "limb" (one polynomial
# coefficient). The specification explains why 4 is a safe choice and what
# breaks if you raise it too far.
BASE_DIGITS = 4
BASE = 10 ** BASE_DIGITS


def to_limbs(text, base_digits=BASE_DIGITS):
    """
    Convert a decimal string into polynomial coefficients.

    "123456789" with base_digits = 4 becomes the little-endian limb array
    [6789, 2345, 1] -- that is, 1*BASE^2 + 2345*BASE^1 + 6789*BASE^0.

    Parameters
    ----------
    text : str
        A decimal integer, possibly with a leading '+' or '-'.
    base_digits : int
        Decimal digits per limb.

    Returns
    -------
    (int, numpy.ndarray)
        The sign (+1 or -1) and the little-endian limb array (dtype int64).
        Handle the sign separately from the magnitude: the transform never
        sees it.
    """
    # TODO: implement this function

    text = text.strip()

    sign = 1

    if text.startswith("+"):
        text = text[1:]
    elif text.startswith("-"):
        sign = -1
        text = text[1:]

    if not text or not text.isdigit():
        raise ValueError("text must be a decimal integer")

    text = text.lstrip("0")

    if not text:
        return 1, np.array([0], dtype=np.int64)

    limbs = []

    for end in range(len(text), 0, -base_digits):
        start = max(0, end - base_digits)
        limbs.append(int(text[start:end]))

    return sign, np.array(limbs, dtype=np.int64)


def from_limbs(sign, limbs, base_digits=BASE_DIGITS):
    """
    Convert limbs back into a decimal string, propagating carries.

    The limbs handed to this function are the convolution result, so they are
    NOT yet reduced: an entry may be far larger than BASE. Sweep from the
    least significant limb upwards, carrying the overflow into the next one,
    then strip leading zeros and re-attach the sign.

    Returns
    -------
    str
        The decimal representation. "0" must come out as "0", not "-0" or "".
    """
    # TODO: implement this function

    base = 10 ** base_digits

    limbs = [int(x) for x in limbs]

    if not limbs:
        return "0"

    for i in range(len(limbs) - 1):
        carry, limbs[i] = divmod(limbs[i], base)
        limbs[i + 1] += carry

    while limbs[-1] >= base:
        carry, limbs[-1] = divmod(limbs[-1], base)
        limbs.append(carry)

    while len(limbs) > 1 and limbs[-1] == 0:
        limbs.pop()

    if len(limbs) == 1 and limbs[0] == 0:
        return "0"

    result = str(limbs[-1])

    for limb in reversed(limbs[:-1]):
        result += f"{limb:0{base_digits}d}"

    if sign < 0:
        result = "-" + result

    return result


def multiply_transform(a, b, engine):
    """
    Multiply two limb arrays through the frequency domain.

    The product of two polynomials is the LINEAR convolution of their
    coefficients, and the transform gives you convolution as a pointwise
    product -- but a DFT of length N gives you CIRCULAR convolution of period
    N. Choose N large enough that the linear result fits, or the high-order
    coefficients wrap around and silently corrupt the answer.

    Steps:
      1. Choose the transform length N (see above; for FFTTransformer it must
         also be a power of two -- next_power_of_two is in transforms.py).
      2. Zero-pad both limb arrays to length N.
      3. Transform both, multiply the two spectra pointwise, inverse-transform.
      4. The result is real up to rounding error: take the real part and round
         to the nearest integer.

    Parameters
    ----------
    a, b : numpy.ndarray of int64
        Little-endian limb arrays.
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    (numpy.ndarray of int64, int)
        The un-carried convolution coefficients, and the transform length N
        you used (report.txt has to state it).
    """
    # TODO: implement this function

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    required_length = len(a) + len(b) - 1

    if isinstance(engine, FFTTransformer):
        N = next_power_of_two(required_length)
    else:
        N = required_length

    a_padded = np.pad(a, (0, N - len(a)))
    b_padded = np.pad(b, (0, N - len(b)))

    A = engine.transform(a_padded)
    B = engine.transform(b_padded)

    C = A * B

    result = engine.inverse(C)

    result = np.rint(result.real).astype(np.int64)

    result = result[:required_length]

    return result, N


def multiply_schoolbook(a, b):
    """
    OPTIONAL baseline: the O(n^2) method everyone learns at school, on limbs.

    Only needed if you want a third curve on your runtime plot. Expect it to
    be competitive for a long time: a NumPy-assisted schoolbook multiply has a
    very small constant factor, and constant factors decide who wins at small
    sizes.
    """
    # TODO (optional): implement this function

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    result = np.zeros(len(a) + len(b) - 1, dtype=np.int64)

    for i in range(len(a)):
        result[i:i + len(b)] += a[i] * b

    return result


def multiply(text_a, text_b, method):
    """
    Multiply two decimal strings and return (product_string, N, limbs_a, limbs_b).

    ``method`` is one of "dft", "fft", "schoolbook" (optional) or "arbitrary"
    (bonus). Pick the engine, convert to limbs, convolve, carry, re-sign.
    """
    # TODO: implement this function

    sign_a, limbs_a = to_limbs(text_a)
    sign_b, limbs_b = to_limbs(text_b)

    sign = sign_a * sign_b

    if method == "dft":
        engine = DFTAnalyzer()
        result, N = multiply_transform(limbs_a, limbs_b, engine)

    elif method == "fft":
        engine = FFTTransformer()
        result, N = multiply_transform(limbs_a, limbs_b, engine)

    elif method == "arbitrary":
        engine = ArbitraryLengthFFT()
        result, N = multiply_transform(limbs_a, limbs_b, engine)

    elif method == "schoolbook":
        result = multiply_schoolbook(limbs_a, limbs_b)
        N = len(limbs_a) + len(limbs_b) - 1

    else:
        raise ValueError(f"Unknown multiplication method: {method}")

    product = from_limbs(sign, result)

    return product, N, limbs_a, limbs_b


def run_single(path, method, out_dir):
    """
    Process one input file and write the required outputs.

    Must produce, inside ``out_dir``:
      product.txt -- the product as a single decimal string
      report.txt  -- input path, method, digit counts of both operands, the
                     base used, limb counts, the transform length N, the digit
                     count of the product, and the verification verdict.
                     It is written by your code; there is no separate
                     write-up to hand in.

    Verification: compare your product against ``int(text_a) * int(text_b)``.
    This is the ONLY place Python's big integers may be used. Print MATCH or
    MISMATCH; a MISMATCH must not be silently swallowed.
    """
    # TODO: implement this function

    os.makedirs(out_dir, exist_ok=True)

    with open(path, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if len(lines) != 2:
        raise ValueError("Input file must contain exactly two integers")

    text_a = lines[0]
    text_b = lines[1]

    product, N, limbs_a, limbs_b = multiply(text_a, text_b, method)

    expected = str(int(text_a) * int(text_b))

    if product == expected:
        verdict = "MATCH"
    else:
        verdict = "MISMATCH"

    print(verdict)

    with open(os.path.join(out_dir, "product.txt"), "w", encoding="utf-8") as f:
        f.write(product + "\n")

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("Task A -- big-integer multiplication by spectral convolution\n")
        f.write(f"input file          : {path}\n")
        f.write(f"method              : {method}\n")
        f.write(
            f"digits of A / B     : "
            f"{len(text_a.lstrip('+-'))} / {len(text_b.lstrip('+-'))}\n"
        )
        f.write(f"base                : 10^{BASE_DIGITS}\n")
        f.write(
            f"limbs of A / B      : "
            f"{len(limbs_a)} / {len(limbs_b)}\n"
        )
        f.write(f"transform length N  : {N}\n")
        f.write(f"digits of product   : {len(product.lstrip('-'))}\n")
        f.write(f"verification        : {verdict}\n")

    if verdict == "MISMATCH":
        print(f"MISMATCH: expected {expected}, got {product}")


# ---------------------------------------------------------------------------
# PROVIDED -- run_benchmark is already written. It calls your multiply(), so
# it starts working as soon as your transforms and multiply() are correct.
# You do not need to modify anything below (though you may extend it).
# ---------------------------------------------------------------------------
DFT_SIZES = [128, 256, 512, 1024, 2048, 4096]
FFT_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
TIME_BUDGET = 20.0          # stop a sweep once one measurement exceeds this


def run_benchmark(out_dir):
    """
    Measure naive DFT against radix-2 FFT over growing operands and write
    runtime_bigmul.png plus the timing table in report.txt.

    Each sweep stops early once a single measurement exceeds TIME_BUDGET
    seconds, so a slow machine (or a slow DFT) simply produces a shorter
    curve rather than hanging.
    """
    def measure(label, method, sizes):
        xs, ys = [], []
        print("%s:" % label)
        for digits in sizes:
            a = random_decimal(digits, seed=digits)
            b = random_decimal(digits, seed=digits + 1)
            seconds = time_best(lambda: multiply(a, b, method), repeats=2)
            xs.append(digits)
            ys.append(seconds)
            print("  %8d digits   %9.4f s" % (digits, seconds))
            if seconds > TIME_BUDGET:
                print("  (stopping this curve -- over the time budget)")
                break
        return xs, ys

    series = {}
    series["Naive DFT"] = measure("naive DFT", "dft", DFT_SIZES)
    series["Radix-2 FFT"] = measure("radix-2 FFT", "fft", FFT_SIZES)
    try:                                    # optional third curve
        series["Schoolbook"] = measure("schoolbook", "schoolbook", FFT_SIZES)
    except NotImplementedError:
        series.pop("Schoolbook", None)
        print("schoolbook: not implemented, skipping that curve")

    plot_path = os.path.join(out_dir, "runtime_bigmul.png")
    plot_runtime_curve(series, plot_path,
                       title="Task A: big-integer multiplication",
                       xlabel="decimal digits per operand",
                       references=("n2", "nlogn"))
    write_report(os.path.join(out_dir, "report.txt"),
                 ["Task A -- runtime benchmark", ""]
                 + timing_table_lines(series, size_label="digits")
                 + ["", "plot: %s" % os.path.basename(plot_path)])
    print("wrote", plot_path)


def main():
    ap = argparse.ArgumentParser(description="Big-integer multiplication by DFT/FFT")
    ap.add_argument("input", nargs="?", help="input file with the two operands")
    ap.add_argument("--engine", default="fft",
                    choices=["dft", "fft", "schoolbook", "arbitrary"])
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--benchmark", action="store_true",
                    help="run the timing study instead of a single multiplication")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.benchmark:
        run_benchmark(args.out_dir)
    else:
        if not args.input:
            ap.error("an input file is required unless --benchmark is given")
        run_single(args.input, args.engine, args.out_dir)


if __name__ == "__main__":
    main()


# python bigmul.py inputs/1.txt --engine dft --out-dir outputs/task_a/1
# python bigmul.py inputs/2.txt --engine dft --out-dir outputs/task_a/2
# python bigmul.py inputs/3.txt --engine fft --out-dir outputs/task_a/3
# python bigmul.py inputs/4.txt --engine fft --out-dir outputs/task_a/4
# python bigmul.py --benchmark --out-dir outputs/task_a/benchmark