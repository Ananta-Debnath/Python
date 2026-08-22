import subprocess
import struct
import random
import re
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Change these to your actual paths.
LOGISIM_JAR = Path(r"D:\Ananta\Academic\BUET\L2-T2\CSE210\Logisim\Logisim-Executables\logisim-generic-2.7.1.jar")
# "D:\Ananta\Academic\BUET\L2-T2\CSE210\Logisim\Logisim-Executables\logisim-generic-2.7.1.jar"
CIRCUIT_FILE = Path(r"D:\Ananta\Academic\BUET\L2-T2\CSE210\Assignment2\FPA_Test.circ")

# IMPORTANT:
# This must be the name of the top-level circuit in Logisim.
CIRCUIT_NAME = "main"

TEST_VECTOR_FILE = Path("fpa_tests.txt")

NUM_RANDOM_TESTS = 1000


# ============================================================
# BINARY16 <-> PYTHON FLOAT
# ============================================================

def float_to_half_bits(x):
    """
    Python float -> IEEE-754 binary16 bit pattern.
    """
    return struct.unpack(
        "<H",
        struct.pack("<e", x)
    )[0]


def half_bits_to_float(bits):
    """
    IEEE-754 binary16 bit pattern -> Python float.
    """
    return struct.unpack(
        "<e",
        struct.pack("<H", bits)
    )[0]


def bits16(x):
    return format(x & 0xFFFF, "016b")


# ============================================================
# BINARY16 INFORMATION
# ============================================================

def decode_half(bits):
    """
    Extract sign, exponent and fraction.
    """

    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF

    return sign, exponent, fraction


# ============================================================
# EXPECTED RESULT
# ============================================================

def expected_result(a_bits, b_bits):
    """
    Calculate expected binary16 result.
    """

    a = half_bits_to_float(a_bits)
    b = half_bits_to_float(b_bits)

    result = a + b

    # --------------------------------------------------------
    # Binary16 limits
    # --------------------------------------------------------

    MAX_NORMAL = 65504.0
    MIN_NORMAL = 2 ** -14

    # --------------------------------------------------------
    # Overflow
    # --------------------------------------------------------

    if result > MAX_NORMAL:
        return 0x7C00, 1, 0

    if result < -MAX_NORMAL:
        return 0xFC00, 1, 0

    # --------------------------------------------------------
    # Underflow
    #
    # For your current design, we're treating values below
    # the minimum NORMAL value as underflow.
    # --------------------------------------------------------

    underflow = (
        result != 0
        and abs(result) < MIN_NORMAL
    )

    # --------------------------------------------------------
    # Normal/subnormal/zero result
    # --------------------------------------------------------

    result_bits = float_to_half_bits(result)

    return result_bits, 0, int(underflow)

# ============================================================
# TEST CASE GENERATION
# ============================================================

def add_test(tests, a_bits, b_bits):
    """
    Add one test case.
    """

    result, overflow, underflow = expected_result(
        a_bits,
        b_bits
    )

    tests.append(
        (
            a_bits,
            b_bits,
            result,
            overflow,
            underflow
        )
    )


def generate_tests():
    tests = []

    # --------------------------------------------------------
    # Basic tests
    # --------------------------------------------------------

    values = [
        0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        10.0,
        -10.0,
        100.0,
        -100.0,
        65504.0,
        -65504.0,
        2 ** -14,
        -(2 ** -14),
        2 ** -15,
        -(2 ** -15),
    ]

    # for a in values:
    #     for b in values:
    #         add_test(
    #             tests,
    #             float_to_half_bits(a),
    #             float_to_half_bits(b)
    #         )

    # --------------------------------------------------------
    # Random binary16 values
    #
    # Generate the actual 16-bit patterns rather than random
    # Python floats. This is important because your circuit
    # accepts 16-bit binary floats.
    # --------------------------------------------------------

    for _ in range(NUM_RANDOM_TESTS):

        while True:
            a = random.randint(0, 0xFFFF)

            exponent = (a >> 10) & 0x1F
            mantissa = a & 0x03FF

            if exponent not in (0x00, 0x1F) and mantissa != 0:
                break

        while True:
            b = random.randint(0, 0xFFFF)

            exponent = (b >> 10) & 0x1F
            mantissa = b & 0x03FF

            if exponent not in (0x00, 0x1F) and mantissa != 0:
                break

        add_test(tests, a, b)

    # --------------------------------------------------------
    # Specific normalization / exponent-difference cases
    # --------------------------------------------------------

    special_values = [
        1.0,
        1.5,
        1.25,
        1.75,
        2.0,
        2 ** -1,
        2 ** -2,
        2 ** -10,
        2 ** -14,
        2 ** -15,
        2 ** -20,
        2 ** -24,
        65504.0,
    ]

    # for a in special_values:
    #     for b in special_values:
    #         add_test(
    #             tests,
    #             float_to_half_bits(a),
    #             float_to_half_bits(b)
    #         )

    #         # Also subtraction through opposite sign.
    #         add_test(
    #             tests,
    #             float_to_half_bits(a),
    #             float_to_half_bits(-b)
    #         )

    return tests


# ============================================================
# WRITE LOGISIM TEST VECTOR
# ============================================================

def write_test_vector(tests):

    with open(TEST_VECTOR_FILE, "w") as f:

        # f.write("# FPA automated test vector\n")
        # f.write("# A[16] B[16] -> Sum[16], Overflow, Underflow\n")
        # f.write("\n")

        # # Logisim requires the widths on multi-bit signals.
        # f.write(
        #     "A[16] B[16] Sum[16] Overflow Underflow\n"
        # )

        for a, b, result, overflow, underflow in tests:

            f.write(
                f"{bits16(a)} "
                f"{bits16(b)} "
                f"{bits16(result)} "
                f"{overflow} "
                f"{underflow}\n"
            )


# ============================================================
# RUN LOGISIM
# ============================================================

def run_logisim():

    if not LOGISIM_JAR.exists():
        print("ERROR: Logisim JAR not found:")
        print(LOGISIM_JAR)
        return False

    if not CIRCUIT_FILE.exists():
        print("ERROR: Circuit file not found:")
        print(CIRCUIT_FILE)
        return False

    command = [
        "java",
        "-jar",
        str(LOGISIM_JAR),

        # Logisim test-vector command
        "-w",

        CIRCUIT_NAME,
        str(TEST_VECTOR_FILE),
        str(CIRCUIT_FILE)
    ]

    print()
    print("Running Logisim...")
    print()

    process = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    print("---------- LOGISIM OUTPUT ----------")
    print(process.stdout)

    if process.stderr:
        print("---------- LOGISIM ERRORS ----------")
        print(process.stderr)

    # --------------------------------------------------------
    # Try to extract number of passed/failed tests.
    # Different 2.7.1 builds use slightly different wording.
    # --------------------------------------------------------

    output = process.stdout + process.stderr

    match = re.search(
        r"Passed\s*:\s*(\d+).*?(?:Failed|Error)\s*:\s*(\d+)",
        output,
        re.IGNORECASE | re.DOTALL
    )

    if match:

        passed = int(match.group(1))
        failed = int(match.group(2))

        print()
        print("==============================")
        print("       FPA TEST RESULTS")
        print("==============================")
        print(f"Passed : {passed}")
        print(f"Failed : {failed}")
        print("==============================")

        if failed == 0:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")

    else:

        print()
        print("Could not parse Logisim's summary.")
        print("Check the Logisim output above.")

    return process.returncode == 0


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Generating tests...")

    tests = generate_tests()

    print(f"Generated {len(tests)} tests.")

    write_test_vector(tests)

    print()
    print(f"Test vector written to:")
    print(TEST_VECTOR_FILE.absolute())

    # run_logisim()