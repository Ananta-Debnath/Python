import numpy as np


def readable_time_ticks(time_values, max_labels=18):
    if len(time_values) <= max_labels:
        return time_values

    step = int(np.ceil(len(time_values) / max_labels))
    ticks = time_values[::step]

    if ticks[-1] != time_values[-1]:
        ticks.append(time_values[-1])

    return ticks


class DiscreteSignal:
    """Finite discrete-time signal with integer indices."""

    # Create a finite discrete-time signal over the given integer range.
    def __init__(self, start_time, end_time):
        self.n = np.arange(start_time, end_time+1)
        self.x = np.zeros_like(self.n, dtype=float)

    # Return the number of stored samples in the signal.
    def __len__(self):
        return len(self.n)

    # Return the integer time indices covered by the signal.
    def times(self):
        return self.n.copy()

    # Return the signal value at the given time index.
    def get_value_at_time(self, t):
        idx = t - self.n[0]
        if idx < 0 or idx >= len(self.n):
            raise ValueError(f"Time {t} is outside the signal.")
        return self.x[idx]

    # Set the signal value at the given time index.
    def set_value_at_time(self, t, value):
        idx = t - self.n[0]
        if idx < 0 or idx >= len(self.n):
            raise ValueError(f"Time {t} is outside the signal.")
        self.x[idx] = value

    # Return a shifted copy of the signal.
    def shift(self, k):
        result = DiscreteSignal(self.n[0]+k, self.n[-1]+k)
        result.x = self.x.copy()

        return result

    # Return the sum of this signal and another signal.
    def add(self, other):
        start = min(self.n[0], other.n[0])
        end = max(self.n[-1], other.n[-1])

        result = DiscreteSignal(start, end)

        # Add values from the first signal
        for i, n in enumerate(self.n):
            result.x[n - start] += self.x[i]

        # Add values from the second signal
        for i, n in enumerate(other.n):
            result.x[n - start] += other.x[i]

        return result

    # Return a scaled copy of the signal.
    def multiply(self, scalar):
        result = DiscreteSignal(self.n[0], self.n[-1])
        result.x = scalar * self.x

        return result

    # Return the nonzero samples of the signal.
    def nonzero_samples(self, tolerance=1e-12):
        mask = np.abs(self.x) > tolerance
        return self.n[mask], self.x[mask]

    def plot(self, title, save_path=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        time_values = list(self.times())
        markerline, stemlines, baseline = ax.stem(time_values, self.values)
        markerline.set_markersize(6)
        baseline.set_color("black")
        baseline.set_linewidth(1)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.35)
        ax.set_xticks(readable_time_ticks(time_values))
        ax.tick_params(axis="x", labelsize=9)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return ax


class LTISystem:
    """Discrete-time LTI system described by a finite impulse response."""

    # Store the impulse response that defines the LTI system.
    def __init__(self, impulse_response):
        if not isinstance(impulse_response, DiscreteSignal):
            raise TypeError("impulse_response must be a DiscreteSignal")
        
        self.h = impulse_response.copy()

    # Return the output time range for the convolution result.
    def output_range(self, input_signal):
        raise NotImplementedError("Complete output_range")

    # Return all shifted and scaled impulse-response components for the input.
    def get_response_components(self, input_signal):
        raise NotImplementedError("Complete get_response_components")

    # Return the system output using superposition of response components.
    def output_by_superposition(self, input_signal):
        raise NotImplementedError("Complete output_by_superposition")

    # Return the nonzero product terms that contribute to one output sample.
    def get_contributions_at_time(self, input_signal, n):
        raise NotImplementedError("Complete get_contributions_at_time")

    # Return one output sample of the LTI system.
    def output_at_time(self, input_signal, n):
        raise NotImplementedError("Complete output_at_time")

    # Return the complete output signal of the LTI system.
    def output(self, input_signal):
        raise NotImplementedError("Complete output")
