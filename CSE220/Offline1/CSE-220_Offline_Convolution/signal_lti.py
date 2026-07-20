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
        # self.times() = np.arange(start_time, end_time+1)
        self.start_time = start_time
        self.end_time = end_time
        len = self.end_time - self.start_time + 1
        self.values = np.zeros(len, dtype=float)

    # Return the number of stored samples in the signal.
    def __len__(self):
        return self.end_time - self.start_time + 1

    # Return the integer time indices covered by the signal.
    def times(self):
        return np.arange(self.start_time, self.end_time + 1)

    # Return the signal value at the given time index.
    def get_value_at_time(self, t):
        idx = t - self.start_time
        if idx < 0 or idx >= len(self):
            raise ValueError(f"Time {t} is outside the signal.")
        return self.values[idx]

    # Set the signal value at the given time index.
    def set_value_at_time(self, t, value):
        idx = t - self.start_time
        if idx < 0 or idx >= len(self):
            raise ValueError(f"Time {t} is outside the signal.")
        self.values[idx] = value

    # Return a shifted copy of the signal.
    def shift(self, k):
        result = DiscreteSignal(self.start_time + k, self.end_time + k)
        result.values = self.values.copy()

        return result

    # Return the sum of this signal and another signal.
    def add(self, other):
        start = min(self.start_time, other.start_time)
        end = max(self.end_time, other.end_time)

        result = DiscreteSignal(start, end)

        # Add values from the first signal
        for i, n in enumerate(self.times()):
            result.values[n - start] += self.values[i]

        # Add values from the second signal
        for i, n in enumerate(other.times()):
            result.values[n - start] += other.values[i]

        return result

    # Return a scaled copy of the signal.
    def multiply(self, scalar):
        result = DiscreteSignal(self.times()[0], self.times()[-1])
        result.values = scalar * self.values

        return result

    # Return the nonzero samples of the signal.
    def nonzero_samples(self, tolerance=1e-12):
        mask = np.abs(self.values) > tolerance
        return zip(self.times()[mask], self.values[mask])

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
        
        self.h = impulse_response.multiply(1.0)  # Woraround for copy

    # Return the output time range for the convolution result.
    def output_range(self, input_signal):
        start_time = self.h.times()[0] + input_signal.times()[0]
        end_time = self.h.times()[-1] + input_signal.times()[-1]

        return start_time, end_time

    # Return all shifted and scaled impulse-response components for the input.
    def get_response_components(self, input_signal):
        nonzero_samples = input_signal.nonzero_samples()
        components = []

        for k, x_k in nonzero_samples:
            shifted_h = self.h.shift(k)
            scaled_h = shifted_h.multiply(x_k)
            components.append(scaled_h)

        return components

    # Return the system output using superposition of response components.
    def output_by_superposition(self, input_signal):
        components = self.get_response_components(input_signal)
        output_signal = DiscreteSignal(*self.output_range(input_signal))

        for component in components:
            output_signal = output_signal.add(component)

        return output_signal

    # Return the nonzero product terms that contribute to one output sample.
    def get_contributions_at_time(self, input_signal, n):
        contributions = []
        for k, x_k in input_signal.nonzero_samples():
            h_index = n - k
            try:
                h_value = self.h.get_value_at_time(h_index)
                y_k = x_k * h_value
                if abs(y_k) > 1e-12:
                    contributions.append(y_k)
            except ValueError:
                continue

        return contributions

    # Return one output sample of the LTI system.
    def output_at_time(self, input_signal, n):
        contributions = self.get_contributions_at_time(input_signal, n)
        return sum(contributions)

    # Return the complete output signal of the LTI system.
    def output(self, input_signal):
        output_signal = DiscreteSignal(*self.output_range(input_signal))
        for n in output_signal.times():
            output_signal.set_value_at_time(n, self.output_at_time(input_signal, n))

        return output_signal
