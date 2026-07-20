import numpy as np
import matplotlib.pyplot as plt

class DiscreteSignal:
    def __init__(self, start_time, end_time):
        self.t = np.arange(start_time, end_time+1)
        self.x = np.zeros_like(self.t, dtype=float)

    def set_value_at_time(self, t, value):
        idx = self.t.searchsorted(t)
        if t != self.t[idx]:
            print("Wrong time value")
        else:
            self.x[idx] = value
    
    def get_value_at_time(self, t):
        idx = self.t.searchsorted(t)
        x = 0
        if t != self.t[idx]:
            print("Wrong time value")
        else:
            x = self.x[idx]
        
        return x
    
    def shift(self, k):
        result = DiscreteSignal(self.t[0]+k, self.t[-1]+k)
        result.x = self.x.copy()

        return result
    
    def add(self, sig: DiscreteSignal):
        start = min(self.t[0], sig.t[0])
        end = max(self.t[-1], sig.t[-1])

        result = DiscreteSignal(start, end)

        # Add values from the first signal
        for i, n in enumerate(self.t):
            result.x[n - start] += self.x[i]

        # Add values from the second signal
        for i, n in enumerate(sig.t):
            result.x[n - start] += sig.x[i]

        return result
    
    def multiply(self, m):
        result = DiscreteSignal(self.t[0], self.t[-1])
        result.x = m*self.x

        return result
    
    def plot(self, title, save_path=None):
        plt.figure(figsize=(6, 4))

        plt.stem(self.t, self.x, basefmt="k-")
        plt.title(title)
        plt.xlabel("n")
        plt.ylabel("x[n]")
        plt.grid(True)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()



# Create original signal
x = DiscreteSignal(-2, 2)
x.set_value_at_time(-2, 1)
x.set_value_at_time(0, 2)
x.set_value_at_time(2, 3)

# Operations
x_shift = x.shift(2)
x_add = x.add(x_shift)
x_mul = x.multiply(2)

# Plot
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

plots = [
    (x, "Original Signal"),
    (x_shift, "Shifted Right by 2"),
    (x_add, "Original + Shifted"),
    (x_mul, "Original × 2")
]

for ax, (sig, title) in zip(axs.flat, plots):
    ax.stem(sig.t, sig.x, basefmt="k-")
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("x[n]")
    ax.grid(True)

plt.tight_layout()
plt.show()



class LTISystem:
    def __init__(self, impulse_response):
        if not isinstance(impulse_response, DiscreteSignal):
            raise TypeError("impulse_response must be a DiscreteSignal")
        
        self.h = impulse_response

    def get_response_components(self, input_signal: DiscreteSignal):
        return None
    
    def output_by_superposition(self, input_signal: DiscreteSignal):
        return None
    
    def get_contributions_at_time(self, input_signal: DiscreteSignal, n):
        return None
    
    def output_at_time(self, input_signal: DiscreteSignal, n):
        return None
    
    def output(self, input_signal: DiscreteSignal):
        return None