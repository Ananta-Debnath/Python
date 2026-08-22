import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
import sys


# =====================================================================
# Given classes — paste your Task 2 implementations where indicated
# =====================================================================

class ContinuousImage:
    """Represents a grayscale image as a continuous 2D spatial signal. (Given)"""

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L').astype(float)
        self.image = self.image / np.max(self.image)
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])


class CFT2D:
    """2D Continuous Fourier Transform. (Given — paste your Task 2 solution)"""

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        self.u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), self.I.shape[1])
        self.v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), self.I.shape[0])

    def compute_cft(self):
        # Calculate real part
        
        # cos(2πux) and sin(2πux)
        cos_ux = np.cos(2 * np.pi * self.u[:, None] * self.x[None, :])

        sin_ux = np.sin(2 * np.pi * self.u[:, None] * self.x[None, :])

        # Multiply by image
        A_real = self.I[None, :, :] * cos_ux[:, None, :]
        B_real = self.I[None, :, :] * sin_ux[:, None, :]

        # Integrate over x
        A_real = np.trapezoid(A_real, self.x, axis=2)
        B_real = np.trapezoid(B_real, self.x, axis=2)

        # cos(2πvy) and sin(2πvy)
        cos_vy = np.cos(2 * np.pi * self.v[:, None] * self.y[None, :])

        sin_vy = np.sin(2 * np.pi * self.v[:, None] * self.y[None, :])

        # We need (u, v, y)
        A_real = A_real[:, None, :] * cos_vy[None, :, :]
        B_real = B_real[:, None, :] * sin_vy[None, :, :]

        # Integrate over y
        real = np.trapezoid(A_real - B_real, self.y, axis=2)

        # Calculate imaginary part

        # sin(2πux) and cos(2πux)
        A_img = self.I[None, :, :] * sin_ux[:, None, :]
        B_img = self.I[None, :, :] * cos_ux[:, None, :]

        # Integrate over x
        A_img = np.trapezoid(A_img, self.x, axis=2)
        B_img = np.trapezoid(B_img, self.x, axis=2)

        # Multiply by cos(2πvy) and sin(2πvy)
        A_img = A_img[:, None, :] * cos_vy[None, :, :]
        B_img = B_img[:, None, :] * sin_vy[None, :, :]

        # Integrate over y
        imag = -np.trapezoid(A_img + B_img, self.y, axis=2)

        return real, imag

    def plot_magnitude(self):
        magnitude = np.sqrt(self.real**2 + self.imag**2)
        log_magnitude = np.log(1 + magnitude)
        plt.imshow(log_magnitude, cmap='gray')


class InverseCFT2D:
    """Inverse 2D-CFT. (Given — paste your Task 2 solution)"""

    def __init__(self, real, imag, u, v, x, y):
        self.real = real
        self.imag = imag
        self.u = u
        self.v = v
        self.x = x
        self.y = y

    def reconstruct(self):
        # Calculate real part
        cos_ux = np.cos(2 * np.pi * self.x[:, None] * self.u[None, :])

        sin_ux = np.sin(2 * np.pi * self.x[:, None] * self.u[None, :])

        A_real = (self.real[None, :, :] * cos_ux[:, :, None])

        B_real = (self.real[None, :, :] * sin_ux[:, :, None])

        # Calculate imaginary part
        A_img = (self.imag[None, :, :] * sin_ux[:, :, None])

        B_img = (self.imag[None, :, :] * cos_ux[:, :, None])

        # Integrate over u
        A_real = np.trapezoid(A_real, self.u, axis=1)
        B_real = np.trapezoid(B_real, self.u, axis=1)

        A_img = np.trapezoid(A_img, self.u, axis=1)
        B_img = np.trapezoid(B_img, self.u, axis=1)

        cos_vy = np.cos(2 * np.pi * self.y[:, None] * self.v[None, :])

        sin_vy = np.sin(2 * np.pi * self.y[:, None] * self.v[None, :])

        A = (A_real - A_img)[:, None, :] * cos_vy[None, :, :]
        B = (B_real + B_img)[:, None, :] * sin_vy[None, :, :]

        # Integrate over v
        real = np.trapezoid(A - B, self.v, axis=2)

        return real.T


# =====================================================================
# Task 1 — band_pass and band_stop filters
# =====================================================================

class FrequencyFilter:

    def high_pass(self, real, imag, cutoff):
        """Given."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - cx) ** 2 + (j - cy) ** 2) <= cutoff:
                    real[i, j] = 0
                    imag[i, j] = 0
        return real, imag

    def band_pass(self, real, imag, r_low, r_high):
        """TODO: retain entries with r_low < d(i,j) <= r_high, zero the rest."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                if not r_low < np.sqrt((i - cx) ** 2 + (j - cy) ** 2) <= r_high:
                    real[i, j] = 0
                    imag[i, j] = 0
        return real, imag

    def band_stop(self, real, imag, r_low, r_high):
        """TODO: zero entries with r_low < d(i,j) <= r_high, retain the rest."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                if r_low < np.sqrt((i - cx) ** 2 + (j - cy) ** 2) <= r_high:
                    real[i, j] = 0
                    imag[i, j] = 0
        return real, imag

    def shift_brightness(self, real, imag, shift_amount):
        """TODO: Task 3. Add shift_amount to the real component of the exact center pixel."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        imag = imag.copy()
        real[cx, cy] += shift_amount
        return real, imag

# =====================================================================
# Task 2 — complementarity check on raw spatial reconstructions
# =====================================================================

class ReconstructionValidator:

    def verify_complementarity(self, I_recon, I_bp, I_bs):
        """TODO: verify the complementarity property. Return (is_valid, delta)."""
        d = np.max(np.abs(I_bp + I_bs - I_recon))
        return d < 1e-9, d


# =====================================================================
# Entry point (given — do not modify)
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 cft_edge_detector.py <input_image>")
        sys.exit(1)

    input_path = sys.argv[1]
    r_low, r_high = 10, 50

    img   = ContinuousImage(input_path)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()

    filt = FrequencyFilter()
    real_bp, imag_bp = filt.band_pass(real, imag, r_low, r_high)
    real_bs, imag_bs = filt.band_stop(real, imag, r_low, r_high)

    def reconstruct(r, im):
        return InverseCFT2D(r, im, cft2d.u, cft2d.v, img.x, img.y).reconstruct()

    I_recon = reconstruct(real,    imag)
    I_bp    = reconstruct(real_bp, imag_bp)
    I_bs    = reconstruct(real_bs, imag_bs)

    validator = ReconstructionValidator()
    is_valid, delta = validator.verify_complementarity(I_recon, I_bp, I_bs)
    print(f"Complementarity check: {is_valid} | max delta: {delta:.2e}")

    def save_edge_map(I_raw, path):
        edge_map = np.abs(I_raw)
        if edge_map.max() > 0:
            edge_map = edge_map / edge_map.max()
        plt.imsave(path, 1 - edge_map, cmap='gray')
        print(f"Saved {path}")

    save_edge_map(I_bp, "pikachu_bandpass.png")
    save_edge_map(I_bs, "pikachu_bandstop.png")

    # Task 3 execution
    real_shifted, imag_shifted = filt.shift_brightness(real, imag, shift_amount=2.0)
    I_brightened = reconstruct(real_shifted, imag_shifted)
    
    # Save brightened image (clip to [0,1], no edge-map inversion)
    I_brightened_clipped = np.clip(I_brightened, 0, 1)
    plt.imsave("pikachu_brightened.png", I_brightened_clipped, cmap='gray')
    print("Saved pikachu_brightened.png")
