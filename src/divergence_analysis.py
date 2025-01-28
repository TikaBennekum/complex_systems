import numpy as np
import matplotlib.pyplot as plt

class ExpAnalysis:
    def __init__(self, unperturbed_file, perturbed_file):
        self.unperturbed_data = np.load(unperturbed_file)
        self.perturbed_data = np.load(perturbed_file)

        self.copy_unperturbed_data = self.unperturbed_data.copy()
        self.copy_perturbed_data = self.perturbed_data.copy()

        print("Shape of Unperturbed Data:", self.unperturbed_data.shape)
        print("Shape of Perturbed Data:", self.perturbed_data.shape)

        self.mean_ground_diff = None
        self.mean_water_diff = None
        self.hamming_water = None
        self.hamming_ground = None
        self.lyapunov_exponent_water = None
        self.lyapunov_exponent_ground = None

    def compute_mean_differences(self):
        self.mean_ground_diff = np.mean(
            np.abs(self.perturbed_data[..., GROUND_HEIGHT] - self.unperturbed_data[..., GROUND_HEIGHT]), 
            axis=(1, 2)
        )
        self.mean_water_diff = np.mean(
            np.abs(self.perturbed_data[..., WATER_HEIGHT] - self.unperturbed_data[..., WATER_HEIGHT]), 
            axis=(1, 2)
        )

    def normalize_differences(self):
        max_diff_water = np.max(self.mean_water_diff)
        max_diff_ground = np.max(self.mean_ground_diff)

        normalized_water_diff = self.mean_water_diff / max_diff_water
        normalized_ground_diff = self.mean_ground_diff / max_diff_ground

        return normalized_water_diff, normalized_ground_diff

    def compute_log_differences(self, normalized_water_diff, normalized_ground_diff):
        log_mean_water_diff = np.log(normalized_water_diff + 1e-10)
        log_mean_ground_diff = np.log(normalized_ground_diff + 1e-10)

        return log_mean_water_diff, log_mean_ground_diff

    def fit_linear_model(self, log_mean_water_diff, log_mean_ground_diff):
        time_steps_full = np.arange(len(self.mean_water_diff))

        slope_water, _ = np.polyfit(time_steps_full, log_mean_water_diff, 1)
        slope_ground, _ = np.polyfit(time_steps_full, log_mean_ground_diff, 1)

        self.lyapunov_exponent_water = slope_water
        self.lyapunov_exponent_ground = slope_ground

    def toggle_binary(self):
        self.unperturbed_data = np.where(self.unperturbed_data > 0, 1, 0)
        self.perturbed_data = np.where(self.perturbed_data > 0, 1, 0)

    def toggle_continuous(self):
        self.unperturbed_data = self.copy_unperturbed_data
        self.perturbed_data = self.copy_perturbed_data

    def compute_hamming_distance(self):
        self.toggle_binary()
        time_steps = self.unperturbed_data.shape[0]
        self.hamming_water = []
        self.hamming_ground = []

        for t in range(time_steps):
            water_hamming = np.sum(
                self.unperturbed_data[t, ..., WATER_HEIGHT] != self.perturbed_data[t, ..., WATER_HEIGHT]
            )
            ground_hamming = np.sum(
                self.unperturbed_data[t, ..., GROUND_HEIGHT] != self.perturbed_data[t, ..., GROUND_HEIGHT]
            )

            self.hamming_water.append(water_hamming)
            self.hamming_ground.append(ground_hamming)

        self.hamming_water = np.array(self.hamming_water)
        self.hamming_ground = np.array(self.hamming_ground)

    def plot_mean_differences(self):
        time_steps_full = np.arange(len(self.mean_water_diff))

        fig, axs = plt.subplots(2, 1, figsize=(8, 8))

        # Subplot for Water Height Difference
        axs[0].plot(time_steps_full, self.mean_water_diff, "b-", label="Water Height Difference")
        axs[0].set_title("Water Height Difference Over Time")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Mean Absolute Difference")
        #axs[0].set_yscale("log")
        axs[0].legend()
        axs[0].grid(True)

        # Subplot for Terrain Height Difference
        axs[1].plot(time_steps_full, self.mean_ground_diff, "g-", label="Ground Height Difference")
        axs[1].set_title("Ground Height Difference Over Time")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Mean Absolute Difference")
        #axs[1].set_yscale("log")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_hamming_distance(self):
        time_steps = np.arange(len(self.hamming_water))

        plt.figure(figsize=(8, 8))
        plt.plot(time_steps, self.hamming_water, "b-", label="Hamming Distance (Water)")
        plt.title("Hamming Distance (Water) Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Hamming Distance")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_analysis(self):
        self.compute_mean_differences()
        normalized_water_diff, normalized_ground_diff = self.normalize_differences()
        log_mean_water_diff, log_mean_ground_diff = self.compute_log_differences(normalized_water_diff, normalized_ground_diff)
        self.fit_linear_model(log_mean_water_diff, log_mean_ground_diff)

        # Compute Hamming distances
        self.compute_hamming_distance()

        # Print results
        print("Mean Ground Height Difference:", self.mean_ground_diff)
        print("Mean Water Height Difference:", self.mean_water_diff)
        print(f"Lyapunov Exponent for Water Height: {self.lyapunov_exponent_water}")
        print(f"Lyapunov Exponent for Ground Height: {self.lyapunov_exponent_ground}")
        print(f"Hamming Distance - Water Height: {self.hamming_water[-1]}, Ground Height: {self.hamming_ground[-1]}")

        # Plot results
        self.plot_mean_differences()
        self.plot_hamming_distance()


# Constants for indexing
GROUND_HEIGHT = 0
WATER_HEIGHT = 1

# Example usage
if __name__ == "__main__":
    analysis = ExpAnalysis("data/unperturbed_data.npy", "data/perturbed_data.npy")
    analysis.run_analysis()
