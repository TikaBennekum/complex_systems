from CA import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Example usage
    width, height, ground_height = 100, 100, 1
    output_file = 'videos/water_simulation.mp4'
    ca = CA(width, height, ground_height)
    ca.run_simulation(1000)
    plt.imshow(ca.grid[:,:, 0])
    plt.colorbar()
    plt.savefig('data/test.png')

