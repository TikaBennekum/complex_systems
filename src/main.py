from CA import *

if __name__ == "__main__":
    # Example usage
    width, height, ground_height = 100, 100, 50
    output_file = 'videos/water_simulation.mp4'
    ca = CA(width, height, ground_height)
    ca.run_simulation(100)
