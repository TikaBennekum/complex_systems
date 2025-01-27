from CA import *
from initial_state_generation import generate_initial_slope, add_central_flow
import matplotlib.pyplot as plt
from visualization2d import *


from cpp_modules import fastCA


def count_num_streams(grid, threshold = 1e-9, skip_rows = 1):
    height, width = grid.shape[:2]
    num_streams = np.zeros(width)
    for row in range(skip_rows, height):
        streams = 0
        water_cont = False
        for col in range(width):
            if grid[row, col, WATER_HEIGHT] > threshold:
                if not water_cont:
                    streams +=1
                water_cont = True
            else:
                water_cont = False
        num_streams[int(streams)] +=1
        print(num_streams)
    return num_streams


def num_streams_histogram():
    
    np.random.seed(42)
    width, height, ground_height, num_steps = 101, 1001, 101*.1, 10000
    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.2, noise_type = 'white')
    
    add_central_flow(initial_state, 5)
    
    
    grids = np.zeros([num_steps, height, width, NUM_CELL_FLOATS])
    grids[0] = initial_state
    
    params = {
        "EROSION_K": EROSION_K,
        "EROSION_C": EROSION_C,
        "EROSION_n": N,
        "EROSION_m": EROSION_EXPONENT,
    }
    
    fastCA.simulate(grids, params)
    
    plt.imshow(grids[-1,:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
    plt.colorbar()
    plt.show()
    
    num_streams = count_num_streams(grids[-1])
    # plt.hist(num_streams, width)
    plt.plot(num_streams, linestyle='', marker='o')
    plt.savefig('data/num_stream_histogram.png')
               
               
def run_and_stream():
    
    np.random.seed(42)
    width, height, ground_height, num_steps = 21, 101, 101*.1, 1000
    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.2, noise_type = 'white')
    
    add_central_flow(initial_state, 1)
    
    
    grids = np.zeros([num_steps, height, width, NUM_CELL_FLOATS])
    grids[0] = initial_state
    
    
    # plt.imshow(grids[-1,:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
    # plt.imshow(grids[0,:,:,WATER_HEIGHT] )
    # plt.colorbar()
    
    plt.savefig('data/cpptest0.png')
    
    params = {
        "EROSION_K": EROSION_K,
        "EROSION_C": EROSION_C,
        "EROSION_n": N,
        "EROSION_m": EROSION_EXPONENT,
    }
    
    fastCA.simulate(grids, params)
    
    # print(grids)
    # save_video(grids, 'videos/cpp_test.mp4')
    # print(grids)
    
    # # plt.imshow(grids[0,:,:,WATER_HEIGHT] )
    plt.imshow(grids[-1,:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
    plt.colorbar()
    plt.show()
    # plt.savefig('data/cpptest.png')

    stream_video(grids, scale=5) 

if __name__ == "__main__":
    # Example usage
    num_streams_histogram()
    # run_and_stream()