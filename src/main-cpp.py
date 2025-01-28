"""
    Course: Complex systems
    Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

    File description:
        This file contains the code to run the system in C++, resulting in faster simulations.
"""

from CA import *
from initial_state_generation import generate_initial_slope, add_central_flow
import matplotlib.pyplot as plt
from visualization2d import *


from cpp_modules import fastCA

default_params = {
    "EROSION_K": EROSION_K,
    "EROSION_C": EROSION_C,
    "EROSION_n": N,
    "EROSION_m": EROSION_EXPONENT,
    "print_params": 0,
}

def run_fastCA(initial_state, steps, steps_per_gen, params=default_params, update_progress = True):
    
    height, width, cell_dim = initial_state.shape
    num_gens = steps //steps_per_gen
    saved_grids = np.zeros([num_gens, height, width, cell_dim])
    saved_grids[0] = initial_state
    gen_grids = np.zeros([steps_per_gen, height, width, cell_dim])
    for gen in range(num_gens):
        if(update_progress):
            print(gen%10, end='', flush=True)
        gen_grids[0] = saved_grids[gen]
        fastCA.simulate(gen_grids, params)
        if gen < num_gens -1:
            saved_grids[gen+1] = gen_grids[-1]
        
    if(update_progress):
        print('')
    return saved_grids
        

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
    return num_streams


def num_streams_histogram(height, width, mean_slope, num_steps, plot_results = False):
    
    np.random.seed(42)
    # width, height, ground_height, num_steps = 101, 1001, 101*.1, 10000
    ground_height = height * mean_slope
    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.2, noise_type = 'white')
    
    add_central_flow(initial_state, 1)
    
    
    # grids = np.zeros([num_steps, height, width, NUM_CELL_FLOATS])
    # grids[0] = initial_state
    
    params = default_params
    
    # fastCA.simulate(grids, params)
    grids = run_fastCA(initial_state, num_steps, 100, params)
    
    num_streams = count_num_streams(grids[-1])
    
    if plot_results:
        stream_video(grids, scale=1)
        
        # plt.imshow(grids[-1,:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
        plt.imshow(grids[-1,:,:,WATER_HEIGHT])
        plt.colorbar()
        plt.show()
    
        # plt.hist(num_streams, width)
        plt.plot(num_streams, linestyle='', marker='o')
        plt.savefig('data/num_stream_histogram.png')
    
    return num_streams
    
    
def num_streams_experiment(output_file, plot_results = False, width=101, height=1000, num_steps = 10000, slopes = np.linspace(0.01,1,10)):
    data = np.zeros([len(slopes), width])
    for i, slope in enumerate(slopes):
        num_streams = num_streams_histogram(height=height, width=width, mean_slope=slope, num_steps=num_steps)
        data[i] = num_streams  
    if output_file is not None:      
        np.save(output_file, data)
    plt.imshow(data)
    plt.colorbar()
    plt.show()
        
def mean_num_streams_graph(data_file, output_file = None, slopes=np.linspace(0.01,1,10)):
    data = np.load(data_file)
    assert data.shape[0] == len(slopes)
    width = data.shape[1]
    data = data / np.sum(data, axis=1, keepdims=True)
    # print(data)
    mean_num_streams = np.sum(data * np.arange(width), axis=1)
    # print(mean_num_streams)
    plt.plot(slopes, mean_num_streams-1, linestyle = '', marker = 'o')
    plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('number of streams -1')
    plt.xlabel('mean slope')
    if output_file is not None:
        plt.savefig(output_file, dpi=600)
    plt.show()
               
               
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
    
    params = default_params
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
    # num_streams_histogram()
    # num_streams_experiment('data/num_streams_data', plot_results=True, num_steps=10000)
    mean_num_streams_graph('data/num_streams_data.npy', 'plots/num_streams_phase_transition_v1.png')
    # run_and_stream()