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
    """
    Run the simulation with the fast cpp version for a number of timesteps
    the cpp simulation is invoked every state_per_gen steps, the grids after each
    cpp run are saved and returned 
    
    Args:
        initial state: grid of ground and water heights at t=0, should contain water source at the top
        steps: number of timesteps to run simulation
        steps_per_gen: number of runs in each individual simulation run - reduce this if grids are too large
        params: additional simulation parameters passed to the cpp simulator
        
    Returns:
        saved_grids: simulation states saved every steps_per_gen timesteps with shape [num_steps //steps_per_gen x height x width x NUM_CELL_FLOATS]
    """
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

def run_single_experiment(height, width, mean_slope, num_steps, flow, steps_per_gen=100, params=default_params):
    """
    Perform one experiment by generating the initial state and running the experiment as described by sim parameters
    
    Args:
        height, width: size of grid
        steps: number of timesteps to run simulation
        steps_per_gen: number of runs in each individual simulation run - reduce this if grids are too large
        params: additional simulation parameters passed to the cpp simulator
        
    Returns:
        saved_grids: simulation states saved every steps_per_gen timesteps with shape [num_steps //steps_per_gen x height x width x NUM_CELL_FLOATS]
    """
    np.random.seed(42)
    # width, height, ground_height, num_steps = 101, 1001, 101*.1, 10000
    ground_height = height * mean_slope    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.2, noise_type = 'white')    
    add_central_flow(initial_state, flow)
    
    grids = run_fastCA(initial_state, num_steps, steps_per_gen, params)
    return grids
        

def count_num_streams(grid, threshold = 1e-9, skip_rows = 1):
    """
    Counts the number of individual streams in all horizontal cross-sections of one simulation state.
    
    Args:
        grid: simulation state
        steps: number of timesteps to run simulation
        steps_per_gen: number of runs in each individual simulation run - reduce this if grids are too large
        params: additional simulation parameters passed to the cpp simulator
        
    Returns:
        num_streams: array of size [width] where num_streams[i] = number of rows where there are i streams        
    """
    height, width = grid.shape[:2]
    num_streams = np.zeros(width)
    for row in range(skip_rows, height -skip_rows):
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


def num_streams_histogram(height, width, mean_slope, num_steps, flow, plot_results = False):
    
    grid = run_single_experiment(height, width, mean_slope, num_steps, flow)
    
    num_streams = count_num_streams(grid[-1])
    
    if plot_results:
        stream_video(grid, scale=1)
        
        # plt.imshow(grids[-1,:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
        plt.imshow(grid[-1,:,:,WATER_HEIGHT])
        plt.colorbar()
        plt.show()
    
        # plt.hist(num_streams, width)
        plt.plot(num_streams, linestyle='', marker='o')
        plt.savefig('data/num_stream_histogram.png')
    
    return num_streams
    
    
def num_streams_experiment(output_file, slopes = np.linspace(0.01,10,20), flows=[0.25, 0.5, 1, 2, 4], width=101, height=1000, num_steps = 10000, steps_per_gen=100, params=default_params):
    """
    Runs the experiments with varying combinations of flows and slopes
    
    Args:
        output_file: location for saving simulation output
        slopes: array of test values for mean slope in the initial state
        flows: array of test values for water flow in the initial state 
        height, width: size of grid
        num_steps: number of timesteps to run simulation
        params: additional simulation parameters passed to the cpp simulator
        
    Returns:
        grids: array of shape [num_slopes x num_flows x num_steps //steps_per_gen x height x width x NUM_CELL_FLOATS]  saved in output_file     
    """
    data = np.zeros([len(slopes), len(flows), num_steps // steps_per_gen, height, width, NUM_CELL_FLOATS])
    for i, slope in enumerate(slopes):
        for j, flow in enumerate(flows):
            grids =  run_single_experiment(height=height, width=width, mean_slope=slope, num_steps=num_steps, steps_per_gen=steps_per_gen, flow=flow, params=params)
            data[i, j] = grids  
    if output_file is not None:      
        np.save(output_file, data)
    
def analyze_num_streams_experiment(data_file, output_file=None, slopes = np.linspace(0.05,1,10), flows=[0.25, 0.5, 1, 2, 4]):
    """
    Analyzes the data generated by the number of streams experiment with varying combinations of flows and slopes
    
    Args:
        data_file: location of the saved simulation states
        output_file: location for saving the plot
        slopes: array of test values for mean slope in the initial state
        flows: array of test values for water flow in the initial state
        
    Returns:
        plot of mean number of streams depending on the parameters - saved to output_file
        plot of the histograms for each slope with constant flow - shown to user   
    """    
    data = np.load(data_file)
    print(data.shape)
    width = data.shape[4]
    num_streams = np.zeros([len(slopes), len(flows), width])
    for i, slope in enumerate(slopes):
        for j, flow in enumerate(flows):
            num_streams[i,j] = count_num_streams(data[i,j,-1])
            
    print(num_streams)
        
    mean_num_streams_graph(num_streams, output_file, slopes, flows)
    plt.imshow(num_streams[:,2])
    plt.colorbar()
    plt.show()
        
        
        
def mean_num_streams_graph(data, output_file = None, slopes=np.linspace(0.01,1,10), flows=[0.25, 0.5, 1, 2, 4]):
    """
    plots the mean number of streams in a cross section dependent on slope for different water flow amounts 
    
    Args:
        data_file: location of the saved simulation states
        output_file: location for saving the plot
        slopes: array of test values for mean slope in the initial state
        flows: array of test values for water flow in the initial state
        
    Returns:
        plot of mean number of streams depending on the parameters - saved to output_file
        plot of the histograms for each slope with constant flow - shown to user   
    """    
    # assert data.shape[0] == len(slopes)
    # assert data.shape[1] == len(flows)
    width = data.shape[2]
    data = data / np.sum(data, axis=2, keepdims=True)
    # print(data)
    mean_num_streams = np.sum(data * np.arange(width), axis=2)
    print(mean_num_streams.shape, mean_num_streams)
    for i,flow in enumerate(flows):
        plt.plot(slopes, mean_num_streams[:,i]-1, linestyle = '-', marker = 'o', label=(flow))
    # plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('number of streams -1')
    plt.xlabel('mean slope')
    plt.legend(title='Water Flow')
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
    slopes=np.linspace(0.01,1,10)
    # num_streams_experiment('data/grids_data1', num_steps=10000, slopes = slopes,flows=[0.25, 0.5, 1, 2, 4], steps_per_gen=1000)
    analyze_num_streams_experiment('data/grids_data1.npy', 'slope_phase_trans.png', slopes=slopes)
    # mean_num_streams_graph('data/num_streams_data.npy', 'plots/num_streams_phase_transition_v1.png')
    # run_and_stream()