from CA import *
from initial_state_generation import generate_initial_slope, add_central_flow
import matplotlib.pyplot as plt
from visualization2d import *


from cpp_modules import fastCA


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    width, height, ground_height, num_steps = 21, 101, 101*.1, 1000
    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.2, noise_type = 'white')
    
    add_central_flow(initial_state, 10)
    
    
    grids = np.zeros([num_steps, height, width, NUM_CELL_FLOATS])
    grids[0] = initial_state
    
    
    # plt.imshow(grids[-1,:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
    plt.imshow(grids[0,:,:,WATER_HEIGHT] )
    plt.colorbar()
    
    plt.savefig('data/cpptest0.png')
    
    params = {
        "EROSION_K": EROSION_K,
        "EROSION_C": EROSION_C,
        "EROSION_n": N,
        "EROSION_m": EROSION_EXPONENT,
    }
    
    fastCA.simulate(grids, params)
    
    # print(grids)
    
    save_video(grids, 'videos/cpp_test')
    # print(grids)
    
    # # plt.imshow(grids[0,:,:,WATER_HEIGHT] )
    # plt.imshow(grids[-1,:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
    # plt.colorbar()
    
    # plt.savefig('data/cpptest.png')

    # plt.imshow(grids[1,:,:,WATER_HEIGHT] )
    # # plt.imshow(grids[1,:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
    # plt.colorbar()
    
    # plt.savefig('data/cpptest1.png')

    # plt.imshow(grids[-1,:,:,WATER_HEIGHT] )
    # # plt.imshow(grids[1,:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
    # plt.colorbar()
    
    # plt.savefig('data/cpptest2.png')
