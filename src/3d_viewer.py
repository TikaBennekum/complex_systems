import vtk
from vtk import vtkNamedColors
import numpy as np
from CA import EROSION_C, EROSION_EXPONENT, EROSION_K, N
from cpp_modules import fastCA
from initial_state_generation import add_central_flow, generate_initial_slope
from constants import *

class BarChartVisualizer:
    def __init__(self, grids):
        self.grids = grids
        self.current_grid_index = 0

        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.slider_widget = self.create_slider()

        self.initialize_chart()  # Initialize chart actors
        self.update_chart()  # Initial rendering


    def create_slider(self):
        slider_rep = vtk.vtkSliderRepresentation2D()
        slider_rep.SetMinimumValue(0)
        slider_rep.SetMaximumValue(len(self.grids) - 1)
        slider_rep.SetValue(0)
        slider_rep.SetTitleText("Grid Index")
        slider_rep.GetSliderProperty().SetColor(1, 0, 0)
        slider_rep.GetTitleProperty().SetColor(1, 1, 1)
        slider_rep.GetLabelProperty().SetColor(1, 1, 1)
        slider_rep.GetTubeProperty().SetColor(1, 1, 1)
        slider_rep.GetCapProperty().SetColor(1, 1, 1)
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(0.1, 0.1)
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(0.9, 0.1)

        slider_widget = vtk.vtkSliderWidget()
        slider_widget.SetInteractor(self.interactor)
        slider_widget.SetRepresentation(slider_rep)
        slider_widget.AddObserver("InteractionEvent", self.slider_callback)  # Use InteractionEvent for immediate updates

        return slider_widget

    def slider_callback(self, obj, event):
        slider_rep = obj.GetRepresentation()
        value = slider_rep.GetValue()
        # Snap to the nearest integer
        snapped_value = int(round(value))
        slider_rep.SetValue(snapped_value)  # Update slider to display the snapped value
        if snapped_value != self.current_grid_index:
            self.current_grid_index = snapped_value
            self.update_chart()


    def initialize_chart(self):
        """Create the initial bar chart actors for the first grid and store them."""
        self.actors = []  # Store references to all actors
        grid = self.grids[0]  # Use the first grid to initialize
        num_rows, num_cols, num_layers = grid.shape

        for i in range(num_rows):
            for j in range(num_cols):
                for layer in range(num_layers):
                    cube = vtk.vtkCubeSource()
                    cube.SetXLength(0.8)
                    cube.SetYLength(0.8)
                    cube.SetZLength(1.0)  # Initial height
                    cube.SetCenter(i, j, 0.5)  # Initial center (half the height)

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(cube.GetOutputPort())

                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    if layer == 0:
                        actor.GetProperty().SetColor(vtkNamedColors().GetColor3d("peru"))
                    elif layer == 1:
                        actor.GetProperty().SetColor(vtkNamedColors().GetColor3d("deepskyblue"))
                    self.renderer.AddActor(actor)

                    self.actors.append((cube, actor, layer))  # Store the cube source and actor

        self.renderer.ResetCamera()  # Set up the camera initially

    def update_chart(self):
        """Update the bar chart heights and positions without recreating actors."""
        grid = self.grids[self.current_grid_index]
        # num_rows, num_cols = grid.shape
        # max_value = np.max(grid) if np.max(grid) > 0 else 1  # Avoid divide-by-zero

        for (i, j, _), (cube, actor, layer) in zip(np.ndindex(grid.shape), self.actors):
            height = grid[i, j, layer]
            if layer == 1:
                height *= 20
            offset = sum(grid[i, j, :layer])
            cube.SetZLength(height)  # Scale height
            cube.SetCenter(i, j, height / 2.0 + offset)  # Update center
            if layer == 1 and height < 1e-6:
                actor.VisibilityOff()
            elif layer == 1:
                actor.VisibilityOn()

        self.render_window.Render()  # Refresh the window



    def run(self):
        self.slider_widget.EnabledOn()
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()

# Example usage:
if __name__ == "__main__":
    # Generate some example 2D grids
    np.random.seed(42)
    width, height, ground_height, num_steps = 21, 101, 101*.1, 10
    
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 0.2, noise_type = 'white')
    
    add_central_flow(initial_state, 1)
    
    
    grids = np.zeros([num_steps, height, width, NUM_CELL_FLOATS])
    grids[0] = initial_state
    
    
    # plt.imshow(grids[-1,:,:,GROUND_HEIGHT] - initial_state[:,:,GROUND_HEIGHT])
    # plt.imshow(grids[0,:,:,WATER_HEIGHT] )
    # plt.colorbar()
    
    # plt.savefig('data/cpptest0.png')
    
    params = {
        "EROSION_K": EROSION_K,
        "EROSION_C": EROSION_C,
        "EROSION_n": N,
        "EROSION_m": EROSION_EXPONENT,
    }
    
    fastCA.simulate(grids, params)

    visualizer = BarChartVisualizer(grids)
    visualizer.run()
