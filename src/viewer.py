"""
    Course: Complex systems
    Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

    File description:
        This file contains a 3D viewer class for our system.
"""

import vtk
from vtk import vtkNamedColors
import numpy as np
from typing import List
from CA import EROSION_C, EROSION_EXPONENT, EROSION_K, N
from cpp_modules import fastCA
from initial_state_generation import add_central_flow, generate_initial_slope
from constants import *

class BarChartVisualizer:
    def __init__(self, grids: np.ndarray, colors: List[str] = ["peru", "deepskyblue"]):
        """
        Initialize the 3D bar chart visualizer.

        ## Inputs
         - `grids`: A 4D numpy array (num_steps, height, width, num_layers) containing the data to visualize. Please don't input to many timesteps I am not sure how well that will work.
         - `colors`: A list of color names (strings) corresponding to each layer of the grid.
        """
        self.grids = grids
        self.colors = colors
        self.current_grid_index = 0  # Index of the grid to visualize

        # VTK renderer, window, and interactor
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        # Slider widget for navigating through grids
        self.slider_widget = self.create_slider()

        # VTK data structures for rendering
        self.poly_data = vtk.vtkPolyData()
        self.points = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()
        self.colors_array = vtk.vtkUnsignedCharArray()
        self.colors_array.SetNumberOfComponents(3)  # RGB colors

        # Initialize and render the chart
        self.initialize_chart()
        self.update_chart()

    def create_slider(self) -> vtk.vtkSliderWidget:
        """
        Create a slider widget for selecting the grid index.

        ## Returns 
         - A configured `vtkSliderWidget`.
        """
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
        slider_widget.AddObserver("InteractionEvent", self.slider_callback)

        return slider_widget

    def slider_callback(self, obj: vtk.vtkSliderWidget, event: str) -> None:
        """
        Callback for the slider widget to update the grid index and refresh the chart.

        ## Inputs
         - `obj`: The slider widget.
         - `event`: The event triggering the callback.
        """
        slider_rep = obj.GetRepresentation()
        value = slider_rep.GetValue()
        snapped_value = int(round(value))
        slider_rep.SetValue(snapped_value)  # Snap slider value to an integer
        if snapped_value != self.current_grid_index:
            self.current_grid_index = snapped_value
            self.update_chart()

    def initialize_chart(self) -> None:
        """
        Initialize the chart with a single actor and mapper.
        """
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()

    def update_chart(self) -> None:
        """
        Update the chart by modifying the polydata based on the current grid.
        """
        self.points.Reset()
        self.cells.Reset()
        self.colors_array.Reset()

        grid = self.grids[self.current_grid_index]
        num_rows, num_cols, num_layers = grid.shape

        # Track the accumulated height for stacking
        base_heights = np.zeros((num_rows, num_cols))

        for i in range(num_rows):
            for j in range(num_cols):
                for layer in range(num_layers):
                    height = grid[i, j, layer]
                    if height > 1e-6:  # Skip zero or near-zero heights
                        self.add_cube(i, j, base_heights[i, j], height, layer)
                        base_heights[i, j] += height

        # Update polydata with new points, cells, and colors
        self.poly_data.SetPoints(self.points)
        self.poly_data.SetPolys(self.cells)
        self.poly_data.GetCellData().SetScalars(self.colors_array)
        self.poly_data.Modified()

        self.render_window.Render()

    def add_cube(self, i: int, j: int, base_height: float, height: float, layer: int) -> None:
        """
        Add a single cube to the polydata.

        ## Inputs
         - `i`: Row index in the grid.
         - `j`: Column index in the grid.
         - `base_height`: Height at the bottom of this cube.
         - `height`: Height of the cube.
         - `layer`: Layer index (used for color selection).
        """
        x_min, x_max = i - 0.4, i + 0.4
        y_min, y_max = j - 0.4, j + 0.4
        z_min, z_max = base_height, base_height + height

        # Add points for the cube
        start_id = self.points.GetNumberOfPoints()
        vertices = [
            (x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min),  # Bottom
            (x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max), (x_min, y_max, z_max),  # Top
        ]
        for vertex in vertices:
            self.points.InsertNextPoint(vertex)

        # Define the six faces of the cube
        faces = [
            (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),  # Side faces
            (0, 1, 2, 3),  # Bottom face
            (4, 5, 6, 7),  # Top face
        ]
        for face in faces:
            quad = vtk.vtkQuad()
            for k in range(4):
                quad.GetPointIds().SetId(k, start_id + face[k])
            self.cells.InsertNextCell(quad)

        # Assign color based on the layer
        color = vtkNamedColors().GetColor3ub(self.colors[layer])
        for _ in faces:
            self.colors_array.InsertNextTypedTuple(color)

    def run(self) -> None:
        """
        Start the visualization.
        """
        self.slider_widget.EnabledOn()
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()


# Example usage
if __name__ == "__main__":
    # Parameters for grid generation
    np.random.seed(42)
    width, height, ground_height, num_steps = 21, 101, 101 * 0.1, 100_000

    # Generate the initial grid state
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude=0.2, noise_type='white')
    add_central_flow(initial_state, 1)

    # Create grids and simulate data
    grids = np.zeros([num_steps, height, width, NUM_CELL_FLOATS])
    grids[0] = initial_state

    params = {
        "EROSION_K": EROSION_K,
        "EROSION_C": EROSION_C,
        "EROSION_n": N,
        "EROSION_m": EROSION_EXPONENT,
    }

    fastCA.simulate(grids, params)

    # Visualize the data
    visualizer = BarChartVisualizer(grids[::1000])
    visualizer.run()
