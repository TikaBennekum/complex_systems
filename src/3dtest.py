import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
import numpy as np

GROUND_HEIGHT = 0
WATER_HEIGHT = 1

def stacked_3d_bar_chart(data, previous):
    # Get the dimensions of the data
    rows, cols = data.shape

    # Create a vtkAppendPolyData to merge individual bar actors
    append_filter = vtk.vtkAppendPolyData()

    # Create bars for each data point
    for i in range(rows):
        for j in range(cols):
            value = data[i, j]
            bottom = previous[i, j]

            # Create a cube source for each bar
            cube = vtk.vtkCubeSource()
            cube.SetXLength(0.8)  # Width of the bar
            cube.SetYLength(0.8)  # Depth of the bar
            cube.SetZLength(value)  # Height of the bar

            # Set the position of the bar
            transform = vtk.vtkTransform()
            transform.Translate(i, j, value / 2.0 + bottom)  # Translate to proper location
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetTransform(transform)
            transform_filter.SetInputConnection(cube.GetOutputPort())
            transform_filter.Update()

            # Append the transformed bar to the append_filter
            append_filter.AddInputData(transform_filter.GetOutput())

    append_filter.Update()

    # Create a mapper and actor for the bars
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(append_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def main():
    # Define a 2D array of points
    data = np.load("videos/output.npy")
    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 0, 1)
    print(f"{data.shape = }")

    # Create a renderer, render window, and render window interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window_interactor = vtk.vtkRenderWindowInteractor()

    render_window.AddRenderer(renderer)
    render_window_interactor.SetRenderWindow(render_window)

    # Add the actor to the renderer
    for layer, color in zip(range(data.shape[0]), ["peru", "deepskyblue"]):
        heights = data[layer]
        previous = data[layer] if layer > 0 else np.zeros_like(heights)
        bar_chart_actor = stacked_3d_bar_chart(heights, previous)
        bar_chart_actor.GetProperty().SetColor(vtkNamedColors().GetColor3d(color))

        renderer.AddActor(bar_chart_actor)

    renderer.SetBackground(0.1, 0.2, 0.4)  # Background color

    # Set up the camera
    renderer.GetActiveCamera().Azimuth(0)
    renderer.GetActiveCamera().Elevation(-50)
    renderer.GetActiveCamera().Yaw(30)
    renderer.GetActiveCamera().Roll(-30)
    renderer.ResetCamera()

    # Start the rendering loop
    render_window.Render()
    render_window_interactor.Start()

if __name__ == "__main__":
    main()
