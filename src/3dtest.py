import numpy as np
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkInteractionWidgets import (
    vtkSliderRepresentation2D,
    vtkSliderWidget
)
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderWindowInteractor
)
from initial_state_generation import generate_initial_slope
from slider_example import make_2d_slider_widget, Slider2DProperties
from CAHistory import CAHistory
from CA import BOTTOM_NEIGHBORS

GROUND_HEIGHT = 0
WATER_HEIGHT = 1


class SliderCallback:
    def __init__(self):
        """
        """
        self.actor_list = []
    
    def add_actor(self, actor, step_nr):
        self.actor_list.append((actor, step_nr))

    def __call__(self, caller, ev):
        slider_widget = caller
        value = int(slider_widget.representation.value)
        # print(self.the_cube)
        for actor, step_nr in self.actor_list:
            if round(value) == step_nr:
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()
        # self.value = value
        # self.sphere_source.phi_resolution = value
        # self.sphere_source.theta_resolution = value * 2


def make_slider_properties(length):
    sp = Slider2DProperties()
    sp.Range.minimum_value = 0
    sp.Range.maximum_value = length - 1
    sp.Position.point1 = (0.3, 0.1)
    sp.Position.point2 = (0.7, 0.1)
    sp.Dimensions.slider_length = 0.05
    sp.Dimensions.slider_width = 0.025
    sp.Dimensions.end_cap_length = 0.02
    sp.Dimensions.title_height = 0.045
    sp.Dimensions.label_height = 0.035
    # Set color properties:
    # Change the color of the knob that slides.
    sp.Colors.slider_color = 'Green'
    # Change the color of the text indicating what the slider controls.
    sp.Colors.title_color = 'AliceBlue'
    # Change the color of the text displaying the value.
    sp.Colors.label_color = 'AliceBlue'
    # Change the color of the knob when the mouse is held on it.
    sp.Colors.selected_color = 'DeepPink'
    # Change the color of the bar.
    sp.Colors.bar_color = 'MistyRose'
    # Change the color of the ends of the bar.
    sp.Colors.bar_ends_color = 'Yellow'

    return sp


def stacked_3d_bar_chart(data, previous):
    # Get the dimensions of the data
    rows, cols = data.shape

    # Create a vtkAppendPolyData to merge individual bar actors
    append_filter = vtk.vtkAppendPolyData()

    cubes = [
        [vtk.vtkCubeSource() for _ in range(data.shape[0])] 
        for _ in range(data.shape[1])
    ]

    # Create bars for each data point
    for i in range(rows):
        for j in range(cols):
            value = data[i, j]
            bottom = previous[i, j]

            # Create a cube source for each bar
            cube = cubes[j][i]
            cube.SetXLength(0.8)  # Width of the bar
            cube.SetYLength(0.8)  # Depth of the bar
            transform_filter = update_cube_height(i, j, value, bottom, cube)

            # Append the transformed bar to the append_filter
            append_filter.AddInputData(transform_filter.GetOutput())

    append_filter.Update()

    # Create a mapper and actor for the bars
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(append_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def update_cube_height(i, j, value, bottom, cube):
    cube.SetZLength(value)  # Height of the bar

            # Set the position of the bar
    transform = vtk.vtkTransform()
    transform.Translate(i, j, (value + bottom) / 2.0 )  # Translate to proper location
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputConnection(cube.GetOutputPort())
    transform_filter.Update()
    return transform_filter


def slider_in_window(interactor):
    tubeWidth = 0.008
    sliderLength = 0.008
    titleHeight = 0.04
    labelHeight = 0.04

    # interactor = vtkRenderWindowInteractor()
    # interactor.SetRenderWindow(render_window)

    slider_rep = vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(0)
    slider_rep.SetMaximumValue(10)
    slider_rep.SetValue(0)
    slider_rep.SetTitleText("Time step")

    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(.1, .1)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(.9, .1)

    slider_rep.SetTubeWidth(tubeWidth)
    slider_rep.SetSliderLength(sliderLength)
    slider_rep.SetTitleHeight(titleHeight)
    slider_rep.SetLabelHeight(labelHeight)

    slider_widget = vtkSliderWidget()
    slider_widget.SetInteractor(interactor)
    slider_widget.SetRepresentation(slider_rep)
    slider_widget.SetAnimationModeToAnimate()
    slider_widget.EnabledOn()
    



def main():
    width, height, ground_height = 11, 101, 10
    initial_state = generate_initial_slope(height, width, ground_height, noise_amplitude = 10)
    
    model = CAHistory(width, height, initial_state, neighbor_list=BOTTOM_NEIGHBORS) # type: ignore
    model.run_simulation(10, show_live=False)
    data = model.get_history()
    print(f"{data.shape = }")
    data = np.swapaxes(data, 2, 3)
    data = np.swapaxes(data, 1, 2)

    # Create a renderer, render window, and render window interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    render_window.AddRenderer(renderer)
    render_window_interactor.SetRenderWindow(render_window)

    sp = make_slider_properties(len(data))
    sp.Range.value = 0
    widget = make_2d_slider_widget(sp, render_window_interactor)

    callback = SliderCallback()

    # Add the actor to the renderer
    for step_nr, step_data in enumerate(data):
        for layer, color in zip(range(step_data.shape[0]), ["peru", "deepskyblue"]):
            heights = step_data[layer]
            previous = step_data[layer] if layer > 0 else np.zeros_like(heights)
            bar_chart_actor = stacked_3d_bar_chart(heights, previous)
            bar_chart_actor.GetProperty().SetColor(vtkNamedColors().GetColor3d(color))
            callback.add_actor(bar_chart_actor, step_nr)
            if not step_nr == 9:
                bar_chart_actor.VisibilityOff()
            renderer.AddActor(bar_chart_actor)

    widget.AddObserver(vtkCommand.InteractionEvent, callback)

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
