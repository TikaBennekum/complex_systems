"""
Course: Complex systems
Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

File description:
    File to test thes system interactively.
"""

import numpy as np
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkInteractionWidgets import vtkSliderRepresentation2D, vtkSliderWidget
from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderWindowInteractor
from initial_state_generation import generate_initial_slope
from slider_example import (
    make_2d_slider_widget,
    Slider2DProperties,
    make_slider_properties,
)


class SliderCallback:
    def __init__(self, cube):
        """ """
        self.cube = cube

    def __call__(self, caller, ev):
        slider_widget = caller
        value = int(slider_widget.representation.value)
        # print(self.the_cube)
        self.cube.SetZLength(value)


def add_cube():

    # Create a vtkAppendPolyData to merge individual bar actors
    append_filter = vtk.vtkAppendPolyData()

    cube = vtk.vtkCubeSource()
    cube.SetXLength(0.8)  # Width of the bar
    cube.SetYLength(0.8)  # Depth of the bar
    transform_filter = update_cube_height(0.8, 0.0, cube)

    # Append the transformed bar to the append_filter
    append_filter.AddInputData(transform_filter.GetOutput())

    append_filter.Update()

    # Create a mapper and actor for the bars
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(append_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor, cube


def update_cube_height(value, bottom, cube):
    cube.SetZLength(value)  # Height of the bar

    # Set the position of the bar
    transform = vtk.vtkTransform()
    transform.Translate(
        0.0, 0.0, (value + bottom) / 2.0
    )  # Translate to proper location
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputConnection(cube.GetOutputPort())
    transform_filter.Update()
    return transform_filter


def main():

    # Create a renderer, render window, and render window interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    render_window.AddRenderer(renderer)
    render_window_interactor.SetRenderWindow(render_window)

    sp = make_slider_properties()
    # sp.Range.value = 0
    widget = make_2d_slider_widget(sp, render_window_interactor)

    # Add the actor to the renderer
    bar_chart_actor, cube = add_cube()
    callback = SliderCallback(cube)

    bar_chart_actor.GetProperty().SetColor(vtkNamedColors().GetColor3d("red"))
    # callback.add_actor(bar_chart_actor, step_nr)
    # if not step_nr == 9:
    #     bar_chart_actor.VisibilityOff()
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
