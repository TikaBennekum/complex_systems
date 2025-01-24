## copied from https://examples.vtk.org/site/PythonicAPI/Widgets/Slider2D/

from dataclasses import dataclass
from typing import Tuple

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import (
    vtkSliderRepresentation2D,
    vtkSliderWidget
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def main():
    colors = vtkNamedColors()

    # A sphere.
    sphere_source = vtkSphereSource(center=(0.0, 0.0, 0.0), radius=4.0)

    mapper = vtkPolyDataMapper()
    sphere_source >> mapper

    actor = vtkActor(mapper=mapper)
    actor.property.SetInterpolationToFlat()
    actor.property.color = colors.GetColor3d('MistyRose')
    actor.property.edge_color = colors.GetColor3d('Tomato')
    actor.property.edge_visibility = True

    # A renderer and render window.
    renderer = vtkRenderer(background=colors.GetColor3d('SlateGray'))
    render_window = vtkRenderWindow(size=(640, 480), window_name='Slider2D')
    render_window.AddRenderer(renderer)

    # An interactor.
    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.render_window = render_window
    style = vtkInteractorStyleTrackballCamera()
    render_window_interactor.interactor_style = style

    # Add the actors to the scene.
    renderer.AddActor(actor)

    # Render an image (lights and cameras are created automatically).
    render_window.Render()

    sp = make_slider_properties()

    sp.Text.title = 'Sphere Resolution'
    sp.Range.value = 5

    widget = make_2d_slider_widget(sp, render_window_interactor)
    cb = SliderCallback(sphere_source)
    widget.AddObserver(vtkCommand.InteractionEvent, cb)

    renderer.Render()
    renderer.active_camera.Dolly(1.0)
    render_window_interactor.Initialize()
    render_window.Render()

    render_window_interactor.Start()


def make_slider_properties():
    sp = Slider2DProperties()
    sp.Range.minimum_value = 3
    sp.Range.maximum_value = 20
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


def make_2d_slider_widget(properties, interactor):
    """
    Make a 2D slider widget.

    :param properties: The 2D slider properties.
    :param interactor: The vtkInteractor.
    :return: The slider widget.

    from https://examples.vtk.org/site/PythonicAPI/Widgets/Slider2D/
    """
    colors = vtkNamedColors()

    slider_rep = vtkSliderRepresentation2D(minimum_value=properties.Range.minimum_value,
                                           maximum_value=properties.Range.maximum_value,
                                           value=properties.Range.value,
                                           title_text=properties.Text.title,
                                           tube_width=properties.Dimensions.tube_width,
                                           slider_length=properties.Dimensions.slider_length,
                                           slider_width=properties.Dimensions.slider_width,
                                           end_cap_length=properties.Dimensions.end_cap_length,
                                           end_cap_width=properties.Dimensions.end_cap_width,
                                           title_height=properties.Dimensions.title_height,
                                           label_height=properties.Dimensions.label_height,
                                           )

    # Set the color properties.
    slider_rep.title_property.color = colors.GetColor3d(properties.Colors.title_color)
    slider_rep.label_property.color = colors.GetColor3d(properties.Colors.label_color)
    slider_rep.tube_property.color = colors.GetColor3d(properties.Colors.bar_color)
    slider_rep.cap_property.color = colors.GetColor3d(properties.Colors.bar_ends_color)
    slider_rep.slider_property.color = colors.GetColor3d(properties.Colors.slider_color)
    slider_rep.selected_property.color = colors.GetColor3d(properties.Colors.selected_color)

    # Set the position.
    slider_rep.point1_coordinate.coordinate_system = properties.Position.coordinate_system
    slider_rep.point1_coordinate.value = properties.Position.point1
    slider_rep.point2_coordinate.coordinate_system = properties.Position.coordinate_system
    slider_rep.point2_coordinate.value = properties.Position.point2

    title_font_family = properties.Text.title_font_family
    match title_font_family:
        case 'Courier':
            slider_rep.title_property.SetFontFamilyToCourier()
        case 'Times':
            slider_rep.title_property.SetFontFamilyToTimes()
        case _:
            slider_rep.title_property.SetFontFamilyToArial()
    slider_rep.title_property.bold = properties.Text.title_bold
    slider_rep.title_property.italic = properties.Text.title_italic
    slider_rep.title_property.shadow = properties.Text.title_shadow
    label_font_family = properties.Text.label_font_family
    match label_font_family:
        case 'Courier':
            slider_rep.label_property.SetFontFamilyToCourier()
        case 'Times':
            slider_rep.label_property.SetFontFamilyToTimes()
        case _:
            slider_rep.label_property.SetFontFamilyToArial()
    slider_rep.label_property.bold = properties.Text.label_bold
    slider_rep.label_property.italic = properties.Text.label_italic
    slider_rep.label_property.shadow = properties.Text.label_shadow

    widget = vtkSliderWidget(representation=slider_rep, interactor=interactor, enabled=True)
    widget.SetAnimationModeToAnimate()

    return widget


@dataclass(frozen=True)
class Coordinate:
    @dataclass(frozen=True)
    class CoordinateSystem:
        VTK_DISPLAY: int = 0
        VTK_NORMALIZED_DISPLAY: int = 1
        VTK_VIEWPORT: int = 2
        VTK_NORMALIZED_VIEWPORT: int = 3
        VTK_VIEW: int = 4
        VTK_POSE: int = 5
        VTK_WORLD: int = 6
        VTK_USERDEFINED: int = 7


@dataclass
class Slider2DProperties:
    @dataclass
    class Colors:
        title_color: str = 'White'
        label_color: str = 'White'
        slider_color: str = 'White'
        selected_color: str = 'HotPink'
        bar_color: str = 'White'
        bar_ends_color: str = 'White'

    @dataclass
    class Dimensions:
        tube_width: float = 0.008
        slider_length: float = 0.01
        slider_width: float = 0.02
        end_cap_length: float = 0.005
        end_cap_width: float = 0.05
        title_height: float = 0.03
        label_height: float = 0.025

    @dataclass
    class Position:
        coordinate_system: int = Coordinate.CoordinateSystem.VTK_NORMALIZED_VIEWPORT
        point1: Tuple = (0.1, 0.1)
        point2: Tuple = (0.9, 0.1)

    @dataclass
    class Range:
        minimum_value: float = 0.0
        maximum_value: float = 1.0
        value: float = 0.0

    @dataclass
    class Text:
        # Font families are: Ariel, Courier and Times
        title: str = ''
        title_font_family = 'Arial'
        title_bold: bool = True
        title_italic: bool = False
        title_shadow: bool = True
        label_font_family = 'Arial'
        label_bold: bool = True
        label_italic: bool = False
        label_shadow: bool = True


class SliderCallback:

    def __init__(self, sphere_source):
        """
        """
        self.sphere_source = sphere_source

    def __call__(self, caller, ev):
        slider_widget = caller
        value = int(slider_widget.representation.value)
        self.sphere_source.phi_resolution = value
        self.sphere_source.theta_resolution = value * 2


if __name__ == '__main__':
    main()
