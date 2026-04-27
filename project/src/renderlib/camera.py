from .transform import Transform
from .object3d import Object
import numpy as np
from numpy.typing import NDArray
import numba
import os
from typing import Iterable
import shutil


DEFAULT_PRECISION = 500
PRECISION = DEFAULT_PRECISION
L_SPACE = np.linspace(0, 1, DEFAULT_PRECISION)
L_SPACE = L_SPACE.reshape((DEFAULT_PRECISION, 1))


class Camera:
    """
    Class for rendering into terminal.
    """

    @staticmethod
    def clear_terminal() -> None:
        """
        Clears terminal.
        """

        os.system("clear")

    @staticmethod
    def get_terminal_size() -> tuple[int, int]:
        """
        Returns dimensions of terminal in '(lines, columns)' format.
        """

        size = shutil.get_terminal_size()
        return size.lines, size.columns

    def __init__(
        self,
        char_x: int | None = None,
        char_y: int | None = None,
        transform: Transform | None = None,
        fov: float = np.pi / 3,
        aspect: float = 1,
        near: float = 0.1,
        far: float = 100.0,
        use_braille_font: bool = True,
        adjust_to_font_size: bool = True,
    ):
        """
        Creates camera.
        'charX' and 'charY' specify render output in characters (is slightly shrunk for border and rendering purposes).
        'transform' initializes cameras transform fallbacks to defaults when unspecified.
        'fow' sets field of view (viewing angle along width).
        'aspect' specifies cameras aspect.
        'near' and 'far' specify distance of near plane and far plane.
        'useBraille' increases resolution by factor of 8 by rendering braille dots as 8 pixels within each character.
        'adjustToFontSize' modifies aspect so font dimensions don't affect the aspect o the camera.
        """

        # Transform for camera
        self.transform = Transform() if transform is None else transform

        if char_x is None or char_y is None:
            char_y, char_x = Camera.get_terminal_size()
            char_y = char_y - 2  # space for redraw alignment

        self.useBrailleFont = use_braille_font
        px = char_x * (2 if use_braille_font else 1)
        py = char_y * (4 if use_braille_font else 1)

        screen_aspect = px / py
        screen_aspect *= 4 / 5  # account for spacing between lines

        if not use_braille_font:
            screen_aspect *= 0.5 if adjust_to_font_size else 1

        # Projection parameters
        self.fov = fov
        self.aspect = screen_aspect * aspect
        self.near = near
        self.far = far

        self.char_plot_size = (char_y, char_x)
        self.plot_size = (py, px)
        self.reset_plot()

    def reset_plot(self) -> None:
        """
        Clears internal frame buffer.
        """

        self.plot = np.zeros(self.plot_size, dtype=bool)

    def get_view_matrix(self) -> NDArray:
        """
        Returns 4x4 view matrix.
        Thats world space -> view space.
        """

        return self.transform.get_inverse_matrix()

    def get_projection_matrix(self) -> NDArray:
        """
        Returns 4x4 perspective projection matrix.
        """

        far = self.far
        near = self.near
        scale = 1 / np.tan(self.fov / 2)
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = scale / self.aspect
        P[1, 1] = scale
        P[2, 2] = (far + near) / (far - near)
        P[2, 3] = -2 * far * near / (far - near)
        P[3, 2] = 1
        return P

    def draw_objects(self, objects: Iterable[Object]):
        """
        This method renders multiple objects to a buffer.
        """

        for o in objects:
            self.draw_object(o)

    def draw_object(self, object: Object):
        """
        This method renders 'object' to a buffer.
        """

        # get matrices
        V = self.get_view_matrix()
        P = self.get_projection_matrix()
        T = object.transform.get_matrix()
        M = P @ V @ T

        #       T@       V@      P@
        # model -> world -> view -> clip space
        vertices = M @ object.mesh.vertices

        # clip space -> ndc x,y e <0, 1>
        w = vertices[3, :]
        w = np.where(w == 0, 1e-8, w)
        # abs due to clipping, can't be used for surface renderers
        vertices[:2, :] /= np.abs(w)

        # ndc -> screen space -> "pixel space"
        x = (0.5 + vertices[0, :] * 0.5) * self.plot_size[1]
        y = (0.5 - vertices[1, :] * 0.5) * self.plot_size[0]  # flip Y

        z = vertices[2, :]  # for clipping
        screen_space = np.vstack([x, y, z])  # (3, N)

        for edge in object.mesh.edges:
            v1 = screen_space[:, edge[0]]
            v2 = screen_space[:, edge[1]]
            draw_line(self.plot, self.plot_size, v1, v2)

    def GetChar(self, x: int, y: int) -> str:
        """
        This method generates character for [x, y] position.
        Expects 'x' and 'y' to be in bounds.
        """

        if not self.useBrailleFont:
            return "#" if self.plot[y, x] else " "

        y *= 4
        x *= 2

        multipliers = np.array([[1, 8], [2, 16], [4, 32], [64, 128]])

        mask = self.plot[y : y + 4, x : x + 2]

        index = np.sum(multipliers * mask)

        BRAILLE_OFFSET = 0x2800
        return chr(BRAILLE_OFFSET + index)

    def Show(self) -> None:
        """
        This method converts the buffer into string and writes it to terminal.
        """

        text = ""

        for y in range(self.char_plot_size[0]):
            for x in range(self.char_plot_size[1]):
                text += self.GetChar(x, y)
            text += "\n"

        # os.system('clear')
        print("\033[H", end="")  # redraw instead of clearing
        print(text, sep="")


# numba compiled -----------------------


@numba.njit(cache=True, fastmath=True, parallel=True)
def draw_line(
    plot: NDArray, plot_size: tuple[int, int], v1: NDArray, v2: NDArray
) -> None:
    """
    This method draws line between positions 'v1' and 'v2'.
    """

    delta = v2 - v1
    step = 1 / PRECISION

    for i in numba.prange(PRECISION):
        point = v1 + i * step * delta

        if point[2] < 0:
            continue
        plot_at(plot, plot_size, point[0], point[1])


@numba.njit(cache=True, fastmath=True)
def plot_at(plot: NDArray, plot_size: tuple[int, int], x: float, y: float) -> None:
    """
    This method sets logical pixel at [x, y] in buffer.
    """

    x = int(x)
    y = int(y)

    if x < 0 or y < 0 or x >= plot_size[1] or y >= plot_size[0]:
        return

    plot[y, x] = True
