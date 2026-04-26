from .transform import Transform
from .object3d import Object
import numpy as np
from numpy.typing import NDArray

import os
from typing import Iterable
import shutil


DEFAULT_PRECISION = 150
L_SPACE = np.linspace(0, 1, DEFAULT_PRECISION)
L_SPACE = L_SPACE.reshape((DEFAULT_PRECISION, 1))


class Camera:
    """
    Class for rendering into terminal.
    """

    @staticmethod
    def ClearTerminal() -> None:
        """
        Clears terminal.
        """

        os.system("clear")

    @staticmethod
    def GetTerminalSize() -> tuple[int, int]:
        """
        Returns dimensions of terminal in '(lines, columns)' format.
        """

        size = shutil.get_terminal_size()
        return size.lines, size.columns

    def __init__(
        self,
        charX: int | None = None,
        charY: int | None = None,
        transform: Transform | None = None,
        fov: float = np.pi / 3,
        aspect: float = 1,
        near: float = 0.1,
        far: float = 100.0,
        useBrailleFont: bool = True,
        adjustToFontSize: bool = True,
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

        if charX is None or charY is None:
            charY, charX = Camera.GetTerminalSize()
            charX = charX - 2  # space for border
            charY = charY - 4  # space for border and for redraw alignment

        self.useBrailleFont = useBrailleFont
        px = charX * (2 if useBrailleFont else 1)
        py = charY * (4 if useBrailleFont else 1)

        screenAspect = px / py
        screenAspect *= 4 / 5  # account for spacing between lines

        if not useBrailleFont:
            screenAspect *= 0.5 if adjustToFontSize else 1

        # Projection parameters
        self.fov = fov
        self.aspect = screenAspect * aspect
        self.near = near
        self.far = far

        self.charPlotSize = (charY, charX)
        self.plotSize = (py, px)
        self.ResetPlot()

    def ResetPlot(self) -> None:
        """
        Clears internal frame buffer.
        """

        self.plot = np.zeros(self.plotSize, dtype=bool)

    def GetViewMatrix(self) -> NDArray:
        """
        Returns 4x4 view matrix.
        Thats world space -> view space.
        """

        return self.transform.GetInverseMatrix()

    def GetProjectionMatrix(self) -> NDArray:
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

    def DrawObjects(self, objects: Iterable[Object]):
        """
        This method renders multiple objects to a buffer.
        """

        for o in objects:
            self.DrawObject(o)

    def DrawObject(self, object: Object):
        """
        This method renders 'object' to a buffer.
        """

        # get matrices
        V = self.GetViewMatrix()
        P = self.GetProjectionMatrix()
        T = object.transform.GetMatrix()
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
        x = (0.5 + vertices[0, :] * 0.5) * self.plotSize[1]
        y = (0.5 - vertices[1, :] * 0.5) * self.plotSize[0]  # flip Y

        z = vertices[2, :]  # for clipping
        screenSpace = np.vstack([x, y, z])  # (3, N)

        for edge in object.mesh.edges:
            v1 = screenSpace[:, edge[0]]
            v2 = screenSpace[:, edge[1]]
            self.DrawLine(v1, v2)

    def DrawLine(self, v1: NDArray, v2: NDArray) -> None:
        """
        This method draws line between positions 'v1' and 'v2'.
        """

        v1 = v1.reshape((1, 3))
        v2 = v2.reshape((1, 3))
        delta = v2 - v1

        points = v1 + L_SPACE @ delta

        for p in points:
            if p[2] < 0:
                continue
            self.Plot(p[0], p[1])

    def Plot(self, x: float, y: float) -> None:
        """
        This method sets logical pixel at [x, y] in buffer.
        """

        x = int(x)
        y = int(y)

        if x < 0 or y < 0 or x >= self.plotSize[1] or y >= self.plotSize[0]:
            return

        # prioritize closer
        self.plot[y, x] = True

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

        brailleOffset = np.sum(multipliers * mask)

        return chr(0x2800 + brailleOffset)

    def Show(self) -> None:
        """
        This method converts the buffer into string and writes it to terminal.
        """

        text = ""

        for y in range(self.charPlotSize[0]):
            text += "|"  # dotted boundary logic
            for x in range(self.charPlotSize[1]):
                text += self.GetChar(x, y)
            text += "|\n"  # dotted boundary logic

        # dotted boundary logic
        dottedLine = "+" + "-" * self.charPlotSize[1]
        dottedLine += "+\n"

        # os.system('clear')
        print("\033[H", end="")  # redraw instead of clearing
        print(dottedLine, text, dottedLine, sep="")
