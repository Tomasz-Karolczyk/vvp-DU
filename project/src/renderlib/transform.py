from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class vec3:
    """
    Class for simplified work with numpy arrays of size 3.
    """

    def __init__(
        self,
        x: float | Any | None = None,
        y: float | None = None,
        z: float | None = None,
    ):
        """
        Creates vector:
        When x is float or None vector will be (x, y, z) None will default to 0.
        When x is vec3 it will be copied.
        When x is anything else 'np.array' will try to do the conversion.
        """

        if x is None and y is None and z is None:
            self.vec = np.zeros(3, dtype=float)

        elif y is None and z is None:
            if isinstance(x, vec3):
                self.vec = x.vec.copy()

            elif isinstance(x, float):
                self.vec = np.array([x, 0.0, 0.0], dtype=float)

            else:
                arr = np.array(x, dtype=float)
                if arr.shape != (3,):
                    raise ValueError("vec3 must be a 1D array of length 3")
                self.vec = arr

        elif z is None:
            self.vec = np.array([x, y, 0.0], dtype=float)

        elif y is None:
            self.vec = np.array([x, 0.0, z], dtype=float)

        else:
            self.vec = np.array([x, y, z], dtype=float)

    def __add__(self, other: "vec3") -> "vec3":
        return vec3(self.vec + other.vec)

    def __sub__(self, other: "vec3") -> "vec3":
        return vec3(self.vec - other.vec)

    def __mul__(self, other: float) -> "vec3":
        return vec3(self.vec * other)

    def __div__(self, other: float) -> "vec3":
        return vec3(self.vec / other)

    def normalize(self) -> "vec3":
        return vec3(self.vec / np.linalg.norm(self.vec))

    @staticmethod
    def one() -> "vec3":
        return vec3(1, 1, 1)

    @staticmethod
    def zero() -> "vec3":
        return vec3()

    @property
    def x(self) -> float:
        return self.vec[0]

    @x.setter
    def x(self, val: float) -> None:
        self.vec[0] = val

    @property
    def y(self) -> float:
        return self.vec[0]

    @y.setter
    def y(self, val: float) -> None:
        self.vec[0] = val

    @property
    def z(self) -> float:
        return self.vec[0]

    @z.setter
    def z(self, val: float) -> None:
        self.vec[0] = val


class Transform:
    """
    Class for storing and working with transform matrix.
    """

    def __init__(
        self,
        position: vec3 | None = None,
        rotation: Rotation | None = None,
        scale: vec3 | None = None,
    ):
        """
        Creates transform where position, rotation and scale are kept as separate values.
        When position is unspecified it defaults to (0, 0, 0).
        When rotation is unspecified it defaults to identity.
        When scale is unspecified it defaults to (1, 1, 1).
        """

        self.position = vec3.zero() if position is None else position
        self.rotation = Rotation.identity() if rotation is None else rotation
        self.scale = vec3.one() if scale is None else scale

    def get_matrix(self) -> NDArray:
        """
        Returns 4x4 transform matrix that performs scaling then rotation and lastly translation.
        """

        T = np.eye(4)
        T[:-1, 3] = self.position.vec

        S = np.diag([*self.scale.vec, 1])

        R = np.eye(4)
        R[:3, :3] = self.rotation.as_matrix()

        return T @ R @ S

    def get_inverse_matrix(self) -> NDArray:
        """
        Returns inverse of 4x4 transform matrix that performs scaling then rotation and lastly translation.
        """

        t = self.get_matrix()
        t = np.linalg.inv(t)
        return t
