from .transform import Transform
import numpy as np
import os
import sys
from pathlib import Path


# Get the working folder
if len(sys.argv) <= 0 or sys.argv[0] is None:
    raise RuntimeError("Cannot determine entry script path")

SCRIPT_DIR = Path(sys.argv[0]).resolve().parent


class Mesh:
    """
    Class for storing 3D mesh.
    """

    def __init__(self, file_path: str):
        """
        Creates mesh from file in format:
        vertices - 4 x ? numpy array (XYZ from file, W = 1).
        edges - 2 x ? numpy array (pairs of indices forming an line)
        """

        vertices = []
        edges = set()  # set to remove duplicate edges

        file_path = os.path.join(SCRIPT_DIR, file_path)

        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    # read vertex line: v x y z
                    parts = line.strip().split()

                    # convert to numbers and add 1 to the end
                    vertex = []
                    for i in range(1, 4):
                        vertex.append(float(parts[i]))
                    vertex.append(1.0)

                    vertices.append(vertex)

                elif line.startswith("f "):
                    # read face line: f v1, v2, v3 ...
                    parts = line.strip().split()[1:]

                    # remove normals, textures and convert indices to 0 format
                    indices = []
                    for p in parts:
                        numAsStr = p.split("/")[0]
                        indices.append(int(numAsStr) - 1)

                    # generate edges for the face as loop (breaks down the face to individual edges)
                    count = len(indices)
                    for i in range(count):
                        a = indices[i]
                        b = indices[(i + 1) % count]

                        if a > b:
                            (a, b) = (b, a)

                        edges.add((a, b))  # store sorted to avoid duplicates

        # convert to numpy
        self.vertices = np.array(vertices).T
        self.edges = np.array(list(edges))


class Object:
    """
    Class for giving mesh transform. (doesn't support object trees).
    """

    def __init__(self, mesh: Mesh):
        """
        Creates object with reference to 'mesh' and unique transformation.
        """

        self.mesh = mesh
        self.transform = Transform()
