from renderlib import Object, Mesh
from renderlib import Camera
from renderlib import vec3
from renderlib import Animate, Rotate, Lerp, LerpScale
import numpy as np
import time

# load models -------------------
cubeMesh = Mesh("mesh/cube.obj")
icosahedronMesh = Mesh("mesh/icosahedron.obj")

# create objects from meshes
cube = Object(cubeMesh)
cube2 = Object(cubeMesh)  # multiple objects may use one mesh
icosahedron = Object(icosahedronMesh)

objects = [cube, icosahedron, cube2]

# setup icosahedron -------------
icosahedron.transform.position = vec3(-7, 2, 7)
icosahedron.transform.scale = vec3.one() * 4

# infinitely rotate
Rotate(icosahedron.transform, vec3(1, 1, 1), 30)

# setup cube --------------------
cube.transform.position = vec3(7, 2, 5)
cube.transform.scale = vec3.one() * 4


# infinitely move back and forth
def l1():
    Lerp(cube.transform, vec3(7, 2, 10), time=3, executeAfter=l2)


def l2():
    Lerp(cube.transform, vec3(7, 2, 5), time=3, executeAfter=l1)


l1()

# setup second cube -------------
SCALE = 1.3

cube2.transform.position = vec3(0, -2, 4)
cube2.transform.scale = vec3.one() * SCALE


# infinitely shrink and grow
scalingStack = []


# instead of chaining 6 functions do it using stack
def l3():
    global scalingStack

    if len(scalingStack) == 0:
        scalingStack = [
            vec3(2, 1, 2),
            vec3(1, 1, 2),
            vec3(1, 1, 1),
            vec3(1, 2, 1),
            vec3(2, 2, 1),
            vec3(2, 2, 2),
        ]

    top = scalingStack.pop()
    LerpScale(cube2.transform, top * SCALE, time=1, executeAfter=l3)


l3()

# setup camera ------------------
camera = Camera(fov=np.pi / 1.5)

# indefinitely rotate around its own axis
Rotate(camera.transform, vec3(0, 0, 1), 10)

# main loop ---------------------
FPS = 20
WAIT_PERIOD = 1 / FPS

Camera.ClearTerminal()
prevTime = time.time()
while True:
    # calculate deltaTime
    currTime = time.time()
    deltaTime = currTime - prevTime
    prevTime = currTime

    # sync with FPS
    sleepTime = WAIT_PERIOD - deltaTime
    if sleepTime > 0:
        time.sleep(sleepTime)

    camera.ResetPlot()

    Animate()

    camera.DrawObjects(objects)

    camera.Show()
