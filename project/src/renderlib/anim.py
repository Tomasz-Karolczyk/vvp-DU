from .transform import Transform, vec3
from scipy.spatial.transform import Rotation
from typing import Callable
import time

deltaTime = 0.01
animationRegistry = []
prevTime = time.time()


def Animate():
    """
    Method that performs step of all active animations.
    """
    global animationRegistry, deltaTime, prevTime

    # remove all finished animations
    animationRegistry = [a for a in animationRegistry if not a.isFinished]

    # calculate deltaTime
    currTime = time.time()
    deltaTime = currTime - prevTime
    prevTime = currTime

    # execute step for all animations
    for a in animationRegistry:
        a.ExecuteStack()


class Animation:
    """
    Base class for working with transforms over time.
    """

    def __init__(
        self,
        target: Transform,
        time: float | None = None,
        executeAfter: Callable | None = None,
    ):
        """
        Creates and registers animation that modifies 'target', for 'time' seconds (None = until stopped manually).
        'executeAfter' is called when 'time' runs out.
        """

        global animationRegistry

        self.isFinished = False
        self.target = target
        self.time = time
        self.executeAfter = executeAfter

        animationRegistry.append(self)

    def ExecuteStack(self) -> None:
        """
        Method that executes animation from end to finish.
        That is time calculation, change of the transform and ending animation properly.
        """

        self.ExecuteBefore()
        self.Execute()
        self.ExecuteAfter()

    def ExecuteBefore(self) -> None:
        """
        Method that calculates time of this animation.
        """

        global deltaTime

        if self.time is None:
            return

        self.time -= deltaTime

    def Execute(self) -> None:
        """
        Fallback method if derived class doesn't specify one.
        """

        pass

    def ExecuteAfter(self) -> None:
        """
        Method that deals with ending animation properly.
        That is marking animation as finished and calling 'executeAfter' function.
        """

        if self.time is None:
            return

        if self.time > 0:
            return

        self.isFinished = True
        if self.executeAfter is not None:
            self.executeAfter()

    def End(self) -> None:
        """
        Method that marks this animation as finished.
        """

        self.isFinished = True


class Rotate(Animation):
    """
    Animation that applies rotation to a transform over time.
    """

    def __init__(
        self,
        target: Transform,
        axis: vec3,
        angle: float,
        time: float | None = None,
        executeAfter: Callable | None = None,
    ):
        """
        Creates and registers animation that applies rotation to 'target', for 'time' seconds (None = until stopped manually).
        Rotation is applied around 'axis' vector ('axis' will be normalized).
        'angle' specifies how fast should rotation be - in degrees per second.
        'executeAfter' is called when 'time' runs out.
        """

        super().__init__(target, time, executeAfter)

        n = axis.normalize()
        self.rotVec = (n * angle).vec

    def Execute(self) -> None:
        """
        This method applies the rotation to 'target' transform.
        """

        rotation = Rotation.from_rotvec(self.rotVec * deltaTime, True)
        self.target.rotation = rotation * self.target.rotation


class Lerp(Animation):
    """
    Animation that linearly interpolates position between two points.
    """

    def __init__(
        self,
        target: Transform,
        end: vec3,
        time: float,
        start: vec3 | None = None,
        executeAfter: Callable | None = None,
    ):
        """
        Creates and registers animation that linearly interpolates 'target's position.
        Interpolation is performed between 'start' and 'end' position or current and 'end' position when 'start' is None.
        It happens in 'time' seconds.
        'executeAfter' is called when 'time' runs out.
        """

        super().__init__(target, time, executeAfter)
        self.initTime = time
        self.end = end
        self.start = target.position if start is None else start

    def Execute(self) -> None:
        """
        This method performs linear interpolation on 'target's position.
        """

        t = 1 - max(0, min(1, self.time / self.initTime))
        delta = self.end - self.start

        self.target.position = self.start + delta * t


class LerpScale(Animation):
    """
    Animation that linearly interpolates scale between two points.
    """

    def __init__(
        self,
        target: Transform,
        end: vec3,
        time: float,
        start: vec3 | None = None,
        executeAfter: Callable | None = None,
    ):
        """
        Creates and registers animation that linearly interpolates 'target's scale.
        Interpolation is performed between 'start' and 'end' scale or current and 'end' scale when 'start' is None.
        It happens in 'time' seconds.
        'executeAfter' is called when 'time' runs out.
        """

        super().__init__(target, time, executeAfter)
        self.initTime = time
        self.end = end
        self.start = target.scale if start is None else start

    def Execute(self) -> None:
        """
        This method performs linear interpolation on 'target's scale.
        """

        t = 1 - max(0, min(1, self.time / self.initTime))
        delta = self.end - self.start

        self.target.scale = self.start + delta * t
