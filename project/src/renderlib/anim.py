from .transform import Transform, vec3
from scipy.spatial.transform import Rotation
from typing import Callable
import time

delta_time = 0.01
animation_registry = []
prev_time = time.time()


def Animate():
    """
    Method that performs step of all active animations.
    """
    global animation_registry, delta_time, prev_time

    # remove all finished animations
    animation_registry = [a for a in animation_registry if not a.is_finished]

    # calculate deltaTime
    current_time = time.time()
    delta_time = current_time - prev_time
    prev_time = current_time

    # execute step for all animations
    for a in animation_registry:
        a.step()


class Animation:
    """
    Base class for working with transforms over time.
    """

    def __init__(
        self,
        target: Transform,
        time: float | None = None,
        execute_after: Callable | None = None,
    ):
        """
        Creates and registers animation that modifies 'target', for 'time' seconds (None = until stopped manually).
        'executeAfter' is called when 'time' runs out.
        """

        global animation_registry

        self.is_finished = False
        self.target = target
        self.time = time
        self.execute_after = execute_after

        animation_registry.append(self)

    def step(self) -> None:
        """
        Method that executes animation from end to finish.
        That is time calculation, change of the transform and ending animation properly.
        """

        self.execute_before()
        self.execute()
        self.execute_end()

    def execute_before(self) -> None:
        """
        Method that calculates time of this animation.
        """

        global delta_time

        if self.time is None:
            return

        self.time -= delta_time

    def execute(self) -> None:
        """
        Fallback method if derived class doesn't specify one.
        """

        pass

    def execute_end(self) -> None:
        """
        Method that deals with ending animation properly.
        That is marking animation as finished and calling 'executeAfter' function.
        """

        if self.time is None:
            return

        if self.time > 0:
            return

        self.end_animation()

    def end_animation(self) -> None:
        """
        Method that marks this animation as finished.
        """

        self.is_finished = True
        if self.execute_after is not None:
            self.execute_after()


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
        execute_after: Callable | None = None,
    ):
        """
        Creates and registers animation that applies rotation to 'target', for 'time' seconds (None = until stopped manually).
        Rotation is applied around 'axis' vector ('axis' will be normalized).
        'angle' specifies how fast should rotation be - in degrees per second.
        'execute_after' is called when 'time' runs out.
        """

        super().__init__(target, time, execute_after)

        n = axis.normalize()
        self.rotVec = (n * angle).vec

    def execute(self) -> None:
        """
        This method applies the rotation to 'target' transform.
        """

        rotation = Rotation.from_rotvec(self.rotVec * delta_time, True)
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
        execute_after: Callable | None = None,
    ):
        """
        Creates and registers animation that linearly interpolates 'target's position.
        Interpolation is performed between 'start' and 'end' position or current and 'end' position when 'start' is None.
        It happens in 'time' seconds.
        'execute_after' is called when 'time' runs out.
        """

        super().__init__(target, time, execute_after)
        self.init_time = time
        self.end = end
        self.start = target.position if start is None else start

    def execute(self) -> None:
        """
        This method performs linear interpolation on 'target's position.
        """

        t = 1 - max(0, min(1, self.time / self.init_time))
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
        execute_after: Callable | None = None,
    ):
        """
        Creates and registers animation that linearly interpolates 'target's scale.
        Interpolation is performed between 'start' and 'end' scale or current and 'end' scale when 'start' is None.
        It happens in 'time' seconds.
        'execute_after' is called when 'time' runs out.
        """

        super().__init__(target, time, execute_after)
        self.init_time = time
        self.end = end
        self.start = target.scale if start is None else start

    def execute(self) -> None:
        """
        This method performs linear interpolation on 'target's scale.
        """

        t = 1 - max(0, min(1, self.time / self.init_time))
        delta = self.end - self.start

        self.target.scale = self.start + delta * t
