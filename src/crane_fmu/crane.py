from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from component_model.model import Model, ModelOperationError  # type: ignore
from component_model.variable import Variable, VariableNP, quantity_direction  # type: ignore
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore
from scipy.spatial.transform import Rotation as Rot  # type: ignore

from crane_fmu.boom import Boom


class Crane(Model):
    """A crane object built from stiff booms and suitable to generate an FMU through PythonFMU.
    The crane should first be instantiated and then the booms added, i.e. the basic boom is registered,
    and through that all booms can be accessed.

    Args:
        name (str): the name of the crane instant
        description (str) = None: An (optional
    """

    def __init__(
        self,
        name: str,
        description: str = "A crane model",
        author: str = "Siegfried Eisinger (DNV)",
        version: str = "0.1",
        **kwargs,
    ):
        """Initialize the crane object."""
        super().__init__(
            name=name, description=description, author=author, version=version, **kwargs
        )
        self._boom0 = None  # placeholder for the first boom
        self.animation = None  # if animation object is defined, this will be set and cause re-drawing during simulation
        self._craneAngularVelocity = VariableNP(
            self,
            name="craneAngularVelocity",
            description="""Rotates the crane base, e.g. due to vessel roll and pitch. Defined as 3D rotation axis cartesian vector,
                         where the normal denotes the (right-handed) rotation direction and the length denotes the rotation velocity in rad/s""",
            causality="input",
            variability="continuous",
            value0=("0.0 rad/s", 0.0, 0.0, 0.0),
            on_step=lambda t, dT: (
                self.boom0.rotate(rot=self.craneAngularVelocity)
                if np.any(self.craneAngularVelocity != 0)
                else None
            ),
            on_set=lambda vec: Rot.from_rotvec(
                vec[0] * np.array(vec[1:]), degrees=False
            ),
        )
        self._craneVelocity = VariableNP(
            self,
            name="craneVelocity",
            description="""Moves the crane base linearly, e.g. due to vessel acceleration or lifting in waves. Defined as 3D cartesian vector,
                                                       where the normal denotes the movement direction and the length donotes the movement velocity""",
            causality="input",
            variability="continuous",
            value0=("0.0 m/s", 0.0, 0.0, 0.0),
            on_step=lambda t, dT: (
                self.boom0.translate(vec=self.craneVelocity * dT)
                if np.any(self.craneVelocity != 0)
                else None
            ),
            on_set=quantity_direction,
        )
        self._changeLoad = Variable(
            self,
            name="changeLoad",
            description="Load added to (or taken off) the end of the last boom (the hook)",
            causality="input",
            variability="continuous",
            value0="0.0 kg",
            on_step=lambda t, dT: self.boom0[-1].change_mass(self.changeLoad * dT),
        )
        #
        # output variables (connectors):
        self._craneTorque = VariableNP(
            self,
            name="craneTorque",
            description="Total torque of the crane with respect to its base, i.e. the sum of static and dynamic torques. Provided as 3D spherical vector",
            causality="output",
            variability="continuous",
            initial="exact",
            value0=("0.0 N*m", "0.0 rad", "0.0 rad"),
        )

    #         self.craneForce = VariableNP( self, name="CraneForce",
    #                                           description="Total linear force of the crane with respect to its base, i.e. the sum of static and dynamic forces. Provided as 3D spherical vector)",
    #                                           causality='output', variability='continuous', initial='exact',
    #                                           value0=( '0.0 N', '0.0 rad', '0.0 rad'))

    @property
    def boom0(self):
        return self._boom0

    @boom0.setter
    def boom0(self, newVal):
        assert isinstance(
            newVal, Boom
        ), f"A boom object expected as first boom on crane. Got {type(newVal)}"
        self._boom0 = newVal

    def booms(self, reverse=False):
        """Iterate through the booms.
        If reverse=True, the last element is first found and the iteration produces the booms from end to start.
        """
        boom = self._boom0
        if reverse:
            while boom.anchor1 is not None:  # walk to the end of the crane
                boom = boom.anchor1
        while boom is not None:
            yield boom
            boom = boom.anchor0 if reverse else boom.anchor1

    def boom_by_name(self, name: str) -> Boom:
        for b in self.booms():
            if b.name == name:
                return b
        raise ModelOperationError("Unknown boom " + name)

    def calc_statics_dynamics(self, dT=None):
        """Run the calc_statics_dynamics on all booms in reverse order, to get all Boom._c_m_sub and dynamics updated."""
        for b in self.booms(reverse=True):
            b.calc_statics_dynamics(dT)

    def do_step(self, currentTime, stepSize):
        """Do a simulation step of size 'stepSize at time 'currentTime.
        The input variables with values not equal to their value0 are listed in self.changedVariables.

        .. assumption:: rotation axis of internal booms is always known. For normal booms that is obvious.
          For rope, the axis is caused by previous movement and is thus known as internal data.
        """
        _ = super().do_step(currentTime, stepSize)  # generic model step activities
        # after all changed input variables are taken into account, update the statics and dynamics of the system
        self.calc_statics_dynamics(dT=stepSize)
        if self.animation is not None:
            self.animation.update(currentTime)
        self.craneTorque = self.boom0.torque
        #        print("Torque: (" + str(round(currentTime, 2)) + ")", self.craneTorque)
        #        print(f"Time {currentTime}, {self.boom0[1].name}, {self.boom0[1].point1}, {self.boom0[1]._tip.getter()}")
        return True


class Animation:
    """Animation of the crane via matplotlib.

    Args:
        crane (Crane): a reference to the crane which shall be animated
        elements (dict)={}: a dict of visual elements to include in the animation.
          Each dictionary element is represented by an empty list which is filled by the element references during init,
          so that their position, ... can be changed during the animation
        interval (float)=0.1: waiting interval between simulation steps in s
        viewAngle (tuple)=(20,45,0): Optional change of initial view angle as (elevation, azimuth, roll) (in degrees)
    """

    def __init__(
        self,
        crane: Crane,
        elements: dict | None = None,
        interval: float = 0.1,
        figsize=(9, 9),
        xlim=(-10, 10),
        ylim=(-10, 10),
        zlim=(0, 10),
        viewAngle: tuple = (20, 45, 0),
    ):
        """Perform the needed initializations of an animation."""
        self.crane = crane
        self.elements = elements
        self.interval = interval

        plt.ion()  # run the GUI event loop
        self.fig = plt.figure(figsize=figsize, layout="constrained")
        ax = Axes3D(fig=self.fig)
        #        ax = plt.axes(projection="3d")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=viewAngle[0], azim=viewAngle[1], roll=viewAngle[2])
        sub: list[list] = [[], [], []]
        if isinstance(self.elements, dict):
            for b in self.crane.booms():  # walk along the series of booms
                if "booms" in self.elements:  # draw booms
                    self.elements["booms"].append(
                        ax.plot(
                            [b.point0[0], b.point1[0]],
                            [b.point0[1], b.point1[1]],
                            [b.point0[2], b.point1[2]],
                            linewidth=b.animationLW,
                        )
                    )
                if (
                    "c_m" in self.elements
                ):  # write mass of boom as string on center of mass point
                    self.elements["c_m"].append(
                        ax.text(
                            b.c_m[0],
                            b.c_m[1],
                            b.c_m[2],
                            s=str(int(b.mass.value0)),
                            color="black",
                        )
                    )
                if "c_m_sub" in self.elements:
                    for i in range(3):
                        sub[i].append(b.c_m_sub[1][i])
            if "c_m_sub" in self.elements and len(sub[0]):
                self.elements["c_m_sub"].append(
                    ax.plot(
                        sub[0], sub[1], sub[2], marker="*", color="red", linestyle=""
                    )
                )  # need to put them in as plot and not scatter3d, such that coordinates can be changed in a good way
            if "currentTime" in self.elements:
                self.elements["currentTime"].append(
                    ax.text(
                        ax.get_xlim()[0],
                        ax.get_ylim()[0],
                        ax.get_zlim()[0],
                        s="time=0",
                        color="blue",
                    )
                )

    def update(self, currentTime=None):
        """Based on the updated crane, update data as defined in elements."""
        sub = [[], [], []]
        for i, b in enumerate(self.crane.booms()):
            if "booms" in self.elements:
                self.elements["booms"][i][0].set_data_3d(
                    [b.point0[0], b.point1[0]],
                    [b.point0[1], b.point1[1]],
                    [b.point0[2], b.point1[2]],
                )
            if "c_m" in self.elements:
                self.elements["c_m"][i].set_x(b.c_m_absolute[0])
                self.elements["c_m"][i].set_y(b.c_m_absolute[1])
                self.elements["c_m"][i].set_z(b.c_m_absolute[2])
            if "c_m_sub" in self.elements:
                for i in range(3):
                    sub[i].append(b.c_m_sub[1][i])
        if "c_m_sub" in self.elements and len(sub[0]):
            self.elements["c_m_sub"][0][0].set_data_3d(sub[0], sub[1], sub[2])
        if "currentTime" in self.elements and currentTime is not None:
            self.elements["currentTime"][0].set_text(
                "time=" + str(round(currentTime, 1))
            )

        self.fig.canvas.draw_idle()  # drawing updated values
        self.fig.canvas.flush_events()  # This will run the GUI event loop until all UI events currently waiting have been processed
        # time.sleep( self.interval)

    def interactive_off(self):
        plt.ioff()
