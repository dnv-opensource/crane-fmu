from __future__ import annotations

from math import sqrt
from typing import Optional

import numpy as np
from component_model.model import Model  # type: ignore
from component_model.variable import Variable, VariableNP, spherical_to_cartesian  # type: ignore
from scipy.spatial.transform import Rotation as Rot  # type: ignore


class BoomInitError(Exception):
    """Special error indicating that something is wrong with the boom definition."""

    pass


class BoomOperationError(Exception):
    """Special error indicating that something went wrong during boom operation (rotations, translations,calculation of CoM,...)."""

    pass


class Boom(object):
    """Boom object, representing one element of a crane,
    modelled as stiff line with length, mass and given degrees of freedom.

    The boom has a

    * position (3D position of hinge, which can be moved by parent)
    * boom (3D vector along the boom from hinge to hinge)

    Basic boom movements are

    * Rotation: active rotation around the hinge, obeying the defined degrees of freedom, or passive initiated by parent booms (around their hinge)
    * Translation: the boom is moved linearly, due to linear movement of the hinge
    * length change: length change of booms (e.g. rope) can be one defined degree of freedom. It is similar to translation, but the hinge does not move, only the end.
    * mass change: mass change of booms (e.g. add or remove load) can be one defined degree of freedom.

    After any movement

    * the internal variables (center of mass, positions, etc.) are re-calculated,
    * the attached booms are moved in accordance
    * and the center of mass of the sub-system (the boom with attached booms) is re-calculated.
    * Finally the parent booms are informed about the change of the sub-system center of mass, leading to re-calculation of their internal variables.

    .. note:: initialization variables are designed for easy management of parameters, while derived internal variables are more complex (often 3D vectors)

    Args:
        name (str): The short name of the boom (unique within FMU)
        description (str) = '':  An optional description of the boom
        anchor0 (Model,Boom): the model (crane) object to which this Boom belongs (if this is the first boom), or anchor of the boom where it is fixed to the crane.
        mass (float): Parameter denoting the (assumed fixed) mass of the boom
        centerOfMass (float,tuple): Parameter denoting the (assumed fixed) position of the center of mass of the boom,
          provided as portion of the length (as float) and optionally the absolute displacements in x- and y-direction (assuming the boom in z-direction),
          e.g. (0.5,'-0.5 m','1m'): halfway down the boom displaced 0.5m in the -x direction and 1m in the y direction
        boom (tuple): A tuple defining the boom relative to the z-axis in spherical (ISO 80000) coordinates, consisting of

           * length: the length of the boom (in length units)
           * polar: a rotation angle for a rotation around the positive y-axis against the clock.
           * azimuth: a rotation angle for a rotation around the positive z-axis against the clock.
          Note: The boom and its range is saved as variable, while the active work variables are the cartesian point0 and direction (cartesian boom vector)
          The range can only be checked if the boom variable is kept updated.
        boommRng (tuple): Range for each of the boom components, relative to the z-axis, i.e. how much the boom can be rotated/lengthened with respect to the z-axis.
          As normal, range components specified as None denote fixed components. Most booms have only one (rotation) degree of freedom.
        dampingQ (float)=0.0: optional possibility to implement a loose connection between booms (dampingQ>0),
          e.g. the crane rope is implemented as a stiff boom of variable length with a loose connection hanging from the previous boom.

          The dampingQ denotes the dimensionless damping quality factor (energy stored/energy lost per radian),
          which is also equal to 2*ln( amplitude/amplitude next period), or pi*frequency*decayTime
        animationLW (int)=5: Optional possibility to change the default line width when performing animations.
          E.g. the pedestal might be drawn with larger and the rope with smaller line width

    Instantiate like:

    .. code-block:: python

       pedestal = Boom( name         ='pedestal',
                        description  = "The vertical crane base, on one side fixed to the vessel and on the other side the first crane boomm is fixed to it.
                          The mass should include all additional items fixed to it, like the operator's cab",
                        anchor0      = crane,
                        mass         = '2000.0 kg',
                        centerOfMass = (0.5, 0,'2 deg'),
                        boom        = ('5.0 m', 0, '0deg'),
                        boomRng      = (None, (0,'360 deg'), None)


    .. todo:: determine the range of forces
    .. limitation:: The mass and the centerOfMass setting of booms is assumed constant. With respect to rope and hook of a crane this means that basically only the mass of the hook is modelled.
    .. assumption:: Center of mass: _c_m is the local c_m measured relative to point0. _c_m_sub is a global quantity
    """

    def __init__(
        self,
        name: str,
        description: str,
        anchor0: Model | Boom,
        mass: str | None = None,
        centerOfMass: float | tuple[float | str, float | str, float | str] = 0.5,
        boom: tuple[float | str, float | str, float | str] | None = None,
        boomRng: tuple[Optional[float], Optional[float], tuple[float | str, float | str]] | None = None,
        dampingQ: float = 0.0,
        animationLW: int = 5,
    ):
        if isinstance(anchor0, Model):  # this is the first boom of the crane
            self._model = anchor0
            self._anchor0 = None
            self._model.boom0 = self  # register the first boom in the model (other booms are accessed through iterator)
        elif isinstance(anchor0, Boom):  # append this boom to anchor0
            self._model = anchor0._model
            self._anchor0 = anchor0
            self._anchor0.anchor1 = self  # register this boom as the Boom fixed to the previous
        else:
            raise BoomInitError(
                "The anchor0 parameter of a Boom must be a Crane object or a Boom object. Found: " + str(type(anchor0))
            )
        self._name = name
        self.description = description
        self._anchor1 = None  # so far the following boom is unknown (set when instantiating the next)
        self.velocity = np.array(
            (0, 0, 0), dtype="float64"
        )  # records the current velocity of the c_m, both with respect to angualar movement (e.g. torque from angular acceleration) and linear movement (e.g. rope)
        self.animationLW = animationLW
        self._mass = Variable(
            self._model,
            self._name + "_mass",
            "The total mass of boom " + name,
            causality="parameter",
            variability="fixed",
            value0=mass,
        )
        self.mass = getattr(self._model, self._name + "_mass")  # access to value (owned by model)
        if not len(str(self._mass.unit)):
            print(f"Warning: Missing unit for mass of boom {self._name}. Include that in the 'mass' parameter")
        self._boom = VariableNP(
            self._model,
            self._name + "_boom",
            "The dimension and direction of the boom from anchor point to anchor point in m and spherical angles",
            causality="input",
            variability="continuous",
            value0=boom,
            rng=boomRng,
        )
        self.boom = getattr(self._model, self._name + "_boom")  # access to value (owned by model)
        assert (
            self.boom[0] > 0 and self._boom.range[0][0] > 0
        ), f"The length of boom {self._name} can become zero, which should not be allowed"

        self.direction = spherical_to_cartesian(self._boom.value0)  # cartesian direction of boom
        if self._anchor0 is None:  # the first boom, anchored at fixed platform or on vessel
            self.point0 = np.array((0, 0, 0), dtype="float64")  # origo
        else:
            self.point0 = anchor0.point1  # the starting point of the boom in the coordinate system of the first boom

        self._tip = VariableNP(  # expose point1
            self._model,
            name + "_tip",
            description="Cartesian vector of the tip (point1) of the boom",
            causality="output",
            variability="continuous",
            value0=(self.point0 + self.direction),
            getter=lambda: self.point1,
        )
        doF = [
            0,
            0,
            0,
        ]  # denotes the degrees of freedom of the boom (length and rotation degrees of freedom in polar and azimuth direction). Most booms have only one degree of freedom.
        if isinstance(boomRng, tuple):  # the degrees of freedom are defined
            for i, rng in enumerate(boomRng):
                if rng is not None and (isinstance(rng, tuple) and rng[0] != rng[1]):
                    doF[i] = 1  # is a degree of freedom
        self.doF = tuple(doF)  # save as non-mutable element
        if self.doF[1] > 0 and self.doF[2] == 0:  # only polar rotations allowed
            self.axis = np.array((0, 1, 0), dtype="float64")
        elif self.doF[2] > 0 and self.doF[1] == 0:  # only azimuth rotations allowed
            self.axis = np.array((0, 0, 1), dtype="float64")
        else:  # undefined. Rotation axis must be supplied explicitly
            self.axis = None  # type: ignore
        self._centerOfMass = VariableNP(
            self._model,
            self._name + "_centerOfMass",
            "Position of the Center of Mass along the boom (line), relative to the total length",
            causality="input",
            variability="continuous",
            value0=((centerOfMass, 0, 0) if not isinstance(centerOfMass, tuple) else centerOfMass),
            rng=((1e-6, 1.0), (None, None), (None, None)),
        )
        self.centerOfMass = getattr(self._model, self._name + "_centerOfMass")  # access to value (owned by model)
        self._c_m = self.c_m  # save the current value, running method self.c_m
        self._c_m_sub = [
            self.mass,
            self._c_m,
        ]  # 'isolated' value as placeholder. Updated by calc_statics_dynamics

        # some input variables (connectors)
        self.angularVelocity = getattr(self._model, self._name + "_angularVelocity")  # access to value (owned by model)
        self._angularVelocity = Variable(
            self._model,
            name=self._name + "_angularVelocity",
            description="Rotates boom according to its defined degree of freedom. Only for boom which initiates the rotation!",
            causality="input",
            variability="continuous",
            value0="0.0 rad/s",
            rng=(),
            on_step=lambda t, dT: (self.rotate(angle=self.angularVelocity) if self.angularVelocity != 0 else None),
        )

        self.lengthVelocity = getattr(self._model, self._name + "_lengthVelocity")  # access to value (owned by model)
        self._lengthVelocity = Variable(
            self._model,
            name=self._name + "_lengthVelocity",
            description="Changes length of the boom (ifa allowed). Only for boom which initiates the length change!",
            causality="input",
            variability="continuous",
            value0="0.0 m/s",
            rng=(0, float("inf")),
            on_step=lambda t, dT: (
                self.change_length(dL=self.lengthVelocity, dT=dT) if self.lengthVelocity != 0 else None
            ),
        )

        self.dampingQ = dampingQ
        if self.dampingQ != 0.0:
            msg = f"The damping quality of {self.name} should be 0 or >0.5. Provided {self.dampingQ}"
            assert dampingQ > 0.5, msg
            self._decayRate = self.calc_decayrate(self.length)
        # do a total re-calculation of _c_m_sub and torque (static) for this boom (trivial) and the reverse connected booms
        for b in self.iter(reverse=True):
            b.calc_statics_dynamics(dT=None)
        self.force = np.array((0, 0, 0), dtype="float64")  # ensure proper initial value

    #        print("BOOM " +self._name +" EndPoints: " +str(self.point0) +", " +str(self.point1) +" dir, length, dampingQ: " +str(self.direction) +", " +str(self.length) +", " +str(self.dampingQ))

    def iter(self, reverse=False):
        """Define an iterator over booms. If reverse=True, find initially the last boom and iterate in reverse order."""
        b = self
        if reverse:  # find the last boom
            while b.anchor1 is not None:
                b = b.anchor1
        else:
            while b.anchor0 is not None:
                b = b.anchor0
        while b is not None:
            yield b
            b = b.anchor0 if reverse else b.anchor1

    def __getitem__(self, idx):
        """Facilitate subscripting booms. 'idx' denotes the connected boom with respect to self.
        Negative indices count from the tail. str indices identify booms by name.
        """
        b = self
        if isinstance(idx, str):  # retrieve by name
            while True:
                if b is None:
                    return None
                elif b.name == idx:
                    return b
                b = b.anchor1

        elif idx >= 0:
            for _ in range(idx):
                b = b.anchor1
                if b is None:
                    raise IndexError("Erroneous index " + str(idx) + " with respect to boom " + self.name)
            return b
        else:
            while b.anchor1 is not None:  # spool to tail
                b = b.anchor1
            for _ in range(abs(idx) - 1):  # go back from tail
                b = b.anchor0
                if b is None:
                    raise IndexError("Erroneous index " + str(idx) + " with respect to boom " + self.name)
            return b

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    @property
    def length(self):
        return np.linalg.norm(self.direction)

    @property
    def anchor0(self):
        return self._anchor0

    @property
    def anchor1(self):
        return self._anchor1

    @anchor1.setter  # called by a child boom to set the attached boom
    def anchor1(self, newVal):
        self._anchor1 = newVal

    @property
    def point1(self):
        return self.point0 + self.direction

    @property
    def c_m(self):
        """Return the local center of mass point relative to self.point0."""
        return self.centerOfMass[0] * self.direction + np.array((self.centerOfMass[1], self.centerOfMass[2], 0))

    @property
    def c_m_absolute(self):
        """Return the local center of mass point as absolute position, assuming that _c_m is updated."""
        return self.point0 + self._c_m

    @property
    def c_m_sub(self):
        """Return the system center of mass as absolute position
        The _c_m_sub needs to be calculated/updated using calc_statics_dynamics.
        """
        return self._c_m_sub

    def calc_statics_dynamics(self, dT: float | None = None):
        """After any movement the local c_m and the c_m of connected booms have changed.
        Thus, after the movement has been transmitted to connected booms, the _c_m_sub of all booms can be updated in reverse order.
        The local _c_m_sub is updated by calling this function, assuming that forward connected booms are updated.
        While updating, also the velocity, the torque (with respect to point0) and the linear force are calculated.
        Since there might happen multiple movements within a time interval, the function must be called explicitly, i.e. from crane.

        Args:
            dT (float)=None: for dynamic calculations, the time for the movement must be provided
              and is then used to calculate velocity, acceleration, torque and force
        """
        if dT is not None:
            c_m_sub1 = np.array(self._c_m_sub[1], dtype="float64")  # make a copy
        if self.anchor1 is None:  # there are no attached booms
            # assuming that _c_m is updated. Note that _c_m_sub is a global vector
            self._c_m_sub = [self.mass, self.point0 + self._c_m]
        else:  # there are attached boom(s)
            # this should be updated if calc_statics_dynamics has been run for attached boom
            [mS, posS] = self.anchor1.c_m_sub
            m = self.mass
            cs = self.c_m_absolute  # the local center of mass as absolute position
            # updated _c_m_sub as absolute position
            self._c_m_sub = [mS + m, (cs * m + mS * posS) / (mS + m)]

        self.torque = self._c_m_sub[0] * np.cross(self.direction, np.array((0, 0, -9.81)))  # static torque
        if dT is not None:  # the time for the movement is provided (dynamic analysis)
            velocity0 = np.array(self.velocity)
            # check for pendulum movements and implement for this time interval if relevant
            self.velocity, acceleration = self._pendulum(dT)
            if self.velocity is None:
                # there was no pendulum movement and the velocity has thus not been calculated. Calculate from _c_m_sub
                self.velocity = (self._c_m_sub[1] - c_m_sub1) / dT
                acceleration = (self.velocity - velocity0) / dT
            self.torque += self._c_m_sub[0] * np.cross(self.direction, acceleration)
            # linear force due to acceleration in boom direction
            self.force = self._c_m_sub[0] / self.length * np.dot(self.direction, acceleration)

    def _pendulum(self, dT: float):
        """For a non-stiff connection, if the _c_m is not exactly below point0, the _c_m acts as a damped pendulum.
        See also `wikipedia article <https://de.wikipedia.org/wiki/Sph%C3%A4risches_Pendel>`_ (the English article is not as good) for detailed formulas
        `with respect to damping: <https://en.wikipedia.org/wiki/Damping>`_
        Note: falling movements (i.e. rope acceleration larger than g) are not allowed (raise error).

        Pendulum movements are integrated into calc_statistics_dynamics if the connection to the boom is non-stiff (dampingQ!=0)

        Args:
            dT (float): The time interval for which the pendulum movement is calculated

        Returns
        -------
            updated velocity and acceleration (of c_m)

        .. assumption:: the center of mass is on the boom line at _centerOfMass[0] relative distance from point0
        ..toDo:: for high initial velocities the energy increases! Check that.
        """
        if self.dampingQ != 0.0:
            assert self.anchor1 is None, "Pendulum movement is so far only implemented for the last boom (the rope)"
            # center of mass factor (we look on the c_m with respect to pendulum movements):
            c = self.centerOfMass[0]
            assert (
                np.linalg.norm(self.direction) >= self.length
            ), f"Rope end (load) falling movements are currently not implemented. Length: {self.length}. Direction: {self.direction}"
            R = c * self.length  # pendulum radius
            r = c * np.array(self.direction)  # the current radius (of c_m) as copy
            rDot = self.velocity  # this is correct if _c_m == _c_m_sub
            if R > 1e-6 and abs(self.direction[2] / self.length) > 1e-10:  # pendulum movement
                term1 = np.cross(r, np.cross(r, np.array((0.0, 0.0, -9.81), dtype="float64"))) / (R * R)
                term2 = np.dot(rDot, rDot) / (R * R) * r
                acceleration = -(term1 + term2 + self._decayRate * rDot)
                newVelocity = rDot + acceleration * dT
                newPosition = r + newVelocity * dT
            else:  # no pendulum movement
                newVelocity = rDot
                newPosition = r
                acceleration = np.array((0.0, 0.0, 0.0), dtype="float64")

            # ensure that the length is unchanged:
            newPosition *= R / np.linalg.norm(newPosition)
            #                print("ENERGY", 0.5*np.dot(newVelocity,newVelocity) + 9.81*(R-np.dot( newPosition, np.array( (0,0,-1), dtype='float64'))))
            self.direction = newPosition / c
            self._c_m = newPosition
            # we return these two for further usage and registration within calc_statics_dynamics:
            return (newVelocity, acceleration)
        # signal to calc_statics_dynamics that velocity and acceleration are not yet calculated
        return (None, None)

    def calc_decayrate(self, newLength):
        if self.dampingQ == 0.0:
            return None
        elif newLength == 0.0:
            return 0
        else:
            return sqrt(9.81 / (newLength * self.centerOfMass[0])) / sqrt(4 * self.dampingQ - 1)

    def rotate(
        self,
        origin: np.ndarray | None = None,
        axis: np.ndarray | None = None,
        angle: float = 0.0,
        rot: Rot | None = None,
        asDeg: bool = False,
        static: bool = False,
    ):
        """Rotate the boom around a rotation axis with respect to origo with angle.

        Args:
            origin (np.ndarray)=None: Optional specification of an alternative origin. E.g. if one of the parent booms is turning
            axis (np.ndarray,tuple)=None: specification of the turning axis as np.ndarray or tuple
              For rotation initiating booms None is acceptable if there is a unique degree of freedom
            angle (float)=0.0: The turning angle
            rot (Rot)=None: alternative specification of rotation as Rotation object
            asDeg (bool)=False: Optional possibility to provide the angle in degrees
            static (bool)=False: Optionally assume static boom movement (relevant only for rope - dampingQ!=0)
              If static=True it is assumed that the rotation happens in infinite time, so that rope direction wrt. origin remains constant (down).
        """
        if rot is None:  # the rotation is not fully specified through the rotation object
            if angle == 0.0:
                return  # nothing to rotate
            if origin is None and axis is None:
                # the rotation is initiated from this boom and the axis is intended specified 'automatic'
                msg = "Undefined rotation axis. Either the rotation of a boom needs to be unique or the axis must be supplied explicitly"
                assert self.axis is not None, msg
                axis = self.axis
            msg = f"{self.name}. The angle and axis cannot be None (keep rotation) if the rotation is undefined"
            assert angle is not None and axis is not None, msg
            rot = Rot.from_rotvec(angle * axis, degrees=asDeg)  # define the rotation object

        if rot.magnitude() == 0.0:  # nothing to rotate
            return
        p0 = np.array(self.point0)  # keep a copy (needed for rope calculation below)
        if origin is not None:  # rotation initiated from boom we are connecting to
            self.point0 = self.anchor0.point1  # adapt point0 to the rotation of the previous boom
            if self.axis is not None:
                self.axis = rot.apply(self.axis)  # rotate the axis itself

        if self.dampingQ == 0.0:  # stiff boom hinge. Adapt self.direction
            #            print("BOOM.rotate", self.name, origin, rot.as_rotvec(), rot.magnitude(), self.direction, end='')
            self.direction = rot.apply(self.direction)
            #            print(" => ", self.direction)
            self._c_m = self.c_m  # update the relative position of the center of mass
        elif not static:  # with a rope, the c_m tries to stay where it was. Only the (max) rope length is ensured
            length = self.length
            newDir = p0 - self.point0 + self._c_m
            newLen = np.linalg.norm(newDir)
            self.direction = newDir * length / newLen
            self._c_m = self.c_m  # measured relative to point0 and must therefore be updated

        if self._anchor1 is not None:  # cascade the rotation to connected booms
            if origin is None:
                # we need to relate to point0 of this boom, as the originator of the rotation
                self.anchor1.rotate(origin=self.point0, rot=rot, static=static)
            else:
                self.anchor1.rotate(origin=origin, rot=rot, static=static)  # keep the origin
        # call from crane: self.calc_statics_dynamics( dT, origin is None) # re-calculate the static and dynamic properties and inform parent booms if needed

    def translate(self, vec: tuple[float] | np.ndarray):
        """Translate the boom and its connected booms by vector vec.
        This should always be initiated from the first boom or from a length-changing boom,
        since it otherwise disconnects the system.

        Args:
            vec (tuple,np.ndarray): the 3D translation vector
        """
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec, dtype="float64")
        self.point0 += vec
        self._c_m += vec
        if self._anchor1 is not None:
            self._anchor1.translate(vec)
        # call from crane: self.calc_statics_dynamics( dT, isInitiator) # re-calculate the static and dynamic properties and inform parent booms if needed

    def change_length(self, dL: float, isInitiator=True, dT: float | None = None):
        """Change the length of the boom (if allowed)
        Note: Instantaneous length velocity changes are accepted, even if they create (small) unrealistic falling movements.

        Args:
            dL (float): length change
            isInitiator (bool)=True: denotes whether this boom is the initiator of the movement
            dT (float)=None: Optional possibility to provide an explicit time for dynamic calculations
        """
        assert self._boom._range[0] is not None, f"The length of boom {self.name} is not changeable."
        # ToDO: make better test
        # need to remember that (as copy):
        before = np.array(self.point1, dtype="float64")
        L = self.length
        assert L + dL > 1e-6, f"Boom length {L} cannot become negative or zero. Change: {dL}."
        if self.dampingQ == 0 or dL <= 0:  # stiff boom or boom (rope) getting shorter
            relDL = (L + dL) / L if L > 0 else dL
            self.direction = relDL * self.direction
        else:  # rope and dL>0 => change right down
            self.direction[2] -= dL
        #            print(self.direction, 'rho, polar, azimuth:', rho, degrees(polar), degrees(azimuth))
        self._c_m = self.c_m
        self._decayRate = self.calc_decayrate(L + dL)
        # cascade the length change to connected booms. For connected booms this becomes a translation
        if self._anchor1 is not None:
            self._anchor1.translate(self.point1 - before, isInitiator=False, dT=dT)
        # call from crane: self.calc_statics_dynamics( dT, isInitiator) # re-calculate the static and dynamic properties and inform parent booms if needed

    def change_mass(self, dM: float, relCOM: float | None = None):
        """Change the mass of the boom, e.g. when adding or releasing a load at the rope.

        Args:
            dM (float): The added or subtracted mass
            relCOM (float)=None: Optional possibility to change the relative c_m point along the boom (between 1e-6 and 1.0), i.e. changing self.centerOfMass

        .. note:: We treat mass changes as non-dynamic effect (dT=None), since the change in c_m position should not be associated with a velocity or acceleration
        .. note:: Mass changes have no direct effect on attached boom (which do normally not exist, since the load is often attached to the last boom)
        """
        if relCOM is not None:
            self.centerOfMass[0] = relCOM
        self.mass += dM
        self._c_m = self.c_m  # re-calculate the own COM
        # call from crane: self.calc_statistics_dynamics( dT=None, isInitiator=True) # re-calculate the static properties and inform parent booms of the change in c_m
