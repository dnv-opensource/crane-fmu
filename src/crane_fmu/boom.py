from __future__ import annotations

from math import sqrt

import numpy as np
from component_model.model import Model  # type: ignore
from component_model.variable import Variable, spherical_to_cartesian  # type: ignore


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
        massCenter (float,tuple): Parameter denoting the (assumed fixed) position of the center of mass of the boom,
          provided as portion of the length (as float) and optionally the absolute displacements in x- and y-direction (assuming the boom in z-direction),
          e.g. (0.5,'-0.5 m','1m'): halfway down the boom displaced 0.5m in the -x direction and 1m in the y direction
        boom (tuple): A tuple defining the boom relative in spherical (ISO 80000) coordinates

           * origin: crane origin or end of connecting boom => cartesian origin
           * pole axis: crane z-direction or direction vector of connecting boom => local cartesian z-axis
           * reference direction in equator plane: crane x-direction or azimuth angle of connecting boom => local cartesian x-axis

           => coordinates:

           * length: the length of the boom (in length units)
           * polar: a rotation angle for a rotation around the negative x-axis (away from z-axis) against the clock.
           * azimuth: a rotation angle for a rotation around the positive z-axis against the clock.

          Note: The boom and its range is used to keep length and local coordinate system up-to-date,
          while the active work variables are the cartesian origin and direction (cartesian boom vector)
        boommRng (tuple): Range for each of the boom components, relative to the z-axis, i.e. how much the boom can be rotated/lengthened with respect to the z-axis.
          As normal, range components specified as None denote fixed components. Most booms have only one (rotation) degree of freedom.
        damping (float)=0.0: optional possibility to implement a loose connection between booms (damping>0),
          e.g. the crane rope is implemented as a stiff boom of variable length with a loose connection hanging from the previous boom.

          The damping denotes the dimensionless damping quality factor (energy stored/energy lost per radian),
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
                        massCenter = (0.5, 0,'2 deg'),
                        boom        = ('5.0 m', 0, '0deg'),
                        boom_rng      = (None, (0,'360 deg'), None)


    .. todo:: determine the range of forces
    .. limitation:: The mass and the massCenter setting of booms is assumed constant. With respect to rope and hook of a crane this means that basically only the mass of the hook is modelled.
    .. assumption:: Center of mass: _c_m is the local c_m measured relative to origin. _c_m_sub is a global quantity
    """

    def __init__(
        self,
        model: Model,
        name: str,
        description: str,
        anchor0: Boom | None = None,
        mass: str = "1 kg",
        mass_rng: tuple | None = None,
        massCenter: float | tuple = 0.5,
        boom: tuple = (1, 0, 0),
        boom_rng: tuple = tuple(),
        damping: float = 0.0,
        animationLW: int = 5,
    ):
        self._model = model
        self.anchor0 = anchor0
        self.anchor1: Boom | None = None  # so far. If a boom is added, this is changed
        self._name = name
        self.description = description
        self.damping = damping
        self.direction = np.array((0, 0, -1), dtype="float64")  # default for non-fixed booms
        self.velocity = np.array((0, 0, 0), dtype="float64")
        #    records the current velocity of the c_m, both with respect to angualar movement (e.g. torque from angular acceleration) and linear movement (e.g. rope)
        self.animationLW = animationLW
        if self.anchor0 is None:  # this defines the fixation of the crane as a 'pseudo-boom'
            boom = (1e-10, 0, 0)  # z-axis in spherical coordinates
            boom_rng = tuple()
            self.origin = np.array((0, 0, -1e-10), dtype="float64")
        else:
            self.origin = self.anchor0.end
            self.anchor0.anchor1 = self
        self.mass = self._interface("mass", mass, mass_rng)
        self.massCenter = list(massCenter) if isinstance(massCenter, tuple) else [massCenter, 0.0, 0.0]
        self.boom = self._interface("boom", boom, boom_rng)
        self.base_angles = self.get_base_angles()
        self.direction = self.get_direction()
        _ = self._interface("end", self.origin + self.length * self.direction)  #! local value is a function (property)
        self._c_m = self.c_m  # save the current value, running method self.c_m
        self._c_m_sub = [self.mass, self._c_m]  # 'isolated' value as placeholder. Updated by calc_statics_dynamics
        if self.damping != 0.0:
            if damping < 0.5:
                raise BoomInitError(f"Damping quality {self.damping} of {self.name} should be 0 or >0.5.") from None
            self._decayRate = self.calc_decayrate(self.length)
        # do a total re-calculation of _c_m_sub and torque (static) for this boom (trivial) and the reverse connected booms
        self.angularVelocity = self._interface(
            "angularVelocity", (0, 0)
        )  # initial value always set to (0,0) with units
        self.lengthVelocity = self._interface("lengthVelocity", 0.0)
        self.torque = self._interface("torque", ("0 N*m", "0 N*m", "0 N*m"))
        self.force = self._interface("force", ("0 N", "0 N", "0 N"))

        self.calc_statics_dynamics(dT=None)
        # print("BOOM " +self._name +" EndPoints: " +str(self.origin) +", " +str(self.end) +" dir, length, damping: " +str(self.direction) +", " +str(self.length) +", " +str(self.damping))

    def _interface(self, name: str, start: str | float | tuple, rng: tuple | None = None):
        """Define interface variables.
        The function is kept separate from the code above to not clutter it too much.
        Arguments are passed on from __init__, as these concern interface issues like range.
        All variable values are registered with the model as self.name+'_'+name,
        such that they are accessible within the boom as self.(variable-name)
        and within the model as self.(boom-name)_(variable-name).
        In addition, the variable object itself is registered as self._(variable-name).
        """
        if name == "mass":
            self._mass = Variable(
                self._model,
                owner=self,
                name=self._name + "_mass",
                description="The total mass of boom " + name,
                causality="input",
                variability="continuous",
                start=start,
                rng=rng,
            )
            if not len(self._mass.unit):
                print(f"Warning: Missing unit for mass of boom {self._name}. Include that in the 'mass' parameter")
        elif name == "boom":
            assert isinstance(
                start, (tuple, list, np.ndarray)
            ), f"The boom variable of {self.name} needs a proper 3D start value"
            if start[0] == 0:
                start = ("0 m", *start[1:])
            for i in range(1, 3):
                if start[i] == 0:
                    start = (*start[:i], "0" + self._model.u_angle, *start[i + 1 :])
                elif not isinstance(start[i], str) or self._model.u_angle not in start[i]:
                    raise BoomInitError(f"All angles shall be provided as {self._model.u_angle}")
            self._boom = Variable(
                self._model,
                owner=self,
                name=self._name + "_boom",
                description="The dimension and direction of the boom from anchor point to anchor point in m and spherical angles",
                causality="input",
                variability="continuous",
                start=start,
                rng=rng,
                on_set=self.boom_setter,
            )
        elif name == "end":
            self._end = Variable(
                self._model,
                owner=self,
                name=self._name + "_end",
                description="Cartesian vector of the end of the boom",
                causality="output",
                variability="continuous",
                start=start,
            )
            self._end.getter = lambda: self.end
        elif name == "angularVelocity":
            self._angularVelocity = Variable(
                self._model,
                owner=self,
                name=self._name + "_angularVelocity",
                description="Rotates boom arround origin according to its defined degree of freedom (polar/azimuth).",
                causality="input",
                variability="continuous",
                start=("0 " + self._model.u_angle, "0 " + self._model.u_angle),
                # on_set= self._angular_velocity_setter, #lambda v: v if hasattr(v,'__iter__') else ( (v,0) if boom_rng[1] is not None else (0,v)),
                on_step=self.angular_velocity_step,
                rng=(),
            )
            self._angularVelocity.setter = self._angular_velocity_setter
        elif name == "lengthVelocity":
            self._lengthVelocity = Variable(
                self._model,
                owner=self,
                name=self._name + "_lengthVelocity",
                description="Changes length of the boom (ifa allowed). Only for boom which initiates the length change!",
                causality="input",
                variability="continuous",
                start=start,
                rng=(),
            )
        elif name == "torque":
            self._torque = Variable(
                self._model,
                owner=self,
                name=self._name + "_torque",
                description="""Torque contribution of the boom with respect to its origin,
                               i.e. the sum of static and dynamic torques. Provided as 3D cartesian vector""",
                causality="output",
                variability="continuous",
                initial="exact",
                start=start,
            )
        elif name == "force":
            self._force = Variable(
                self._model,
                owner=self,
                name=self._name + "_force",
                description="Total linear force of the crane with respect to its base, i.e. the sum of static and dynamic forces. Provided as 3D cartesian vector)",
                causality="output",
                variability="continuous",
                initial="exact",
                start=start,
            )
        return getattr(self._model, self._name + "_" + name)

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

    #     @property
    #     def boom(self):
    #         return getattr(self._model, self._name + "_boom")  # access to value (owned by model)

    def boom_setter(self, val: np.ndarray | tuple):
        """Set length and angles of boom (if allowed) and ensure consistency with other booms.
        This is called from the general setter function after the units and range are checked
        and before the variable value itself is changed within the model.

        Args:
            val (array-like): new value of boom. Elements of the array can be set to None (keep value)
            initial (bool)=False: optional possibility to configure the boom initially, avoiding dynamics.
        """
        type_change = 0  # bit coded
        if not hasattr(self, "boom"):  # not yet initialized
            initial = True
            self.boom = val
        else:
            initial = False
        length = self.length  # remember the previous length
        for i in range(3):
            if val[i] is not None and val[i] != self.boom[i]:
                if i > 0 and self.damping != 0 and not initial:
                    print("WARNING. Attempt to directly set the angle of a rope. Does not make sense")
                    return
                else:
                    self.boom[i] = val[i]
                    type_change |= 1 << i
        if self.damping != 0:
            if length < self.boom[0] and not initial:  # non-stiff connection (increased rope length)
                self.direction = self.get_direction()
            elif initial:
                self.direction = normalized(spherical_to_cartesian((self.length, *self.boom[1:])))
        elif type_change > 1:  # not only a length change. direction must be updated
            self.direction = self.get_direction()

        if self.anchor1 is not None:
            self.anchor1.update_child()
        return self.boom

    def _angular_velocity_setter(self, v: float | tuple | list | np.ndarray, idx: int | None = None):
        """Set angularVelocity."""
        rng = self._boom.range  # Note: boom is 3-dim, but angularVelocity is 2-dim
        arg = [0.0, 0.0]
        if rng[1][0] != rng[1][1] and rng[2][0] != rng[2][1]:  # two rotation degrees of freedom
            if idx is not None and isinstance(v, float):
                arg[idx] = v
                arg[1 - idx] = self.angularVelocity[1 - idx]  # leave the other alone
            elif idx is None and isinstance(v, (tuple, list, np.ndarray)):
                arg = list(v)
            else:
                raise BoomOperationError(
                    f"Angular velocity of boom {self.name} requires both polar and azimuth: {v}, {idx}, {hasattr(v,'__itter__')}"
                )
        else:
            for i in range(1, 3):
                if rng[i][0] != rng[i][1] and rng[3 - i][0] == rng[3 - i][1]:
                    arg[i - 1] = v if isinstance(v, float) else v[i - 1]
                    break
        self.angularVelocity = np.array(arg, float)

    def angular_velocity_step(self, t, dt):
        """Step angular velocity. As this is the derivative of boom angles, boom angles are stepped."""
        if self.angularVelocity[0] != 0 and self.angularVelocity[1] != 0:
            self.boom_setter((None, self.boom[1] + self.angularVelocity[0], self.boom[2] + self.angularVelocity[1]))
        elif self.angularVelocity[0] != 0:
            self.boom_setter((None, self.boom[1] + self.angularVelocity[0], None))
        elif self.angularVelocity[1] != 0:
            self.boom_setter((None, None, self.boom[2] + self.angularVelocity[1]))

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    @property
    def length(self):
        return self.boom[0]

    @property
    def end(self):
        return self.origin + self.length * self.direction

    @property
    def c_m(self):
        """Return the local center of mass point relative to self.origin."""
        return self.massCenter[0] * self.length * self.direction + np.array((self.massCenter[1], self.massCenter[2], 0))

    @property
    def c_m_sub(self):
        """Return the system center of mass as absolute position
        The _c_m_sub needs to be calculated/updated using calc_statics_dynamics.
        """
        return self._c_m_sub

    def get_base_angles(self):
        """Azimuth and polar angles of parent."""
        if self.anchor0 is None:
            return [0, 0]
        else:
            return self.anchor0.boom[1:] + self.anchor0.get_base_angles()

    def get_direction(self):
        """Get the new direction vector (cartesian) after a change in crane (base_angles, local angles, boom length).

        * fixed boom: Fixed connection. Angle conserved.
        * rope: center of mass of rope tries to stay in position, but length is unchanged.
          => if this would make the length longer, the new direction is anchor->com
             if this would make the length shorter, the c.o.m falls vertically, keeping the length constant.

        Note: self.origin is updated after get_direction() is run,
          such that self.origin represents the anchor position before the movement,
          while self.anchor1.end represents the new anchor position.
        """
        if self.damping > 0:  # flexible joint (rope)
            com_len = self.length * self.massCenter[0]  # length between anchor and c.o.m (unchanged!)
            com0 = self.origin + com_len * self.direction  # the previous absolute c.o.m. point
            anchor_to_com = com0 - self.anchor0.end
            anchor_to_com_len = np.linalg.norm(anchor_to_com)
            if anchor_to_com_len >= com_len:  # rope is dragged (or unchanged). Normalize anchor_to_com
                return anchor_to_com / anchor_to_com_len
            else:  # rope falls, keeping length constant
                anchor_to_com[2] = -sqrt(com_len**2 - anchor_to_com[0] ** 2 - anchor_to_com[1] ** 2)
                # we choose only the negative z-komponent, excluding loads in upper half
                return anchor_to_com / com_len
        else:
            _angle = self.base_angles + self.boom[1:]
            return normalized(spherical_to_cartesian((self.length, *_angle)))

    def update_child(self):
        """Update this boom after the parent boom has changed length or angles."""
        if self.anchor0 is not None:
            self.base_angles = self.anchor0.base_angles + self.anchor0.boom[1:]
            self.direction = self.get_direction()
            self.origin = self.anchor0.end  # do that last, so that the previous value remains available
            if self.anchor1 is not None:
                self.anchor1.update_child()

    def translate(self, vec: tuple | np.ndarray, cnt: int = 0):
        """Translate the whole crane. Can obviously only instantiated by the first boom."""
        if isinstance(vec, tuple):
            vec = np.array(vec, dtype="float64")
        if cnt > 0 or self.anchor0 is None:  # can only be initiated by base!
            self.origin += vec
            if self.anchor1 is not None:
                self.anchor1.translate(vec, cnt + 1)

    def calc_statics_dynamics(self, dT: float | None = None):
        """After any movement the local c_m and the c_m of connected booms have changed.
        Thus, after the movement has been transmitted to connected booms, the _c_m_sub of all booms can be updated in reverse order.
        The local _c_m_sub is updated by calling this function, assuming that forward connected booms are updated.
        While updating, also the velocity, the torque (with respect to origin) and the linear force are calculated.
        Since there might happen multiple movements within a time interval, the function must be called explicitly, i.e. from crane.

        Args:
            dT (float)=None: for dynamic calculations, the time for the movement must be provided
              and is then used to calculate velocity, acceleration, torque and force
        """
        if dT is not None:
            c_m_sub1 = np.array(self._c_m_sub[1], dtype="float64")  # make a copy
        if self.anchor1 is None:  # there are no attached booms
            # assuming that _c_m is updated. Note that _c_m_sub is a global vector
            self._c_m_sub = [self.mass, self.origin + self._c_m]
        else:  # there are attached boom(s)
            # this should be updated if calc_statics_dynamics has been run for attached boom
            [mS, posS] = self.anchor1.c_m_sub
            m = self.mass
            cs = self.origin + self.c_m  # the local center of mass as absolute position
            # updated _c_m_sub as absolute position
            self._c_m_sub = [mS + m, (cs * m + mS * posS) / (mS + m)]
        self.torque = self._c_m_sub[0] * np.cross(self._c_m_sub[1], np.array((0, 0, -9.81)))  # static torque
        if dT is not None:  # the time for the movement is provided (dynamic analysis)
            velocity0 = np.array(self.velocity)
            # check for pendulum movements and implement for this time interval if relevant
            self.velocity, acceleration = self._pendulum(dT)
            if self.velocity is None:
                # there was no pendulum movement and the velocity has thus not been calculated. Calculate from _c_m_sub
                self.velocity = (self._c_m_sub[1] - c_m_sub1) / dT
                acceleration = (self.velocity - velocity0) / dT
            assert (
                np.linalg.norm(self.velocity) < 1e50
            ), f"The velocity {self.velocity} is far too high. Time intervals too large?"
            self.torque += self._c_m_sub[0] * np.cross(self._c_m_sub[1], acceleration)
            # linear force due to acceleration in boom direction
            self.force = self._c_m_sub[0] * np.dot(self.direction, acceleration) * self.direction

        # Ensure that links between variable values on Boom level and model level are maintained:
        # setattr( self._model, self._torque.local_name, self.torque)
        # setattr( self._model, self._force.local_name, self.force)
        if self.anchor0 is not None:
            self.anchor0.calc_statics_dynamics(dT)

    def _pendulum(self, dt: float):
        r"""For a non-stiff connection, if the _c_m is not exactly below origin, the _c_m acts as a damped pendulum.
        See also `wikipedia article <https://de.wikipedia.org/wiki/Sph%C3%A4risches_Pendel>`_ (the English article is not as good) for detailed formulas
        `with respect to damping: <https://en.wikipedia.org/wiki/Damping>`_
        Note: falling movements (i.e. rope acceleration larger than g) are not allowed (raise error).

        Pendulum movements are integrated into calc_statistics_dynamics if the connection to the boom is non-stiff (damping!=0)

        Args:
            dt (float): The time interval for which the pendulum movement is calculated

        Returns
        -------
            updated velocity and acceleration (of c_m)

        .. assumption:: the center of mass is on the boom line at _massCenter[0] relative distance from origin
        .. math::

            \\ddot\vec r=-frac{\vec r \\cross (\vec r \\ cross \vec g)}{R^2} - frac{\\dot\vec r^2}{R^2} \vec r

        ..toDo:: for high initial velocities the energy increases! Check that.
        """

        def update_r_dr(r, dr_dt, dt):
            """Update position and speed using time step dt. Return updated values as tuple."""
            gravitational = np.cross(r, np.cross(r, np.array((0.0, 0.0, -9.81), dtype="float64"))) / (R * R)
            centripetal = np.dot(dr_dt, dr_dt) / (R * R) * r
            acc = -(gravitational + centripetal + self._decayRate * dr_dt)
            r += dr_dt * dt + 0.5 * acc * dt * dt
            dr_dt += acc * dt
            return (r, dr_dt, acc)

        if self.damping != 0.0:
            assert self.anchor1 is None, "Pendulum movement is so far only implemented for the last boom (the rope)"
            # center of mass factor (we look on the c_m with respect to pendulum movements):
            c = self.massCenter[0]
            R = c * self.length  # pendulum radius
            r = R * self.direction  # the current radius vector (of c_m) as copy
            dr_dt = self.velocity  # velocity at start of interval. This is correct if _c_m == _c_m_sub
            if R > 1e-6 and abs(self.direction[2]) > 1e-10:  # pendulum movement
                max_dv = 0.00001  # maximum allowed speed change in one sub-iteration step
                t = 0.0
                _dt = 1e-6
                while t < dt:
                    (r0, dr_dt0, acc) = update_r_dr(r, dr_dt, _dt)
                    (r1, dr_dt1, acc) = update_r_dr(r, dr_dt, _dt / 2)
                    (r2, dr_dt2, acc) = update_r_dr(r1, dr_dt1, _dt / 2)
                    abs_dr = abs(np.linalg.norm(dr_dt2) - np.linalg.norm(dr_dt0))
                    if abs_dr < 1e-10 or abs_dr / np.linalg.norm(dr_dt) < max_dv:  # accuracy ok
                        t += _dt
                        r = r2
                        dr_dt = dr_dt2
                        if abs_dr < 1e-10 or abs_dr / np.linalg.norm(dr_dt) < 0.5 * max_dv:  # accuracy too expensive
                            _dt *= 2  # try doubling _dt
                    else:  # accuracy not good enough
                        _dt *= 0.5  # retry with half interval
                        assert _dt > 1e-12, f"The step width {_dt} got unacceptably small in pendulum calculation"
                        print(f"Retry @{t}: {_dt}")

                # print(f"Pendulum. grav:{gravitational}, cent:{centripetal}, decay:{self._decayRate * rDot}. acc:{acc}")
                # ensure that the length is unchanged:
                r *= R / np.linalg.norm(r)
            else:  # no pendulum movement
                acc = np.array((0.0, 0.0, 0.0), dtype="float64")

            # print("ENERGY", 0.5*np.dot(newVelocity,newVelocity) + 9.81*(R-np.dot( newPosition, np.array( (0,0,-1), dtype='float64'))))
            self.direction = normalized(r)
            self._c_m = r
            # we return these two for further usage and registration within calc_statics_dynamics:
            return (dr_dt, acc)
        return (None, None)  # signal to calc_statics_dynamics that velocity and acceleration are not yet calculated

    def calc_decayrate(self, newLength):
        if self.damping == 0.0:
            return None
        elif newLength == 0.0:
            return 0
        else:
            return sqrt(9.81 / (newLength * self.massCenter[0])) / sqrt(4 * self.damping - 1)

    def change_length(self, dL: float):
        """Change the length of the boom (if allowed)
        Note: Instantaneous length velocity changes are accepted, even if they create (small) unrealistic falling movements.

        Args:
            dL (float): length change
        """
        self.boom_setter((self.boom[0] + dL, None, None))

    def change_mass(self, dM: float, center: float | None = None):
        """Change the mass of the boom, e.g. when adding or releasing a load at the rope.

        Args:
            dM (float): The added or subtracted mass
            relCOM (float)=None: Optional possibility to change the relative c_m point along the boom (between 1e-6 and 1.0), i.e. changing self.massCenter

        .. note:: We treat mass changes as non-dynamic effect (dt=None), since the change in c_m position should not be associated with a velocity or acceleration
        .. note:: Mass changes have no direct effect on attached boom (which do normally not exist, since the load is often attached to the last boom)
        """
        if center is not None:
            self.massCenter[0] = center
        self.mass += dM
        self._c_m = self.c_m  # re-calculate the own COM
        # call from crane: self.calc_statistics_dynamics( dt=None, isInitiator=True) # re-calculate the static properties and inform parent booms of the change in c_m


def normalized(vec: np.ndarray):
    assert len(vec) == 3, f"{vec} should be a 3-dim vector"
    norm = np.linalg.norm(vec)
    assert norm > 0, f"Zero norm detected for vector {vec}"
    return vec / norm
