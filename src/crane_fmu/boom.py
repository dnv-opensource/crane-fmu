from __future__ import annotations

from math import sqrt

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
        boom (tuple): A tuple defining the boom relative in spherical (ISO 80000) coordinates

           * origin: crane origin or tip of connecting boom => cartesian origin
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
                        boom_rng      = (None, (0,'360 deg'), None)


    .. todo:: determine the range of forces
    .. limitation:: The mass and the centerOfMass setting of booms is assumed constant. With respect to rope and hook of a crane this means that basically only the mass of the hook is modelled.
    .. assumption:: Center of mass: _c_m is the local c_m measured relative to origin. _c_m_sub is a global quantity
    """

    def __init__(
        self,
        model: Model,
        name: str,
        description: str,
        anchor0: Boom | None = None,
        mass: str | None = None,
        mass_rng: tuple | None = None,
        centerOfMass: float | tuple = 0.5,
        boom: tuple | None = None,
        boom_rng: tuple | None = None,
        dampingQ: float = 0.0,
        animationLW: int = 5,
    ):
        self._model = model
        self.anchor0 = anchor0
        self.anchor1 : Boom | None = None # so far. If a boom is added, this is changed
        self._name = name
        self.description = description
        self.dampingQ = dampingQ
        self.direction = np.array( (0,0,-1), dtype='float64') # default for non-fixed booms
        self.velocity = np.array( (0, 0, 0), dtype="float64")
        #    records the current velocity of the c_m, both with respect to angualar movement (e.g. torque from angular acceleration) and linear movement (e.g. rope)
        self.animationLW = animationLW
        if self.anchor0 is None: # this defines the fixation of the crane as a 'pseudo-boom'
            boom = (1e-10,0,0) #z-axis in spherical coordinates
            boom_rng = tuple()
            self.origin = np.array( (0,0,-1e-10), dtype='float64')
        else:
            self.origin = self.anchor0.end
            self.anchor0.anchor1 = self
        self._mass = Variable(
            self._model,
            self._name + "_mass",
            "The total mass of boom " + name,
            causality="input",
            variability="continuous",
            rng=mass_rng,
            value0=mass,
        )
        self.mass = getattr(self._model, self._name + "_mass")  # access to value (owned by model)
        if not len(str(self._mass.unit)):
            print(f"Warning: Missing unit for mass of boom {self._name}. Include that in the 'mass' parameter")
        self._centerOfMass = VariableNP(
            self._model,
            self._name + "_centerOfMass",
            "Position of the Center of Mass along the boom (line), relative to the total length",
            causality="input",
            variability="continuous",
            value0=((centerOfMass, 0, 0) if not isinstance(centerOfMass, tuple) else centerOfMass),
            rng=((1e-6, 1.0), (None, None), (None, None)),
        )
        self.centerOfMass =getattr(self._model, self._name + "_centerOfMass")  # access to value (owned by model)

        self._boom = VariableNP(
            self._model,
            self._name + "_boom",
            "The dimension and direction of the boom from anchor point to anchor point in m and spherical angles",
            causality = "input",
            variability = "continuous",
            value0 = boom,
            rng = boom_rng,
            on_set = self.boom_setter,
        )
        self.base_angles = self.get_base_angles()
        self.direction = self.get_direction()
        self._end = VariableNP(  # expose end
            self._model,
            self._name + "_end",
            description="Cartesian vector of the tip (end) of the boom",
            causality="output",
            variability="continuous",
            value0 = self.end,
            getter = lambda: self.end,
        )

        self._c_m = self.c_m  # save the current value, running method self.c_m
        self._c_m_sub = [self.mass, self._c_m]  # 'isolated' value as placeholder. Updated by calc_statics_dynamics

        # some input variables (connectors)
        self._angularVelocity = VariableNP(
            self._model,
            name=self._name + "_angularVelocity",
            description="Rotates boom arround origin according to its defined degree of freedom (polar/azimuth).",
            causality="input",
            variability="continuous",
            value0= ("0.0 deg/s", "0.0 deg/s"),
            on_step = self.angular_velocity_step,
            rng=(),
        )
        self.angularVelocity = getattr(self._model, self._name + "_angularVelocity")  # access to value (owned by model)

        self._lengthVelocity = Variable(
            self._model,
            name=self._name + "_lengthVelocity",
            description="Changes length of the boom (ifa allowed). Only for boom which initiates the length change!",
            causality="input",
            variability="continuous",
            value0="0.0 m/s",
            rng=(0, float("inf")),
        )
        self.lengthVelocity = getattr(self._model, self._name + "_lengthVelocity")  # access to value (owned by model)
        self._lengthVelocity.on_step=lambda t, dT: (
                self.change_length(dL=self.lengthVelocity, dT=dT) if self.lengthVelocity != 0 else None
            )

        if self.dampingQ != 0.0:
            msg = f"The damping quality of {self.name} should be 0 or >0.5. Provided {self.dampingQ}"
            assert dampingQ > 0.5, msg
            self._decayRate = self.calc_decayrate(self.length)
        # do a total re-calculation of _c_m_sub and torque (static) for this boom (trivial) and the reverse connected booms
        self.calc_statics_dynamics(dT=None)
        self.force = np.array((0, 0, 0), dtype="float64")  # ensure proper initial value

        # print("BOOM " +self._name +" EndPoints: " +str(self.origin) +", " +str(self.end) +" dir, length, dampingQ: " +str(self.direction) +", " +str(self.length) +", " +str(self.dampingQ))

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
    def boom(self):
        return getattr(self._model, self._name + "_boom")  # access to value (owned by model)

    # @boom.setter
    def boom_setter(self, val:np.ndarray|tuple, initial:bool=False):
        """Set length and angles of boom (if allowed) and ensure consistency with other booms.
        This is called from the general setter function after the units and range are checked
        and before the variable value itself is changed within the model.

        Args:
            val (array-like): new value of boom. Elements of the array can be set to None (keep value)
            initial (bool)=False: optional possibility to configure the boom initially, avoiding dynamics.
        """
        type_change = 0 # bit coded
        if not hasattr(self, 'boom'): # not yet initialized
            setattr(self, 'boom', val)
        length = self.length # remember the previous length
        for i in range(3):
            if val[i] is not None and val[i] != self.boom[i]:
                if i>0 and self.dampingQ != 0 and not initial:
                    print("WARNING. Attempt to set directly set the angle of a rope. Does not make sense")
                    return
                else:
                    self.boom[i] = val[i]
                    type_change |= (1 << i)
        if self.dampingQ != 0:
            if length < self.boom[0] and not initial: # non-stiff connection (increased rope length)
                self.direction = self.dir_rope_falling( self.direction)
            elif initial:
                self.direction = normalized(spherical_to_cartesian( (self.length, *self.boom[1:])))
        elif type_change > 1: # not only a length change. direction must be updated
            self.direction = self.get_direction()

        print(f"BOOM.boom_setter {self.name}->{val}, initial:{initial}, dir:{self.direction}, length:{self.length}, anchor:{self.origin}")
        if self.anchor1 is not None:
            self.anchor1.update_child()
        return self.boom

    def angular_velocity_step(self, t, dt):
        """Step angular velocity. As this is the derivative of boom angles, boom angles are stepped"""
        if self.angularVelocity[0] != 0 and self.angularVelocity[1] != 0:
            self.boom_setter( (None, self.boom[1]+self.angularVelocity[0], self.boom[2]+self.angularVelocity[1]))
        elif self.angularVelocity[0] != 0:
            self.boom_setter( (None, self.boom[1]+self.angularVelocity[0], None))
        elif self.angularVelocity[1] != 0:
            self.boom_setter( (None, self.boom[2]+self.angularVelocity[1], None))

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
        return self.origin + self.length*self.direction

    @property
    def c_m(self):
        """Return the local center of mass point relative to self.origin."""
        return self.centerOfMass[0] * self.length* self.direction + np.array((self.centerOfMass[1], self.centerOfMass[2], 0))

    @property
    def c_m_sub(self):
        """Return the system center of mass as absolute position
        The _c_m_sub needs to be calculated/updated using calc_statics_dynamics.
        """
        return self._c_m_sub


    def get_base_angles(self):
        """Azimuth and polar angles of parent"""
        if self.anchor0 is None:
            return [0,0]
        else:
            return self.anchor0.boom[1:] + self.anchor0.get_base_angles()

    def dir_rope_falling(self, dir0):
        """Calculate an updated direction when the rope is falling, observing base-direction and length"""
        return normalized( dir0 + (sqrt( self.boom[0]**2 - dir0[0]**2 - dir0[1]**2) - dir0[2])* np.array( (0,0,-1)))

    def get_direction(self):
        """Get the new direction vector after a change in crane (base_angles, local angles, boom length)."""
        if self.dampingQ > 0: # flexible joint with previous boom (rope)
            # center of mass of rope tries to stay in same vertical line, even if parent moves. Rope length conserved.
            fac = self.centerOfMass[0] # fraction of length where c.o.m is located
            com0 = self.origin + self.length*self.direction*fac # the previous absolute c.o.m. point
            len1 = np.linalg.norm( self.anchor0.end - com0) / fac
            dir1 = normalized( com0 - self.anchor0.end) # shorten to length in direction of com0
            if len1 < self.length: # rope falls, keeping length constant
                dir1 = self.dir_rope_falling( dir1)
            return dir1
        else:
            _angle = self.base_angles + self.boom[1:]
            return normalized(spherical_to_cartesian( (self.length, *_angle)))


    def update_child(self):
        """Update this boom after the parent boom has changed length or angles."""
        if self.anchor0 is not None:
            self.base_angles = self.anchor0.base_angles + self.anchor0.boom[1:]
            self.direction = self.get_direction()
            self.origin = self.anchor0.end # do that last, so that the previous value remains available
            if self.anchor1 is not None:
                self.anchor1.update_child()


    def translate(self, vec:tuple | np.ndarray, cnt:int=0):
        """Translation of the whole crane. Can obviously only instantiated by the first boom."""
        if isinstance(vec, tuple):
            vec = np.array( vec, dtype="float64")
        if cnt>0 or self.anchor0 is None: # can only be initiated by base!
            self.origin += vec
            if self.anchor1 is not None:
                self.anchor1.translate( vec, cnt+1)


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
            cs = self.origin + self.c_m # the local center of mass as absolute position
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
        if self.anchor0 is not None:
            self.anchor0.calc_statics_dynamics( dT)

    def _pendulum(self, dT: float):
        """For a non-stiff connection, if the _c_m is not exactly below origin, the _c_m acts as a damped pendulum.
        See also `wikipedia article <https://de.wikipedia.org/wiki/Sph%C3%A4risches_Pendel>`_ (the English article is not as good) for detailed formulas
        `with respect to damping: <https://en.wikipedia.org/wiki/Damping>`_
        Note: falling movements (i.e. rope acceleration larger than g) are not allowed (raise error).

        Pendulum movements are integrated into calc_statistics_dynamics if the connection to the boom is non-stiff (dampingQ!=0)

        Args:
            dT (float): The time interval for which the pendulum movement is calculated

        Returns
        -------
            updated velocity and acceleration (of c_m)

        .. assumption:: the center of mass is on the boom line at _centerOfMass[0] relative distance from origin
        ..toDo:: for high initial velocities the energy increases! Check that.
        """
        if self.dampingQ != 0.0:
            assert self.anchor1 is None, "Pendulum movement is so far only implemented for the last boom (the rope)"
            # center of mass factor (we look on the c_m with respect to pendulum movements):
            c = self.centerOfMass[0]
            R = c * self.length  # pendulum radius
            r = R * self.direction  # the current radius vector (of c_m) as copy
            rDot = self.velocity  # this is correct if _c_m == _c_m_sub
            if R > 1e-6 and abs(self.direction[2]) > 1e-10:  # pendulum movement
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
            self.direction = normalized(newPosition)
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


    def change_length(self, dL: float):
        """Change the length of the boom (if allowed)
        Note: Instantaneous length velocity changes are accepted, even if they create (small) unrealistic falling movements.

        Args:
            dL (float): length change
        """
        self.boom_setter( (self.boom[0]+dL, None, None))

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

def normalized( vec:np.ndarray):
    return vec / np.linalg.norm( vec)
