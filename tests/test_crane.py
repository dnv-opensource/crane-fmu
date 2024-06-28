from math import radians, sqrt, pi

import matplotlib.pyplot as plt
import numpy as np
from component_model.model import Model
from component_model.variable import spherical_to_cartesian
from crane_fmu.boom import Boom
from crane_fmu.crane import Crane
from crane_fmu.logger import get_module_logger
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pytest

# from mpl_toolkits.mplot3d.art3d import Line3D
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore
from scipy.spatial.transform import Rotation as Rot

logger = get_module_logger(__name__, level=1)

def np_arrays_equal(arr1, arr2, dtype="float64", eps=1e-7):
    assert len(arr1) == len(arr2), "Length not equal!"
    if isinstance(arr2, (tuple, list)):
        arr2 = np.array(arr2, dtype=dtype)
    assert isinstance(arr1, np.ndarray) and isinstance(
        arr2, np.ndarray
    ), "At least one of the parameters is not an ndarray!"
    assert arr1.dtype == arr2.dtype, f"Arrays are of type {arr1.dtype} != {arr2.dtype}"

    for i in range(len(arr1)):
        assert abs(arr1[i] - arr2[i]) < eps, f"Component {i}: {arr1[i]} != {arr2[i]}"


def aligned(pi):
    """Check whether all points pi are on the same straight line."""
    assert (
        len(pi) > 2
    ), f"Checking whether points are on the same line should include at least 3 points. Got only {len(pi)}"
    directions = [pi[i] - pi[0] for i in range(1, len(pi))]
    n_dir0 = directions[0] / np.linalg.norm(directions[0])
    for i in range(1, len(directions)):
        np_arrays_equal(n_dir0, directions[i] / np.linalg.norm(directions[i]))

def pendulum_relax( rope:Boom, show:bool, steps:int=1000, dt:float=0.01):
    x = []
    for i in range(steps): # let the pendulum relax
        rope.calc_statics_dynamics(dt)
        x.append( rope.end[2])
    if show:
        fig, ax = plt.subplots()
        ax.plot( x)
        plt.show()


@pytest.fixture
def crane(scope="module", autouse=True):
    Model.instances = []
    crane = Crane("crane")
    pedestal = crane.add_boom(
        name="pedestal",
        description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
        mass="2000.0 kg",
        centerOfMass=(0.5, "-1m", "0.8m"),
        boom=("3.0 m", "0deg", "0deg"),
        boom_rng=(None, None, ("0deg", "360deg")),
    )
    boom1 = crane.add_boom(
        name="boom1",
        description="The first boom. Can be lifted",
        mass="200.0 kg",
        centerOfMass=(0.5, 0, 0),
        boom=("10.0 m", "90deg", "0deg"),
        boom_rng=(None, ("-90deg", "90deg"), None),
    )
    boom2 = crane.add_boom(
        name="boom2",
        description="The second boom. Can be lifted whole range",
        mass="100.0 kg",
        centerOfMass=(0.5, 0, 0),
        boom=("5.0 m", "-180deg", "0deg"),
        boom_rng=(None, ("-180deg", "180deg"), None),
    )
    rope = crane.add_boom(
        name="rope",
        description="The rope fixed to the last boom. Flexible connection",
        mass="50.0 kg",  # so far basically the hook
        centerOfMass=1.0,
        boom=("0.5 m", "180deg", "0 deg"),
        boom_rng=(("0.5m", "20m"), ("90deg", "270deg"), ("-180deg", "180deg")),
        dampingQ=2,
    )
    return crane

#@pytest.mark.skip()
def test_initial(crane):
    """Test the initial state of the crane."""
    # test indexing of booms
    booms = [b.name for b in crane.booms()]
    assert booms == ['fixation', 'pedestal', 'boom1', 'boom2', 'rope']
    fixation, pedestal, boom1, boom2, rope  = [b for b in crane.booms()]
    assert crane.boom0.name == "fixation", "Boom0 should be fixation"
    bs = crane.booms()  # iterator generator for the booms based on crane object
    next(bs)
    assert next(bs).name == "pedestal", "Should be 'pedestal'"
    bs = crane.booms(reverse=True)
    assert next(bs).name == "rope", "First reversed should be 'rope'"
    assert next(bs).name == "boom2", "Next reversed should be 'boom2'"

    assert pedestal[0].name == "pedestal", "pedestal[0] should be 'pedestal'"
    assert pedestal[1].name == "boom1", "pedestal[1] should be 'boom1'"
    assert pedestal[-1].name == "rope", "pedestal[-1] should be 'rope'"
    assert pedestal[-2].name == "boom2", "pedestal[-2] should be 'boom2'"

    #for i,b in enumerate(crane.booms()):
    #    print( f"Boom {i}: {b.name}")
    assert list(crane.booms())[2].name == "boom1", "'boom1' from boom iteration expected"
    assert list(crane.booms(reverse=True))[1].name == "boom2", "'boom2' from reversed boom iteration expected"

    assert pedestal.length == 3.0
    assert boom1.length == 10.0
    assert boom2.length == 5.0
    assert pedestal.anchor1.name == "boom1"
    assert boom1.anchor1.name == "boom2"
    assert pedestal.name == "pedestal"
    assert pedestal.mass == 2000.0, f"Found {pedestal.mass}"

    np_arrays_equal(pedestal.origin, (0, 0, 0))
    np_arrays_equal(pedestal.direction, (0, 0, 1))
    assert pedestal.length == 3
    assert pedestal._mass.unit == "kilogram"
    np_arrays_equal(crane.pedestal_end, boom1.origin)
    np_arrays_equal(boom1.origin, (0, 0, 3.0))
    np_arrays_equal(boom1.direction, (1, 0, 0))
    assert boom1.length == 10
    np_arrays_equal(crane.boom1_end, boom2.origin)
    np_arrays_equal(boom2.origin, (10, 0, 3))
    np_arrays_equal(boom2.direction, (-1, 0, 0))
    assert boom2.length == 5
    np_arrays_equal(crane.boom2_end, (5,0,3))
    np_arrays_equal(rope.origin, (5, 0, 3))
    np_arrays_equal(rope.end, (5, 0, 2.5))


#@pytest.mark.skip()
def test_sequence(crane, show = False):
    f, p, b1, b2, r = [b for b in crane.booms()]
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    np_arrays_equal(r.origin + r.length*r.direction, (5, 0, 2.5))
    np_arrays_equal(crane.rope_end, (5, 0, 2.5))
    # boom1 up
    b1.boom_setter((None,0, None))
    np_arrays_equal(r.end, (0+0.5/sqrt(2), 0, 3 + 10 - 5 - 0.5/sqrt(2)), eps=0.1) # somewhat lower due to length
    pendulum_relax( r, show=show)
    np_arrays_equal(r.end, (0, 0, 3 + 10 - 5 - 0.5), eps=0.001) # equilibrium position
    # boom1 45 deg up
    b1.boom_setter( (None, radians(45), None))
    pendulum_relax( r, show=show)
    np_arrays_equal(r.end, [(10 - 5) / sqrt(2), 0, 3 - 0.5 + (10 - 5) / (sqrt(2))], eps=0.001)
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    # boom2 0 deg (in line with boom1)
    b2.boom_setter( (None, radians(0), None))
    pendulum_relax( r, show=show)
    np_arrays_equal(r.end, [(10 + 5) / sqrt(2), 0, 3 - 0.5 + (10 + 5) / (sqrt(2))], eps=0.001)
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    # rope 0.5m -> 5m
    r.boom_setter( (5.0, None, None))
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    np_arrays_equal(r.end, [(10 + 5) / sqrt(2), 0, 3 - 5 + (10 + 5) / (sqrt(2))], eps=0.001)
    # turn base 45 deg
    p.boom_setter( (None, None, radians(45)))
    pendulum_relax( r, show=show)
    np_arrays_equal( r.end, ((10 + 5) / sqrt(2) / sqrt(2), (10 + 5) / sqrt(2) / sqrt(2), 3 - 5 + (10 + 5) / (sqrt(2))),
                     eps=0.001)
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    # boom1 up. Dynamic
    c_m_0 = r.origin + r.c_m  # c_m before rotation
    len_0 = r.length
    for angle in range(45,0,-1):
        b1.boom_setter( (None, angle, None))
        crane.calc_statics_dynamics(1.0)
    assert len_0 == r.length, "Length of rope has changed!"


#@pytest.mark.skip()
def test_change_length(crane, show: bool = False):
    f, p, b1, b2, r = [b for b in crane.booms()]
    if show:
        show_it(crane)
    b2.boom_setter( (None, 0, None))
    assert r.anchor1 is None
    np_arrays_equal(b2.end, r.origin)
    r.boom_setter( (3,None,None))
    pendulum_relax(r, show)
    np_arrays_equal(r.end, (15.0, 0, 0), eps=0.001)
    np_arrays_equal(r.end, r._end.getter())
    if show:
        show_it(crane)


#@pytest.mark.skip()
def test_boom_position(crane, show: bool = False):
    """Testing boom positions which are also used in MobileCrane.cases"""
    f,p, b1, b2, r = [b for b in crane.booms()]
    np_arrays_equal( crane.pedestal_end, (0,0,3))
    np_arrays_equal( crane.boom1_end, (10,0,3))
    np_arrays_equal( crane.boom2_end, (5,0,3))
    p._boom.setter( np.array( (3,0,pi/2))) # represents an initial setting, boom1 in y-direction
    np_arrays_equal( b1.end, (0,10,3.0))


#@pytest.mark.skip()
def test_rotation(crane, show: bool = False):
    f, p, b1, b2, r = [b for b in crane.booms()]
    b1.boom_setter( (None, 0, None))  # b1 up
    np_arrays_equal(b1.direction, (0, 0, 1))
    assert b1.length == 10
    np_arrays_equal(b2.origin, (0, 0, 13))
    np_arrays_equal(b2.end, (0, 0, 8))
    np_arrays_equal(b1.c_m, (0, 0, 5))  # measured relative to its origin!
    np_arrays_equal(b2.c_m, (0, 0, -2.5))  # measured relative to its origin!
    np_arrays_equal(r.origin, (0, 0, 8))
    assert abs(r.length - 0.5) < 1e-10, f"Unexpected length {r.length}"
    if show:
        show_it(crane, markCOM=True, markSubCOM=False)

    b1.boom_setter( (None, radians(90), None))  # b1 east (as initially)
    np_arrays_equal(b1.direction, (1, 0, 0))
    assert b1.length == 10
    np_arrays_equal(b1.c_m, (5, 0, 0))
    np_arrays_equal(b2.direction, (-1, 0, 0))
    assert b2.length == 5
    np_arrays_equal(b2.c_m, (-2.5, 0, 0))
    if show:
        show_it(crane, markCOM=True, markSubCOM=False)

    p.boom_setter( (None, None, radians(-90)))  # turn p so that b1 south
    np_arrays_equal(b1.direction, (0, -1, 0))

    np_arrays_equal(b1.c_m, (0, -5, 0))
    np_arrays_equal(b2.direction, (0, 1, 0))
    assert b2.length == 5
    np_arrays_equal(b2.c_m, (0, 2.5, 0))
    if show:
        show_it(crane, markCOM=True, markSubCOM=False)


#@pytest.mark.skip()
def test_c_m(crane, show: bool = False):
    # Note: Boom.c_m is a local measure, calculated from Boom.origin
    f, p, b1, b2, r = [b for b in crane.booms()]
    r.change_mass(-50) # to get the simple c_m calculations right
    if show:
        show_it(crane, markCOM=True, markSubCOM=False)
    # initial c_m location
    #        print("Initial c_m:", p.c_m, b1.c_m, b2.c_m)
    np_arrays_equal(p.c_m, (-1, 0.8, 1.5)) # 2000 kg
    np_arrays_equal(b1.c_m, (5, 0, 0))     # 200 kg
    np_arrays_equal(b2.c_m, (-2.5, 0, 0))  # 100 kg
    #        self.show_it( (p,b1,b2), markCOM=True, markSubCOM=True)
    # all booms along a line in z-direction
    b2.boom_setter( (None, 0, None))
    b1.boom_setter( (None, 0, None))
    #        self.show_it( (p,b1,b2), markCOM=True, markSubCOM=True)
    np_arrays_equal(b1.c_m, (0, 0, 5))
    np_arrays_equal(b2.c_m, (0, 0, 2.5))
    # update all subsystem center of mass points. Need to do that from last boom!
    crane.calc_statics_dynamics(dT=None)
    if show:
        show_it(crane, markCOM=True, markSubCOM=False)
    np_arrays_equal(b2.c_m_sub[1], b2.origin + b2.c_m)
    np_arrays_equal(b1.c_m_sub[1], (0, 0, 3 + 5 + 100 / 300 * (15.5 - 8)))
    assert p.c_m_sub[1][2] == 1.5 + 300 / 2300 * (b1.c_m_sub[1][2] - 1.5)
    b1.boom_setter( (None, radians(45), None))
    b2.boom_setter( (None, radians(30), None))
    p.boom_setter( (None, None, radians(-20)))


def animate_sequence(pedestal, seq=(), nSteps=10):
    """Generate animation frames for a sequence of rotations. To be used as 'update' argument in FuncAnimation.
    A sequence element consists of a boom and an angle, which then is rotated in nSteps.
    To do updates of statics and dynamics we need to know the last boom.
    """
    for [b, a] in seq:
        aStep = a / nSteps
        for _ in range(nSteps):
            b.rotate(angle=aStep, asDeg=True)
            # update all subsystem center of mass points. Need to do that from last boom!
            pedestal.model.calc_statics_dynamics(dT=None)
            yield (pedestal)

@pytest.mark.skip("Animate crane movement")
def test_animation(crane):
    p, b1, b2, r = [b for b in crane.boom0.iter()]

    def init():
        """Perform the needed initializations."""
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 10)
        b = crane.boom0
        while True:
            if b.name == "pedestal":
                lw = 10
            elif b.name == "rope":
                lw = 2
            else:
                lw = 5
            lines.append(
                ax.plot(
                    [b.origin[0], (b.origin + b.direction)[0]],
                    [b.origin[1], (b.origin + b.direction)[1]],
                    [b.origin[2], (b.origin + b.direction)[2]],
                    linewidth=lw,
                )
            )
            if b.anchor1 is None:
                break
            else:
                b = b.anchor1

    def update(p):
        """Based on the updated first boom (i.e. the whole crane), draw any desired data"""
        i = 0
        b = p
        while True:
            lines[i][0].set_data_3d(
                [b.origin[0], (b.origin + b.direction)[0]],
                [b.origin[1], (b.origin + b.direction)[1]],
                [b.origin[2], (b.origin + b.direction)[2]],
            )
            i += 1
            if b.anchor1 is None:
                break
            else:
                b = b.anchor1

    p, b1, b2, r = list(crane.booms())
    fig = plt.figure(figsize=(9, 9), layout="constrained")
    ax = plt.axes(projection="3d")  # , data=line)
    lines = []

    _ = FuncAnimation(
        fig,
        update,
        frames=animate_sequence(p, seq=((p, -90), (b1, -45), (b2, 180))),
        init_func=init,
        interval=1000,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    np_arrays_equal(r.origin, (0, -15 / sqrt(2), 3 + 15 / sqrt(2)))


def show_it(_crane, markCOM=True, markSubCOM=True):
    # update all subsystem center of mass points. Need to do that from last boom!
    _crane.calc_statics_dynamics(dT=None)
    fig = plt.figure(figsize=(9, 9), layout="constrained")
    ax = Axes3D(fig=fig)
    # ax = plt.axes(projection="3d")  # , data=line)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 10)

    lines = []
    c_ms = []
    subCOMs = []
    for i, b in enumerate(_crane.booms()):
        lw = 10 if i == 0 else 5
        lines.append(
            ax.plot(
                [b.origin[0], b.end[0]],
                [b.origin[1], b.end[1]],
                [b.origin[2], b.end[2]],
                linewidth=lw,
            )
        )
        if markCOM:
            #                print("SHOW_COM", b.name, b.c_m)
            c_ms.append(
                ax.text(
                    b.c_m[0],
                    b.c_m[1],
                    b.c_m[2],
                    s=str(int(b._mass.value0)),
                    color="black",
                )
            )
        if markSubCOM:
            [_, x] = b.c_m_sub
            #                print("SHOW_SUB_COM", m, x)
            subCOMs.append(ax.plot(x[0], x[1], x[2], marker="*", color="red"))

    #         print( lines)
    #         print( c_ms)
    #         print( subCOMs)
    #         lines[0][0].set_data_3d( [1,2], [3,4], [5,6])
    #         c_ms[0].set_x( -2.0)
    #         c_ms[0].set_y( 2.0)
    #         c_ms[0].set_z( 3.0)
    #         subCOMs[0][0].set_data_3d( [5,5], [5,5] ,[5,5])
    plt.show()


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Return code {retcode}"
#    c = crane()
    #test_initial( )
