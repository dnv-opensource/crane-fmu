from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from component_model.model import Model
from crane_fmu.boom import Boom
from crane_fmu.crane import Crane
from crane_fmu.logger import get_module_logger
from matplotlib.animation import FuncAnimation

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


def test_rot():
    """Basic usage of the scipy rotation functions"""
    z90 = Rot.from_rotvec(np.pi / 2 * np.array([0, 0, 1]))  # 90 deg rotation around Z
    print(z90)
    z45 = Rot.from_rotvec(np.array((0, 0, np.pi / 4)))
    print(z45)
    np_arrays_equal(z90.apply(np.array([1, 0, 0])), (0, 1, 0))
    # rotation around the point (-1,0,0):
    origin = np.array((-1, 0, 0))
    np_arrays_equal(z90.apply(np.array((1, 0, 0)) - origin) + origin, (-1, 2, 0))


def test_crane(includeRope=True):
    Model.instances = []
    crane = Crane("crane")
    pedestal = Boom(
        name="pedestal",
        description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
        anchor0=crane,
        mass="2000.0 kg",
        centerOfMass=(0.5, "-1m", "0.8m"),
        boom=("3.0 m", "0deg", "0deg"),
        boomRng=(None, None, ("0deg", "360deg")),
    )
    boom1 = Boom(
        name="boom 1",
        description="The first boom. Can be lifted",
        anchor0=pedestal,
        mass="200.0 kg",
        centerOfMass=(0.5, 0, 0),
        boom=("10.0 m", "90deg", "0deg"),
        boomRng=(None, ("-90deg", "90deg"), None),
    )
    boom2 = Boom(
        name="boom 2",
        description="The second boom. Can be lifted whole range",
        anchor0=boom1,
        mass="100.0 kg",
        centerOfMass=(0.5, 0, 0),
        boom=("5.0 m", "90deg", "180deg"),
        boomRng=(None, ("-180deg", "180deg"), None),
    )
    if includeRope:
        rope = Boom(
            name="rope",
            description="The rope fixed to the last boom. Flexible connection",
            anchor0=boom2,
            mass="50.0 kg",  # so far basically the hook
            centerOfMass=0.95,
            boom=("0.5 m", "180deg", "0 deg"),
            boomRng=(("0.5m", "20m"), ("90deg", "270deg"), ("-180deg", "180deg")),
            dampingQ=50,
        )
    test_initial(crane, onlyVariable=False)
    if includeRope:
        return crane, pedestal, boom1, boom2, rope
    else:
        return crane, pedestal, boom1, boom2


def test_initial(crane: Crane, onlyVariable: bool = True):
    """Test the initial state of the crane."""
    # test indexing of booms
    booms = [b for b in crane.boom0.iter()]
    if len(booms) == 3:
        [pedestal, boom1, boom2] = booms
        rope = None
    else:
        [pedestal, boom1, boom2, rope] = booms
    if not onlyVariable:
        assert crane.boom0.name == "pedestal", "Boom0 should be pedestal"
        bs = crane.booms()  # iterator generator for the booms based on crane object
        next(bs)
        assert next(bs).name == "boom 1", "Should be 'boom 1'"
        bs = crane.booms(reverse=True)
        if rope is None:
            b = next(bs)
            assert (
                b.name == "boom 2"
            ), f"First reversed should be 'boom2'. Found {b.name}"
        else:
            assert next(bs).name == "rope", "First reversed should be 'rope'"
            assert next(bs).name == "boom 2", "Next reversed should be 'boom 2'"

        assert pedestal[0].name == "pedestal", "pedestal[0] should be 'pedestal'"
        assert pedestal[1].name == "boom 1", "pedestal[1] should be 'boom 1'"
        if rope is None:
            assert pedestal[-1].name == "boom 2", "pedestal[-1] should be 'boom 2'"
        else:
            assert pedestal[-1].name == "rope", "pedestal[-1] should be 'rope'"
            assert pedestal[-2].name == "boom 2", "pedestal[-2] should be 'boom 2'"

        assert (
            list(pedestal.iter())[1].name == "boom 1"
        ), "'boom 1' from boom iteration expected"
        if rope is None:
            assert (
                list(boom2.iter(reverse=True))[1].name == "boom 1"
            ), "'boom 1' from reversed boom iteration expected"
        else:
            assert (
                list(rope.iter(reverse=True))[1].name == "boom 2"
            ), "'boom 2' from reversed boom iteration expected"

        assert pedestal.length == 3.0
        assert boom1.length == 10.0
        assert boom2.length == 5.0
        assert pedestal.anchor1.name == "boom 1"
        assert boom1.anchor1.name == "boom 2"
        assert pedestal.name == "pedestal"
        assert pedestal.mass == 2000.0, f"Found {pedestal.mass}"

    np_arrays_equal(pedestal.point0, (0, 0, 0))
    np_arrays_equal(pedestal.direction, (0, 0, 3))
    assert pedestal._mass.unit == "kilogram"
    np_arrays_equal(boom1.point0, (0, 0, 3.0))
    np_arrays_equal(boom1.direction, (10, 0, 0))
    np_arrays_equal(boom2.point0, (10, 0, 3))
    np_arrays_equal(boom2.direction, (-5, 0, 0))
    if rope is not None:
        np_arrays_equal(rope.point0, (5, 0, 3))
        np_arrays_equal(rope.point1, (5, 0, 2.5))


def test_sequence(show: bool = True):
    crane, p, b1, b2, r = test_crane()
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    np_arrays_equal(r.point0 + r.direction, (5, 0, 2.5))
    # boom1 up
    b1.rotate(angle=-90, asDeg=True, static=True)
    np_arrays_equal(r.point0 + r.direction, (0, 0, 3 + 10 - 5 - 0.5))
    # boom1 45 deg up
    b1.rotate(angle=+45, asDeg=True, static=True)
    np_arrays_equal(
        r.point0 + r.direction, [(10 - 5) / sqrt(2), 0, 3 - 0.5 + (10 - 5) / (sqrt(2))]
    )
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    # boom2 180 deg (in line with boom1)
    b2.rotate(angle=180, asDeg=True, static=True)
    np_arrays_equal(
        r.point0 + r.direction, [(10 + 5) / sqrt(2), 0, 3 - 0.5 + (10 + 5) / (sqrt(2))]
    )
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    # rope 0.5m -> 5m
    r.change_length(4.5)
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    np_arrays_equal(
        r.point0 + r.direction, [(10 + 5) / sqrt(2), 0, 3 - 5 + (10 + 5) / (sqrt(2))]
    )
    # turn base 45 deg
    p.rotate(angle=45, asDeg=True, static=True)
    np_arrays_equal(
        r.point0 + r.direction,
        [
            (10 + 5) / sqrt(2) / sqrt(2),
            (10 + 5) / sqrt(2) / sqrt(2),
            3 - 5 + (10 + 5) / (sqrt(2)),
        ],
    )
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)

    crane, p, b1, b2, r = test_crane()
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    # boom1 up. Dynamic
    c_m_0 = r.point0 + r.c_m  # c_m before rotation
    len_0 = r.length
    b1.rotate(angle=-90, asDeg=True, static=False)
    aligned([r.point0, c_m_0, r.point0 + r.c_m])
    assert len_0 == r.length, "Length of rope has changed!"


def test_rotate_instant(show: bool = True):
    crane, p, b1, b2, r = test_crane()
    r.change_length(5 / 0.95 - r.length)  # rope c_m on z=-2
    len0 = r.length
    a_c_m0 = r.c_m_absolute
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)
    b1.rotate(angle=90, asDeg=True, static=False)
    assert (
        len0 == r.length
    ), "Length {len0} should not change during rotation. Found {r.length} after"
    np_arrays_equal(
        a_c_m0, r.c_m_absolute
    )  # instantaneous rotations try to keep the c_m constant
    #    print(f"AFTER: rop0={r.point0}, abs.c_m={r.c_m_absolute}, length={r.length}, rope1={r.point0+r.direction}")
    if show:
        show_it(crane, markCOM=True, markSubCOM=True)


def test_change_length(show: bool = True):
    crane, p, b1, b2, r = test_crane()
    if show:
        show_it(crane)
    b2.rotate(angle=180, asDeg=True, static=True)
    assert r.anchor1 is None
    np_arrays_equal(r.point1, r._tip.getter())
    r.change_length(dL=2.5)
    np_arrays_equal(r.point1, (15.0, 0, 0))
    np_arrays_equal(r.point1, r._tip.getter())
    if show:
        show_it(crane)


def test_rotation(show: bool = True):
    crane, p, b1, b2, r = test_crane()
    b1.rotate(angle=-90, asDeg=True)  # b1 up
    np_arrays_equal(b1.direction, (0, 0, 10))
    np_arrays_equal(b2.point0, (0, 0, 13))
    np_arrays_equal(b2.point1, (0, 0, 8))
    np_arrays_equal(b1.c_m, (0, 0, 5))  # measured relative to its point0!
    np_arrays_equal(b2.c_m, (0, 0, -2.5))  # measured relative to its point0!
    np_arrays_equal(r.point0, (0, 0, 8))
    assert abs(r.length - 0.5) < 1e-10, f"Unexpected length {r.length}"
    if show:
        show_it(crane, markCOM=True, markSubCOM=False)

    b1.rotate(angle=90, asDeg=True)  # b1 east (as initially)
    np_arrays_equal(b1.direction, (10, 0, 0))
    np_arrays_equal(b1.c_m, (5, 0, 0))
    np_arrays_equal(b2.direction, (-5, 0, 0))
    np_arrays_equal(b2.c_m, (-2.5, 0, 0))
    if show:
        show_it(crane, markCOM=True, markSubCOM=False)

    p.rotate(angle=-90, asDeg=True)  # turn p so that b1 south
    np_arrays_equal(b1.direction, (0, -10, 0))
    np_arrays_equal(b1.c_m, (0, -5, 0))
    np_arrays_equal(b2.direction, (0, 5, 0))
    np_arrays_equal(b2.c_m, (0, 2.5, 0))
    if show:
        show_it(crane, markCOM=True, markSubCOM=False)


def test_c_m(show: bool = True):
    # Note: Boom.c_m is a local measure, calculated from Boom.point0
    (
        crane,
        p,
        b1,
        b2,
    ) = test_crane(includeRope=False)
    if show:
        show_it(crane, markCOM=True, markSubCOM=False)
    # initial c_m location
    #        print("Initial c_m:", p.c_m, b1.c_m, b2.c_m)
    np_arrays_equal(p.c_m, (-1, 0.8, 1.5))
    np_arrays_equal(b1.c_m, (5, 0, 0))
    np_arrays_equal(b2.c_m, (-2.5, 0, 0))
    #        self.show_it( (p,b1,b2), markCOM=True, markSubCOM=True)
    # all booms along a line in z-direction
    b2.rotate(angle=-180, asDeg=True)
    b1.rotate(angle=-90.0, asDeg=True)
    #        self.show_it( (p,b1,b2), markCOM=True, markSubCOM=True)
    np_arrays_equal(b1.c_m, (0, 0, 5))
    np_arrays_equal(b2.c_m, (0, 0, 2.5))
    # update all subsystem center of mass points. Need to do that from last boom!
    crane.calc_statics_dynamics(dT=None)
    if show:
        show_it(crane, markCOM=True, markSubCOM=False)
    np_arrays_equal(b2.c_m_sub[1], b2.c_m_absolute)
    np_arrays_equal(b1.c_m_sub[1], (0, 0, 3 + 5 + 100 / 300 * (15.5 - 8)))
    assert p.c_m_sub[1][2] == 1.5 + 300 / 2300 * (b1.c_m_sub[1][2] - 1.5)
    b1.rotate(angle=-45, asDeg=True)
    b2.rotate(angle=30, asDeg=True)
    p.rotate(angle=-20, asDeg=True)


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


def test_animation():
    crane, p, b1, b2, r = test_crane(includeRope=True)

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
                    [b.point0[0], (b.point0 + b.direction)[0]],
                    [b.point0[1], (b.point0 + b.direction)[1]],
                    [b.point0[2], (b.point0 + b.direction)[2]],
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
                [b.point0[0], (b.point0 + b.direction)[0]],
                [b.point0[1], (b.point0 + b.direction)[1]],
                [b.point0[2], (b.point0 + b.direction)[2]],
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
    np_arrays_equal(r.point0, (0, -15 / sqrt(2), 3 + 15 / sqrt(2)))


def show_it(crane, markCOM=True, markSubCOM=True):
    # update all subsystem center of mass points. Need to do that from last boom!
    crane.calc_statics_dynamics(dT=None)
    fig = plt.figure(figsize=(9, 9), layout="constrained")
    ax = Axes3D(fig=fig)
    # ax = plt.axes(projection="3d")  # , data=line)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 10)

    lines = []
    c_ms = []
    subCOMs = []
    for i, b in enumerate(crane.booms()):
        lw = 10 if i == 0 else 5
        lines.append(
            ax.plot(
                [b.point0[0], b.point1[0]],
                [b.point0[1], b.point1[1]],
                [b.point0[2], b.point1[2]],
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
                    s=str(int(b.mass.initialVal)),
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
    #    test_rot()
    crane = test_crane()
    test_sequence(show=0)
    test_rotate_instant(show=0)
    test_rotation(show=0)
    test_change_length(show=0)
    test_c_m(show=0)
    test_animation()
