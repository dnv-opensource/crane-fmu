import shutil
from math import radians

from component_model.model import Model
from crane_fmu.boom import Boom
from crane_fmu.crane import Crane  # , Animation
from fmpy import dump, plot_result, simulate_fmu
from fmpy.validation import validate_fmu


class MobileCrane(Crane):
    """Simple mobile crane for FMU testing purposes.
    The crane has a short pedestal, one variable-length stiff boom and a rope.
    The size and weight of the various parts can be configured.

    Args:

    """

    def __init__(
        self,
        name="mobile_crane",
        description="Simple mobile crane (for FMU testing) with short pedestal, one variable-length elevation boom and a rope",
        author="DNV, SEACo project",
        version="0.2",
        pedestalMass="10000.0 kg",
        pedestalHeight="3.0 m",
        boomMass="1000.0 kg",
        boomLength0="8 m",
        boomLength1="50 m",
        **kwargs,
    ):
        super().__init__(
            name=name, description=description, author=author, version=version, **kwargs
        )
        pedestal = Boom(
            name="pedestal",
            description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
            anchor0=self,
            mass=pedestalMass,
            centerOfMass=(0.5, "-1m", 0),
            boom=(pedestalHeight, "0deg", "0deg"),
            boomRng=(None, None, ("0deg", "360deg")),
        )
        boom = Boom(
            name="boom",
            description="The boom. Can be lifted and length can change within the given range",
            anchor0=pedestal,
            mass=boomMass,
            centerOfMass=(0.5, 0, 0),
            boom=(boomLength0, "90deg", "0deg"),
            boomRng=((boomLength0, boomLength1), (0, "90deg"), None),
        )
        _ = Boom(
            name="rope",
            description="The rope fixed to the last boom. Flexible connection",
            anchor0=boom,
            mass="50.0 kg",  # so far basically the hook
            centerOfMass=0.95,
            boom=("1e-6 m", "180deg", "0 deg"),
            boomRng=(
                ("1e-6 m", boomLength1),
                ("90deg", "270deg"),
                ("-180deg", "180deg"),
            ),
            dampingQ=50.0,
            animationLW=2,
        )
        # make sure that _comSub is calculated for all booms:
        self.calc_statics_dynamics(None)

    def do_step(self, currentTime, stepSize):
        super().do_step(currentTime, stepSize)
        print(f"Time {currentTime}, {self.rope_tip}")
        return True


def test_make_mobilecrane():
    asBuilt = Model.build(
        "test_mobile_crane_FMU.py",
        project_files=[
            "../crane_fmu",
        ],
    )
    val = validate_fmu(asBuilt.name)
    assert not len(
        val
    ), f"Validation of the modelDescription of {asBuilt.name} was not successful. Errors: {val}"
    dump(asBuilt.name)
    shutil.copy(
        asBuilt.name, "./OSP_model/"
    )  # copy the created FMU also to the OSP_model folder


def test_run_mobilecrane():
    result = simulate_fmu(
        "OSP_model/MobileCrane.fmu",
        stop_time=1.0,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={
            "pedestal_mass": 10000.0,
            "pedestal_boom[0]": 3.0,
            "boom_mass": 1000.0,
            "boom_boom[0]": 8,
            "boom_boom[1]": radians(50),
            "craneAngularVelocity[0]": 0.1,
            "craneAngularVelocity[1]": 0.0,
            "craneAngularVelocity[2]": 0.0,
            "craneAngularVelocity[3]": 1.0,
        },
    )
    plot_result(result)


# def test_dll():
#     bb = WinDLL(os.path.abspath(os.path.curdir) + "\\BouncingBall.dll")
#     bb.fmi2GetTypesPlatform.restype = c_char_p
#     print(bb.fmi2GetTypesPlatform(None))
#     bb.fmi2GetVersion.restype = c_char_p
#     print(bb.fmi2GetVersion(None))


if __name__ == "__main__":
    test_make_mobilecrane()
    test_run_mobilecrane()
#    test_dll()
