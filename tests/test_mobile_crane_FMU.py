import shutil
from math import radians

from component_model.model import Model
from crane_fmu.boom import Boom
from crane_fmu.crane import Crane  # , Animation
from fmpy import dump, plot_result, simulate_fmu
from fmpy.validation import validate_fmu
import pytest

class MobileCrane(Crane):
    """Simple mobile crane for FMU testing purposes.
    The crane has a short pedestal, one variable-length stiff boom and a rope.
    The size and weight of the various parts can be configured.

    Args:
        name (str) : name of the crane type
        description (str) : short description
        author (str)
        version (str)
        pedestalMass (str) : mass of the pedestal - quantity and unit as string
        pedestalHeight (str) : height (fixed) of the pedestal, with units
        boomMass (str) : mass of the single boom, with units
        boomLength0 (str) : minimum length of the boom, with units
        boomLength1 (str) : maximum length of the boom, with units
    """

    def __init__(
        self,
        name:str ="mobile_crane",
        description:str ="Simple mobile crane (for FMU testing) with short pedestal, one variable-length elevation boom and a rope",
        author:str="DNV, SEACo project",
        version:str="0.2",
        pedestalMass:str="10000.0 kg",
        pedestalHeight:str="3.0 m",
        boomMass:str="1000.0 kg",
        boomLength0:str="8 m",
        boomLength1:str="50 m",
        **kwargs,
    ):
        super().__init__(name=name, description=description, author=author, version=version, **kwargs)
        pedestal = self.add_boom(
            name="pedestal",
            description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
            mass=pedestalMass,
            centerOfMass=(0.5, "-1m", 0),
            boom=(pedestalHeight, "0deg", "0deg"),
            boom_rng=(None, None, ("0deg", "360deg")),
        )
        boom = self.add_boom(
            name="boom",
            description="The boom. Can be lifted and length can change within the given range",
            mass=boomMass,
            centerOfMass=(0.5, 0, 0),
            boom=(boomLength0, "90deg", "0deg"),
            boom_rng=((boomLength0, boomLength1), (0, "90deg"), None),
        )
        _ = self.add_boom(
            name="rope",
            description="The rope fixed to the last boom. Flexible connection",
            mass="50.0 kg",  # so far basically the hook
            centerOfMass=0.95,
            boom=("1e-6 m", "180deg", "0 deg"),
            boom_rng=(
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
        status = super().do_step(currentTime, stepSize)
        #print(f"Time {currentTime}, {self.rope_tip}")
        #print(f"MobileCrane.do_step. Status {status}")
        return status


def test_make_mobilecrane():
    asBuilt = Model.build(
        "test_mobile_crane_FMU.py",
        project_files=[
            "../crane_fmu",
        ],
    )
    val = validate_fmu(asBuilt.name)
    assert not len(val), f"Validation of the modelDescription of {asBuilt.name} was not successful. Errors: {val}"
    dump(asBuilt.name)
    shutil.copy(asBuilt.name, "./OSP_model/")  # copy the created FMU also to the OSP_model folder
    shutil.copy(asBuilt.name, "../../case_study/tests/data/MobileCrane/")  # ... and to case_study (to be deleted)

#@pytest.mark.skip("Run the FMU")
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
            "pedestal_boom[2]": radians(90),
            "boom_mass": 1000.0,
            "boom_boom[0]": 8,
            "boom_boom[1]": radians(45),
            "boom_angularVelocity[0]": 0.02,
            "rope_boom[0]": 1e-6,
            "fixation_angularVelocity[0]": 0.0,
            "fixation_angularVelocity[1]": 0.0,
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
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Return code {retcode}"
