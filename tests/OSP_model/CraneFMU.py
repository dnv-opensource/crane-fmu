from crane_fmu.crane import Crane
from crane_fmu.boom import Boom


class MobileCrane(Crane):
    """Simple mobile crane for FMU testing purposes.
    The crane has a short pedestal, one variable-length stiff boom and a rope.
    The size and weight of the various parts can be configured.

    Args:

    """

    def __init__(
        self,
        name: str = "mobile_crane",
        description: str = "Simple mobile crane (for FMU testing) with short pedestal, one variable-length elevation boom and a rope",
        author: str = "DNV, SEACo project",
        version: str = "0.2",
        pedestalMass: str = "10000.0 kg",
        pedestalHeight: str = "3.0 m",
        boomMass: str = "1000.0 kg",
        boomLength0: str = "8 m",
        boomLength1: str = "50 m",
        **kwargs: dict[str, str | float],
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
