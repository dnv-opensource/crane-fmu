Introduction
============
The package extends the `component_model` package to provide the means to construct cranes.
The package can be used to define concrete cranes and simulate their usage, see `tests/resources/mobile-crane.py` as example.

Getting Started
---------------
A concrete crane model should consist of an extension of the Crane object, 

* defining general model variables like `name`, `author`, `version`, etc.
* defining the anticipated set of booms
* if there is a wish to keep parameters configurable, these can be kept as parameters of the extended model.
* In a separate script, instantiate the concrete crane, test it and run `Model.build(...)` to make the FMU.

See the file `tests/resources/mobile_crane.py` and related test files 
as an example on how a concrete crane can be defined and used.


1.	Install the `crane_fmu` package: ``pip install crane_fmu``
2.	Software dependencies: `PythonFMU`, `component_model`, `numpy`, `pint`, `uuid`, `ElementTree`
3.	Latest releases: Version 0.1, based on component_model version 0.1

Usage example
-------------
This is a simple mobile crane, like they are used on e.g. building sites with 

* a short pedestal, which can be turn around, 
* one boom with variable length and which can be lifted
* one rope, where loads can be attached.

.. code-block:: Python

    from crane_fmu.crane import Crane

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
            name: str = "mobile_crane",
            description: str = "Simple mobile crane (for FMU testing) with short pedestal, one variable-length elevation boom and a rope",
            author: str = "DNV, SEACo project",
            version: str = "0.2",
            pedestalMass: str = "10000.0 kg",
            pedestalHeight: str = "3.0 m",
            boomMass: str = "1000.0 kg",
            boomLength0: str = "8 m",
            boomLength1: str = "50 m",
            rope_mass_range: tuple = ("50kg", "2000 kg"),
            **kwargs,
        ):
            super().__init__(name=name, description=description, author=author, version=version, **kwargs)
            _ = self.add_boom(
                name="pedestal",
                description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
                mass=pedestalMass,
                massCenter=(0.5, -1.0, 0.8),
                boom=(pedestalHeight, "0deg", "0deg"),
                boom_rng=(None, None, ("0deg", "360deg")),
            )
            _ = self.add_boom(
                name="boom",
                description="The boom. Can be lifted and length can change within the given range",
                mass=boomMass,
                massCenter=(0.5, 0, 0),
                boom=(boomLength0, "90deg", "0deg"),
                boom_rng=((boomLength0, boomLength1), (0, "90deg"), None),
            )
            _ = self.add_boom(
                name="rope",
                description="The rope fixed to the last boom. Flexible connection",
                mass="50.0 kg",  # so far basically the hook
                massCenter=0.95,
                mass_rng=rope_mass_range,
                boom=("1e-6 m", "180deg", "0 deg"),
                boom_rng=(
                    ("1e-6 m", boomLength1),
                    ("90deg", "270deg"),
                    ("-180deg", "180deg"),
                ),
                damping=50.0,
                animationLW=2,
            )
            # make sure that _comSub is calculated for all booms:
            self.calc_statics_dynamics(None)

        def do_step(self, currentTime, stepSize):
            status = super().do_step(currentTime, stepSize)
            # print(f"Time {currentTime}, {self.rope_tip}")
            # print(f"MobileCrane.do_step. Status {status}")
            return status



Testing and usage of the `MobileCrane.fmu` is demonstrated as part of related `test_***` files.


Contribute
----------
Anybody in the FMU, OSPand SEACo community is especially welcome to contribute to this code, to make it better, 
and especially including other features from model assurance and from SEACo issues.