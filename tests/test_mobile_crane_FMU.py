from math import cos, radians, sin, sqrt

import numpy as np
from fmpy import dump, plot_result, simulate_fmu
from fmpy.validation import validate_fmu
import pytest

np.set_printoptions(formatter={"float_kind": "{:.2f}".format})


def arrays_equal(arr1, arr2, eps=1e-7):
    assert len(arr1) == len(arr2), "Length not equal!"

    for i in range(len(arr1)):
        assert abs(arr1[i] - arr2[i]) < eps, f"Component {i}: {arr1[i]} != {arr2[i]}"


def mass_center(xs: tuple):
    """Calculate the total mass center of a number of point masses provided as 4-tuple"""
    M, c = 0.0, np.array((0, 0, 0), float)
    for x in xs:
        M += x[0]
        c += x[0] * np.array(x[1:], float)
    return (M, c / M)


def test_mass_center():
    def do_test(Mc, _M, _c):
        assert Mc[0] == _M, f"Mass not as expected: {Mc[0]} != {_M}"
        arrays_equal(Mc[1], _c, 1e-10)

    do_test(mass_center(((1, -1, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0))), 4, (0, 0, 0))
    do_test(mass_center(((1, 1, 1, 0), (1, 1, -1, 0), (1, -1, -1, 0), (1, -1, 1, 0))), 4, (0, 0, 0))


# @pytest.mark.skip("Do not make a new FMU just now")
def test_mobilecrane_fmu(mobile_crane_fmu, show: bool):
    """The mobileCrane is build within the fixture 'mobile_crane_fmu'.
    Validate the FMU here and dump its interface.
    """
    val = validate_fmu(str(mobile_crane_fmu))
    assert not len(
        val
    ), f"Validation of the modelDescription of {mobile_crane_fmu.name} was not successful. Errors: {val}"
    if show:
        dump(mobile_crane_fmu)


# @pytest.mark.skip("Run the FMU")
def test_run_mobilecrane(mobile_crane_fmu, show: bool):
    result = simulate_fmu(  # static run
        str(mobile_crane_fmu),
        stop_time=0.1,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={
            "pedestal_mass": 10000.0,
            "pedestal_boom[0]": 3.0,
            "pedestal_boom[2]": 0.0,
            "boom_mass": 1000.0,
            "boom_boom[0]": 8,
            "boom_boom[1]": 45.0,  # input as deg, internal: rad
            "rope_boom[0]": 1e-6,
        },
    )
    # result is a list of tuples. Each tuple contains (time, output-variables)
    assert abs(result[0][19] - 8) < 1e-9, f"Default start value {result[0][19]}. Default start value of boom end!"
    assert result[1][0] == 0.01, "fmpy does not seem to deal properly with the step_size argument!"
    assert abs(result[1][19] - 8 / sqrt(2)) < 1e-14, f"Initial setting {result[1][19]} visible only after first step!"
    M, c = mass_center(
        ((10000, -1, 0, 1.5), (1000, 4 / sqrt(2), 0, 3 + 4 / sqrt(2)), (50, 8 / sqrt(2), 0, 3 + 8 / sqrt(2)))
    )

    result = simulate_fmu(
        str(mobile_crane_fmu),
        stop_time=0.1,
        step_size=0.1,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={
            "pedestal_mass": 10000.0,
            "pedestal_boom[0]": 3.0,
            "pedestal_boom[2]": 0.0,
            "boom_mass": 1000.0,
            "boom_boom[0]": 8,
            "boom_boom[1]": 45.0,  # input as deg, internal: rad
            "pedestal_angularVelocity[1]": radians(1.0),  # azimuthal movement 1 deg per time step
            "rope_boom[0]": 1e-6,
            "fixation_angularVelocity[0]": 0.0,
            "fixation_angularVelocity[1]": 0.0,
        },
    )
    assert (
        abs(result[1][19] - 8 / sqrt(2) * cos(radians(1.0))) < 1e-9
    ), f"Initial setting {result[1][19]} visible only after first step!"
    assert abs(result[10][19] - 8 / sqrt(2) * cos(radians(10))) < 1e-9, f"Final position of boom {result[10][19]}"
    assert abs(result[10][20] - 8 / sqrt(2) * sin(radians(10))) < 1e-9, f"Final position of boom {result[10][20]}"
    assert abs(result[10][21] - 3 - 8 / sqrt(2)) < 1e-9, f"Final position of boom {result[10][21]}"
    if show:
        plot_result(result)
        
if __name__ == "__main__":
   retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "True",  __file__])
   assert retcode == 0, f"Non-zero return code {retcode}"

