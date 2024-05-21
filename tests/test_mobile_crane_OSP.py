import os

from component_model.model import make_osp_system_structure

# from component_model.component_FMUs import InputTable #Note: needed even if only running the FMU!
from component_model.plotter import VisualSimulator
from libcosimpy.CosimEnums import (
    CosimExecutionState,
    CosimVariableCausality,
    CosimVariableType,
    CosimVariableVariability,
)
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave


def test_make_OSP_system_structure():
    make_osp_system_structure(
        "OspSystemStructure",
        version="0.1",
        baseStepSize=0.1,
        models={
            "simpleTable": {"interpolate": True},
            "mobileCrane": {"pedestal.mass": 5000.0, "boom.boom[0]": 20.0},
        },
        connections=(("simpleTable", "outputs[0]", "mobileCrane", "pedestal.angularVelocity"),),
    )


def test_visual_simulation_1():
    simulator = VisualSimulator()
    simulator.start(
        points_3d=[
            (
                ("mobileCrane", "pedestal.cartesianEnd.0"),
                ("mobileCrane", "pedestal.cartesianEnd.1"),
                ("mobileCrane", "pedestal.cartesianEnd.2"),
            ),
            (
                ("mobileCrane", "boom.cartesianEnd.0"),
                ("mobileCrane", "boom.cartesianEnd.1"),
                ("mobileCrane", "boom.cartesianEnd.2"),
            ),
            (
                ("mobileCrane", "rope.cartesianEnd.0"),
                ("mobileCrane", "rope.cartesianEnd.1"),
                ("mobileCrane", "rope.cartesianEnd.2"),
            ),
        ],
        osp_system_structure="OspSystemStructure.xml",
    )


def test_visual_simulation_2():
    simulator = VisualSimulator()
    simulator.start(
        points_3d=[
            (
                ("mobileCrane", "pedestal.cartesianEnd.0"),
                ("mobileCrane", "pedestal.cartesianEnd.1"),
                ("mobileCrane", "pedestal.cartesianEnd.2"),
            ),
            (
                ("mobileCrane", "rope.cartesianEnd.0"),
                ("mobileCrane", "rope.cartesianEnd.1"),
                ("mobileCrane", "rope.cartesianEnd.2"),
            ),
        ],
        osp_system_structure="OspSystemStructure.xml",
    )


def test_simpletable():
    """Stand-alone test of SimpleTable.fmu using OSP"""
    perform_test("table")


def test_mobilecrane():
    """Stand-alone test of MobileCrane.fmu using OSP"""
    perform_test("crane")


def test_crane_OSP():
    """Test of Crane with simpleTable, based on OspSystemConfig file"""
    perform_test("crane-with-table")


def perform_test(what="crane-with-table"):
    """Test the stand-alone crane using internal events to change variable values
    what='crane': stand-alone simulation of crane
    what='table': stand-alone simulation of table
    what='crane-with-table': simulation of crane with input from table
    """
    assert os.path.exists("./SimpleTable.fmu"), "SimpleTable.fmu not found in this folder. Should be copied beforehand."
    assert os.path.exists(
        "./MobileCrane.fmu"
    ), "MobileCrane.fmu not found in this folder. Should be generated and copied (by ../test_mobile_crane_FMU.py)"

    # log_output_level( CosimLogLevel.WARNING)
    if what == "crane" or what == "table":  # single tests without config file
        simulator = CosimExecution.from_step_size(
            step_size=1e8
        )  # empty execution object with fixed time step in nanos (alternatives are config files)
    if what == "crane":
        crane = CosimLocalSlave(fmu_path="MobileCrane.fmu", instance_name="mobileCrane")
        iCrane = simulator.add_local_slave(crane)
    if what == "table":
        table = CosimLocalSlave(fmu_path="SimpleTable.fmu", instance_name="simpleTable")
        iTable = simulator.add_local_slave(table)
    if what == "crane-with-table":  # system simulation
        sys_path = os.path.join(os.path.abspath(os.path.curdir), "OspSystemStructure.xml")
        assert os.path.exists(sys_path), "OspSystemStructure file not found"
        print("PATH", sys_path)
        simulator = CosimExecution.from_osp_config_file(sys_path)
    sim_status = simulator.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
    infos = simulator.slave_infos()
    #    print("DIR", dir(simulator))
    #    print( simulator.execution_status)
    if what == "table":
        assert len(infos) == 1, "# installed slaves"
        assert infos[0].name.decode() == "simpleTable"
        assert simulator.num_slaves() == 1, "# installed slaves"
        assert simulator.slave_index_from_instance_name("simpleTable") == 0, "Slave index"
        assert simulator.num_slave_variables(0) == 4, "# slave variables"
        assert simulator.real_time_simulation_enabled(), "real time enabled"  # why?
        assert simulator.slave_variables(0)[0].name.decode() == "outs[0]"
        assert simulator.slave_variables(0)[0].reference == 0
        assert simulator.slave_variables(0)[0].type == CosimVariableType.REAL.value
        assert simulator.slave_variables(0)[0].causality == CosimVariableCausality.OUTPUT.value
        assert simulator.slave_variables(0)[0].variability == CosimVariableVariability.CONTINUOUS.value

    #    assert simulator.current_time==0, "Simulator current_time"
    if what != "crane":
        iTable = simulator.slave_index_from_instance_name("simpleTable")
        vars = simulator.slave_variables(iTable)
        for var in vars:
            print(f"Slave variable (table) ({var.reference}): {var.name}")

    if what != "table":
        iCrane = simulator.slave_index_from_instance_name("mobileCrane")
        vars = simulator.slave_variables(iCrane)
        for var in vars:
            print(f"Slave variable (crane) ({var.reference}): {var.name}")

    f_t = CosimObserver.create_time_series()
    simulator.add_observer(observer=f_t)
    if what == "crane" or what == "crane-with-table":  # observe crane output
        assert f_t.start_time_series(slave_index=iCrane, value_reference=9, variable_type=CosimVariableType.REAL)
        assert f_t.start_time_series(slave_index=iCrane, value_reference=10, variable_type=CosimVariableType.REAL)
        assert f_t.start_time_series(slave_index=iCrane, value_reference=11, variable_type=CosimVariableType.REAL)
    manipulator = CosimManipulator.create_override()
    simulator.add_manipulator(manipulator=manipulator)
    if what == "crane" or what == "crane-with-table":  # manipulate crane
        simulator.real_initial_value(slave_index=iCrane, variable_reference=11, value=1.0)
        manipulator.slave_real_values(slave_index=iCrane, variable_references=[11, 12, 13], values=[1.0, 1.0, 0.0])

    simulator.simulate_until(target_time=1e9)  # automatic stepping with stopTime in nanos (alternative: .step())
    if what == "crane" or what == "crane-with-table":  # extract and print crane data
        t, s, torque0 = f_t.time_series_real_samples(iCrane, value_reference=9, from_step=1, sample_count=11)
        t, s, torque1 = f_t.time_series_real_samples(iCrane, value_reference=10, from_step=1, sample_count=11)
        t, s, torque2 = f_t.time_series_real_samples(iCrane, value_reference=11, from_step=1, sample_count=11)
        for i in range(len(t)):
            print(f"{t[i]/1e9}, {torque0[i]}, {torque1[i]}, {torque2[i]}")
    print(f"Simulation {what} finalized")


if __name__ == "__main__":
    #    test_make_OSP_system_structure()
    test_mobilecrane()  # Stand-alone test of MobileCrane.fmu using OSP
#    test_simpletable() # Stand-alone test of SimpleTable.fmu using OSP
#    test_crane_OSP() # Test of Crane with simpleTable, based on OspSystemConfig file
#    test_crane_OSP() # Test of Crane with simpleTable, based on OspSystemConfig file

#     test_visual_simulation_1()
#     test_visual_simulation_2()
