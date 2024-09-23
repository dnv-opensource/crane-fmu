from component_model.plotter import VisualSimulator

if __name__ == "__main__":
    simulator = VisualSimulator()
    simulator.start(
        points_3d=[
            (
                ("mobileCrane", "pedestal_end[0]"),
                ("mobileCrane", "pedestal_end[1]"),
                ("mobileCrane", "pedestal_end[2]"),
            ),
            (
                ("mobileCrane", "boom_end[0]"),
                ("mobileCrane", "boom_end[1]"),
                ("mobileCrane", "boom_end[2]"),
            ),
            (
                ("mobileCrane", "rope_end[0]"),
                ("mobileCrane", "rope_end[1]"),
                ("mobileCrane", "rope_end[2]"),
            ),
        ],
        osp_system_structure="../tests/resources/OspSystemStructure.xml",
    )
