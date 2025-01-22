from CA import CA, Cell
import numpy as np


def test_simulation():
    system = CA(20, 40, 5)
    output = system.run_output_last_state(100)
    test_output = ""
    with open("test_systems/output.txt", "r") as file:
        test_output = file.read()
    print(test_output)
    assert output == test_output
