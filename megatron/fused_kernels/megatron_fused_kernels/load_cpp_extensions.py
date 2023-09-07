import pathlib
from pybind11.setup_helpers import Pybind11Extension

srcpath = pathlib.Path(__file__).parent.absolute()


def get_helpers_extension():
    """Get the helpers pybind11 extension."""
    return [
        Pybind11Extension(
            name="helpers",
            sources=[str(srcpath / "helpers.cpp")],
            language="c++",
        )
    ]
