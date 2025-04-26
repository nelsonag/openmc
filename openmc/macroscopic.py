from warnings import warn

from openmc.checkvalue import check_type


class Macroscopic(str):
    """A Macroscopic object that can be used in a material.

    .. deprecated:: 0.15.3
            Use the openmc.Nuclide capability instead.

    Parameters
    ----------
    name : str
        Name of the macroscopic data, e.g. UO2

    Attributes
    ----------
    name : str
        Name of the nuclide, e.g. UO2

    """

    def __new__(cls, name):
        check_type('name', name, str)
        warnings.warn(
            "This class is deprecated, use 'Nuclide' instead", FutureWarning)
        return super().__new__(cls, name)

    @property
    def name(self):
        return self
