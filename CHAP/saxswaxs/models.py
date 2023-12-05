# System modules
from functools import cache
from typing import Literal, Optional

# Third party modules
import numpy as np
from pydantic import (BaseModel,
                      validator,
                      constr,
                      conint,
                      confloat)
from pyFAI.units import AZIMUTHAL_UNITS, RADIAL_UNITS


class IntegrationConfig(BaseModel):
    """Class representing a complete set of parameters for performaing
    pyFAI integration.

    :ivar title: title of the integration
    :type title: str
    :ivar integration_type: type of integration, one of "azimuthal",
        "radial", or "cake"
    :type integration_type: str
    :ivar radial_units: radial units for the integration, defaults to
        `'q_A^-1'`
    :type radial_units: str, optional
    :ivar radial_min: minimum radial value for the integration range
    :type radial_min: float, optional
    :ivar radial_max: maximum radial value for the integration range
    :type radial_max: float, optional
    :ivar radial_npt: number of points in the radial range for the
        integration
    :type radial_npt: int, optional
    :ivar azimuthal_units: azimuthal units for the integration
    :type azimuthal_units: str, optional
    :ivar azimuthal_min: minimum azimuthal value for the integration
        range
    :type azimuthal_min: float, optional
    :ivar azimuthal_max: maximum azimuthal value for the integration
        range
    :type azimuthal_max: float, optional
    :ivar azimuthal_npt: number of points in the azimuthal range for
        the integration
    :type azimuthal_npt: int, optional
    :ivar include_errors: option to include pyFAI's calculated Poisson errors
        with the integration results, defaults to `False`
    :type include_errors: bool, optional
    :ivar right_handed: for radial and cake integration, reverse the
        direction of the azimuthal coordinate from pyFAI's convention,
        defaults to True
    :type right_handed: bool, optional

    """
    title: constr(strip_whitespace=True, min_length=1)
    integration_type: Literal['azimuthal', 'radial', 'cake']
    radial_units: Literal[*RADIAL_UNITS.keys()] = 'q_A^-1'
    radial_min: confloat(ge=0)
    radial_max: confloat(gt=0)
    radial_npt: conint(gt=0) = 1800
    azimuthal_units: Literal[*AZIMUTHAL_UNITS.keys()] = 'chi_deg'
    azimuthal_min: confloat(ge=-180) = -180
    azimuthal_max: confloat(le=360) = 180
    azimuthal_npt: conint(gt=0) = 3600
    error_model: Optional[Literal['poisson', 'azimuthal']] = None
    right_handed: bool = True

    def validate_range_max(range_name:str):
        """Validate the maximum value of an integration range.

        :param range_name: The name of the integration range
            (e.g. radial, azimuthal).
        :type range_name: str
        :return: The callable that performs the validation.
        :rtype: callable
        """
        def _validate_range_max(cls, range_max, values):
            """Check if the maximum value of the integration range is
            greater than its minimum value.

            :param range_max: The maximum value of the integration
                range.
            :type range_max: float
            :param values: The values of the other fields being
                validated.
            :type values: dict
            :raises ValueError: If the maximum value of the
                integration range is not greater than its minimum
                value.
            :return: The validated maximum range value
            :rtype: float
            """
            range_min = values.get(f'{range_name}_min')
            if range_min < range_max:
                return range_max
            raise ValueError(
                'Maximum value of integration range must be '
                'greater than minimum value of integration range '
                f'({range_name}_min={range_min}).')
        return _validate_range_max

    _validate_radial_max = validator(
        'radial_max',
        allow_reuse=True)(validate_range_max('radial'))
    _validate_azimuthal_max = validator(
        'azimuthal_max',
        allow_reuse=True)(validate_range_max('azimuthal'))

    @validator('error_model')
    def validate_error_model(cls, error_model, values):
        integration_type = values.get('integration_type')
        if error_model is not None and integration_type == 'radial':
            print(
                'warning: errors will not be included for radial integration')
            return None
        return error_model

    def get_azimuthal_adjustment(self):
        """To enable a continuous range of integration from pyFAI in
        the azimuthal direction for radial and cake integration,
        obtain an offset by which the detectors will be artificially
        rotated about the beam. The azimuthal coordinates for
        integrated data will be shifted back by the same amount.

        :return: Offset for the azimuthal axis in degrees
        :rtype: float
        """
        # Fix chi discontinuity at 180 degrees for now (default for
        # pyFAI)
        chi_disc = 180
        # Force a right-handed coordinate system if requested
        if self.right_handed:
            chi_min = 360 - self.azimuthal_max
            chi_max = 360 - self.azimuthal_min
        else:
            chi_min = self.azimuthal_min
            chi_max = self.azimuthal_max
        # If the discontinuity is crossed, artificially rotate the
        # detectors to achieve a continuous azimuthal integration
        # range 
        if chi_min < chi_disc and chi_max > chi_disc:
            chi_offset = chi_max - chi_disc
        else:
            chi_offset = 0
        return chi_offset

    @property
    def integrated_data_coordinates(self):
        """Return a dictionary of coordinate arrays for navigating the
        dimension(s) of the integrated data produced by this instance
        of `IntegrationConfig`.

        :return: A dictionary with either one or two keys: 'azimuthal'
            and/or 'radial', each of which points to a 1-D `numpy`
            array of coordinate values.
        :rtype: dict[str,np.ndarray]
        """
        coords = {}
        if self.integration_type in ('radial', 'cake'):
            coords[self.azimuthal_units] = np.linspace(
                self.azimuthal_min, self.azimuthal_max, self.azimuthal_npt)
        if self.integration_type in ('azimuthal', 'cake'):
            coords[self.radial_units] = np.linspace(
                self.radial_min, self.radial_max, self.radial_npt)
        return coords

    @property
    def integrated_data_dims(self):
        """Return a tuple of the coordinate labels for the integrated
        data produced by this instance of `IntegrationConfig`.
        """
        directions = list(self.integrated_data_coordinates.keys())
        dim_names = [getattr(self, f'{direction}_units')
                     for direction in directions]
        return dim_names

    @property
    def integrated_data_shape(self):
        """Return a tuple representing the shape of the integrated
        data produced by this instance of `IntegrationConfig` for a
        single scan step.
        """
        return tuple([len(coordinate_values)
                      for coordinate_name, coordinate_values
                      in self.integrated_data_coordinates.items()])
