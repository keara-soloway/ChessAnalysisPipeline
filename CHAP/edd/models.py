# third party modules
import numpy as np
from pathlib import PosixPath
from pydantic import (BaseModel,
                      confloat,
                      conint,
                      conlist,
                      constr,
                      FilePath,
                      root_validator,
                      validator)
from scipy.interpolate import interp1d
from typing import Literal, Optional

# local modules
from CHAP.utils.material import Material
from CHAP.utils.parfile import ParFile
from CHAP.utils.scanparsers import SMBMCAScanParser as ScanParser


class MCAElementConfig(BaseModel):
    """Class representing metadata required to configure a single MCA
    detector element.

    :ivar detector_name: name of the MCA used with the scan
    :type detector_name: str
    :ivar num_bins: number of channels on the MCA
    :type num_bins: int
    :ivar include_bin_ranges: list of MCA channel index ranges whose
        data should be included after applying a mask
    :type include_bin_ranges: list[list[int]]
    """
    detector_name: constr(strip_whitespace=True, min_length=1) = 'mca1'
    num_bins: Optional[conint(gt=0)]
    include_bin_ranges: Optional[
        conlist(
            min_items=1,
            item_type=conlist(
                item_type=conint(ge=0),
                min_items=2,
                max_items=2))] = None

    @validator('include_bin_ranges', each_item=True)
    def validate_include_bin_range(cls, value, values):
        """Ensure no bin ranges are outside the boundary of the detector"""
        num_bins = values.get('num_bins')
        if num_bins is not None:
            value[1] = min(value[1], num_bins)
        return value

    def mca_mask(self):
        """Get a boolean mask array to use on this MCA element's data.

        :return: boolean mask array
        :rtype: numpy.ndarray
        """
        mask = np.asarray([False] * self.num_bins)
        bin_indices = np.arange(self.num_bins)
        for min_, max_ in self.include_bin_ranges:
            _mask = np.logical_and(bin_indices > min_, bin_indices < max_)
            mask = np.logical_or(mask, _mask)
        return mask

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        d['include_bin_ranges'] = [
            list(d['include_bin_ranges'][i]) \
            for i in range(len(d['include_bin_ranges']))]
        return d


class MCAScanDataConfig(BaseModel):
    """Class representing metadata required to locate raw MCA data for
    a single scan and construct a mask for it.

    :ivar spec_file: Path to the SPEC file containing the scan
    :ivar scan_number: Number of the scan in `spec_file`
    :ivar detectors: list of detector element metadata configurations

    :ivar detector_name: name of the MCA used with the scan
    :ivar num_bins: number of channels on the MCA

    :ivar include_bin_ranges: list of MCA channel index ranges whose
        data should be included after applying a mask
    """
    spec_file: Optional[FilePath]
    scan_number: Optional[conint(gt=0)]
    par_file: Optional[FilePath]
    scan_column: Optional[Union[conint(ge=0), str]]

    detectors: conlist(min_items=1, item_type=MCAElementConfig)

    _parfile: Optional[ParFile]
    _scanparser: Optional[ScanParser]

    class Config:
        underscore_attrs_are_private = False

    @root_validator
    def validate_root(cls, values):
        """Validate the `values` dictionary. Fill in a value for
        `_scanparser` and `num_bins` (if the latter was not already
        provided)

        :param values: dictionary of field values to validate
        :type values: dict
        :return: the validated form of `values`
        :rtype: dict
        """
        spec_file = values.get('spec_file')
        par_file = values.get('par_file')
        if spec_file and par_file:
            raise ValueError('Use either spec_file or par_file, not both')
        elif spec_file:
            values['_scanparser'] = ScanParser(values.get('spec_file'),
                                               values.get('scan_number'))
            values['_parfile'] = None
        elif par_file:
            if 'scan_column' not in values:
                raise ValueError(
                    'When par_file is used, scan_column must be used, too')
            values['_parfile'] = ParFile(values.get('par_file'))
            if isinstance(values['scan_column'], str):
                if values['scan_column'] not in values['_parfile'].column_names:
                    raise ValueError(
                        f'No column named {values["scan_column"]} in '
                        + '{values["par_file"]}. Options: '
                        + ', '.join(values['_parfile'].column_names))
            #values['spec_file'] = values['_parfile'].spec_file
            values['_scanparser'] = ScanParser(
                values['_parfile'].spec_file,
                values['_parfile'].good_scan_numbers()[0])
        else:
            raise ValueError('Must use either spec_file or par_file')

        for detector in values.get('detectors'):
            if detector.num_bins is None:
                try:
                    detector.num_bins = values['_scanparser']\
                        .get_detector_num_bins(detector.detector_name)
                except Exception as exc:
                    raise ValueError('No value found for num_bins') from exc
        return values

    @property
    def scanparser(self):
        try:
            scanparser = self._scanparser
        except:
            scanparser = ScanParser(self.spec_file, self.scan_number)
            self._scanparser = scanparser
        return scanparser

    def mca_data(self, detector_config, scan_step_index=None):
        """Get the array of MCA data collected by the scan.

        :param detector_config: detector for which data will be returned
        :type detector_config: MCAElementConfig
        :return: MCA data
        :rtype: np.ndarray
        """
        detector_name = detector_config.detector_name
        if self._parfile is not None:
            if scan_step_index is None:
                import numpy as np
                data = np.asarray(
                    [ScanParser(self._parfile.spec_file, scan_number)\
                     .get_all_detector_data(detector_name)[0] \
                     for scan_number in self._parfile.good_scan_numbers()])
            else:
                data = ScanParser(
                    self._parfile.spec_file,
                    self._parfile.good_scan_numbers()[scan_step_index])\
                    .get_all_detector_data(detector_name)
        else:
            if scan_step_index is None:
                data = self.scanparser.get_all_detector_data(
                    detector_name)
            else:
                data = self.scanparser.get_detector_data(
                    detector_config.detector_name, self.scan_step_index)
        return data

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k,v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        if d.get('_parfile') is None:
            del d['par_file']
            del d['scan_column']
        else:
            del d['spec_file']
            del d['scan_number']                
        for k in ('_scanparser', '_parfile'):
            if k in d:
                del d[k]
        return d


class MaterialConfig(BaseModel):
    """Model for parameters to characterize a sample material

    :ivar hexrd_h5_material_file: path to a HEXRD materials.h5 file containing
        an entry for the material properties.
    :ivar hexrd_h5_material_name: Name of the material entry in
        `hexrd_h5_material_file`.
    :ivar lattice_parameter_angstrom: lattice spacing in angstrom to use for
        a cubic crystal.
    """
    material_file: Optional[FilePath]
    material_name: Optional[constr(strip_whitespace=True, min_length=1)]
    lattice_parameters_angstroms: Optional[confloat(gt=0)]
    sgnum: Optional[int]
    atoms: Optional[list[str]]
    pos: Optional[list]

    _material: Optional[Material]

    class Config:
        underscore_attrs_are_private = False

    @root_validator
    def validate_material(cls, values):
        from CHAP.utils.material import Material
        values['_material'] = Material(**values)
        return values

    def unique_ds(self, tth_tol=0.15, tth_max=90.0):
        """Get a list of unique HKLs and their lattice spacings

        :param tth_tol: minimum resolvable difference in 2&theta
            between two unique HKL peaks, defaults to `0.15`.
        :type tth_tol: float, optional
        :param tth_max: detector rotation about hutch x axis, defaults
            to `90.0`.
        :type tth_max: float, optional
        :return: unique HKLs and their lattice spacings in angstroms
        :rtype: np.ndarray, np.ndarray
        """
        return self._material.get_ds_unique(tth_tol=tth_tol,
                                            tth_max=tth_max)

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k,v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        if '_material' in d:
            del d['_material']
        return d


class MCAElementCalibrationConfig(MCAElementConfig):
    """Class representing metadata & parameters required for
    calibrating a single MCA detector element.

    :ivar max_energy_kev: maximum channel energy of the MCA in keV
    :ivar tth_max: detector rotation about hutch x axis, defaults to `90`.
    :ivar hkl_tth_tol: minimum resolvable difference in 2&theta between two
        unique HKL peaks, defaults to `0.15`.
    :ivar fit_hkls: list of unique HKL indices to fit peaks for in the
        calibration routine
    :ivar tth_initial_guess: initial guess for 2&theta
    :ivar slope_initial_guess: initial guess for detector channel energy
        correction linear slope, defaults to `1.0`.
    :ivar intercept_initial_guess: initial guess for detector channel energy
        correction y-intercept, defaults to `0.0`.
    :ivar tth_calibrated: calibrated value for 2&theta, defaults to None
    :ivar slope_calibrated: calibrated value for detector channel energy
        correction linear slope, defaults to `None`
    :ivar intercept_calibrated: calibrated value for detector channel energy
        correction y-intercept, defaluts to None
    """
    max_energy_kev: confloat(gt=0)
    tth_max: confloat(gt=0, allow_inf_nan=False) = 90.0
    hkl_tth_tol: confloat(gt=0, allow_inf_nan=False) = 0.15
    fit_hkls: Optional[conlist(item_type=conint(ge=0), min_items=1)] = None
    tth_initial_guess: confloat(gt=0, le=tth_max, allow_inf_nan=False)
    slope_initial_guess: float = 1.0
    intercept_initial_guess: float = 0.0
    tth_calibrated: Optional[confloat(gt=0, allow_inf_nan=False)]
    slope_calibrated: Optional[confloat(allow_inf_nan=False)]
    intercept_calibrated: Optional[confloat(allow_inf_nan=False)]

    def fit_ds(self, material):
        """Get a list of HKLs and their lattice spacings that will be
        fit in the calibration routine

        :return: HKLs to fit and their lattice spacings in angstroms
        :rtype: np.ndarray, np.ndarray
        """

        unique_hkls, unique_ds = material.unique_ds(
            tth_tol=self.hkl_tth_tol, tth_max=self.tth_max)

        fit_hkls = np.array([unique_hkls[i] for i in self.fit_hkls])
        fit_ds = np.array([unique_ds[i] for i in self.fit_hkls])

        return fit_hkls, fit_ds


class MCAElementDiffractionVolumeLengthConfig(MCAElementConfig):
    """Class representing input parameters required to perform a
    diffraction volume length measurement for a single MCA detector
    element.

    :ivar measurement_mode: placeholder for recording whether the
        measured DVL value was obtained through the automated
        calculation or a manual selection.
    :type measurement_mode: Literal['manual', 'auto']
    :ivar sigma_to_dvl_factor: to measure the DVL, a gaussian is fit
        to a reduced from of the raster scan MCA data. This variable
        is a scalar that converts the standard deviation of the
        gaussian fit to the measured DVL.
    :type sigma_to_dvl_factor: Optional[Literal[1.75, 1., 2.]]
    :ivar dvl_measured: placeholder for the measured diffraction
        volume length before writing data to file.
    """
    measurement_mode: Optional[Literal['manual', 'auto']] = 'auto'
    sigma_to_dvl_factor: Optional[Literal[1.75, 1., 2.]] = 1.75
    dvl_measured: Optional[confloat(gt=0)] = None

    def dict(self, *args, **kwargs):
        """If measurement_mode is 'manual', exclude
        sigma_to_dvl_factor from the dict representation.
        """
        d = super().dict(*args, **kwargs)
        if self.measurement_mode == 'manual':
            del d['sigma_to_dvl_factor']
        return d


class DiffractionVolumeLengthConfig(MCAScanDataConfig):
    """Class representing metadata required to perform a diffraction
    volume length calculation for an EDD setup using a steel-foil
    raster scan.

    :ivar detectors: list of individual detector elmeent DVL
        measurement configurations
    :type detectors: list[MCAElementDiffractionVolumeLengthConfig]
    """
    detectors: conlist(min_items=1,
                       item_type=MCAElementDiffractionVolumeLengthConfig)

    @property
    def scanned_vals(self):
        """Return the list of values visited by the scanning motor
        over the course of the raster scan.

        :return: list of scanned motor values
        :rtype: np.ndarray
        """
        if self._parfile is not None:
            return self._parfile.get_values(
                self.scan_column,
                scan_numbers=self._parfile.good_scan_numbers())
        return self.scanparser.spec_scan_motor_vals[0]

    @property
    def scanned_dim_lbl(self):
        """Return a label for plot axes corresponding to the scanned
        dimension

        :rtype: str
        """
        if self._parfile is not None:
            return self.scan_column
        return self.scanparser.spec_scan_motor_mnes[0]

class CeriaConfig(MaterialConfig):
    """Model for a Material representing CeO2 used in calibrations.

    :ivar hexrd_h5_material_name: Name of the material entry in
        `hexrd_h5_material_file`, defaults to `'CeO2'`.
    :ivar lattice_parameter_angstrom: lattice spacing in angstrom to use for
        the cubic CeO2 crystal, defaults to `5.41153`.
    """
    material_name: constr(strip_whitespace=True, min_length=1) = 'CeO2'
    lattice_parameters_angstroms: confloat(gt=0) = 5.41153


class MCACeriaCalibrationConfig(MCAScanDataConfig):
    """
    Class representing metadata required to perform a Ceria calibration for an
    MCA detector.

    :ivar scan_step_index: Index of the scan step to use for calibration,
        optional. If not specified, the calibration routine will be performed
        on the average of all MCA spectra for the scan.

    :ivar flux_file: csv file containing station beam energy in eV (column 0)
        and flux (column 1)

    :ivar material: material configuration for Ceria
    :type material: CeriaConfig

    :ivar detectors: list of individual detector element calibration
        configurations
    :type detectors: list[MCAElementCalibrationConfig]

    :ivar max_iter: maximum number of iterations of the calibration routine,
        defaults to `10`.
    :ivar tune_tth_tol: stop iteratively tuning 2&theta when an iteration
        produces a change in the tuned value of 2&theta that is smaller than
        this value, defaults to `1e-8`.
    """
    scan_step_index: Optional[conint(ge=0)]

    flux_file: FilePath

    material: CeriaConfig

    detectors: conlist(min_items=1, item_type=MCAElementCalibrationConfig)

    max_iter: conint(gt=0) = 10
    tune_tth_tol: confloat(ge=0) = 1e-8

    def mca_data(self, detector_config):
        """Get the 1D array of MCA data to use for calibration.

        :param detector_config: detector for which data will be returned
        :type detector_config: MCAElementConfig
        :return: MCA data
        :rtype: np.ndarray
        """
        if self.scan_step_index is None:
            data = super().mca_data(detector_config)
            if self.scanparser.spec_scan_npts > 1:
                data = np.average(data, axis=1)
            else:
                data = data[0]
        else:
            data = super().mca_data(detector_config,
                                    scan_step_index=self.scan_step_index)
        return data

    def flux_correction_interpolation_function(self):
        """
        Get an interpolation function to correct MCA data for relative energy
        flux of the incident beam.

        :return: energy flux correction interpolation function
        :rtype: scipy.interpolate._polyint._Interpolator1D
        """

        flux = np.loadtxt(self.flux_file)
        energies = flux[:,0]/1.e3
        relative_intensities = flux[:,1]/np.max(flux[:,1])
        interpolation_function = interp1d(energies, relative_intensities)
        return interpolation_function


def select_hkls(detector, material, tth, y, x, interactive):
    """Return a plot of `detector.fit_hkls` as a matplotlib
    figure. Optionally modify `detector.fit_hkls` by interacting with
    a matplotlib figure.

    :param detector: the detector to set `fit_hkls` on
    :type detector: MCAElementConfig
    :param material: the material to pick HKLs for
    :type material: MaterialConfig
    :param tth: diffraction angle two-theta
    :type tth: float
    :param y: reference y data to plot
    :type y: np.ndarray
    :param x: reference x data to plot
    :type x: np.ndarray
    :param interactive: show the plot and allow user interactions with
        the matplotlib figure
    :type interactive: bool
    :return: plot showing the user-selected HKLs
    :rtype: matplotlib.figure.Figure
    """
    import numpy as np
    from scipy.constants import physical_constants
    hkls, ds = material.unique_ds(
        tth_tol=detector.hkl_tth_tol, tth_max=detector.tth_max)
    peak_locations = 1e7 * physical_constants['Planck constant in eV/Hz'][0] \
                     * physical_constants['speed of light in vacuum'][0] \
                     / (2. * ds * np.sin(0.5 * np.radians(tth)))
    pre_selected_peak_indices = detector.fit_hkls \
                                if detector.fit_hkls else []
    from CHAP.utils.general import select_peaks
    selected_peaks, figure = select_peaks(
        y, x, peak_locations,
        peak_labels=[str(hkl)[1:-1] for hkl in hkls],
        pre_selected_peak_indices=pre_selected_peak_indices,
        mask=detector.mca_mask(),
        interactive=interactive,
        xlabel='MCA channel energy (keV)',
        ylabel='MCA intensity (counts)',
        title='Mask and HKLs for fitting')

    selected_hkl_indices = [int(np.where(peak_locations == peak)[0][0]) \
                            for peak in selected_peaks]
    detector.fit_hkls = selected_hkl_indices

    return figure


def select_tth_initial_guess(detector, material, y, x):
    """Show a matplotlib figure of a reference MCA spectrum on top of
    HKL locations. The figure includes an input field to adjust the
    initial tth guess and responds by updating the HKL locations based
    on the adjusted value of the initial tth guess.

    :param detector: the detector to set `tth_inital_guess` on
    :type detector: MCAElementConfig
    :param material: the material to show HKLs for
    :type material: MaterialConfig
    :param y: reference y data to plot
    :type y: np.ndarray
    :param x: reference x data to plot
    :type x: np.ndarray
    :return: None
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox
    import numpy as np
    from scipy.constants import physical_constants

    tth_initial_guess = detector.tth_initial_guess \
                        if detector.tth_initial_guess is not None \
                        else 5.0
    hkls, ds = material.unique_ds(
        tth_tol=detector.hkl_tth_tol, tth_max=detector.tth_max)
    hc = 1e7 * physical_constants['Planck constant in eV/Hz'][0] \
         * physical_constants['speed of light in vacuum'][0]
    def get_peak_locations(tth):
        return hc / (2. * ds * np.sin(0.5 * np.radians(tth)))

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('MCA channel energy (keV)')
    ax.set_ylabel('MCA intensity (counts)')
    ax.set_title('Adjust initial guess for $2\\theta$')
    hkl_lines = [ax.axvline(loc, c='k', ls='--', lw=1) \
                 for loc in get_peak_locations(tth_initial_guess)]

    # Callback for tth input
    def new_guess(tth):
        try:
            tth = float(tth)
        except:
            raise ValueError(f'Cannot convert {new_tth} to float')
        for i, (line, loc) in enumerate(zip(hkl_lines,
                                            get_peak_locations(tth))):
            line.remove()
            hkl_lines[i] = ax.axvline(loc, c='k', ls='--', lw=1)
        ax.get_figure().canvas.draw()
        detector.tth_initial_guess = tth

    # Setup tth input
    plt.subplots_adjust(bottom=0.25)
    tth_input = TextBox(plt.axes([0.125, 0.05, 0.15, 0.075]),
                        '$2\\theta$: ',
                        initial=tth_initial_guess)
    cid_update_tth = tth_input.on_submit(new_guess)

    # Setup "Confirm" button
    def confirm_selection(event):
        plt.close()
    confirm_b = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
    cid_confirm = confirm_b.on_clicked(confirm_selection)

    # Show figure for user interaction
    plt.show()

    # Disconnect all widget callbacks when figure is closed
    tth_input.disconnect(cid_update_tth)
    confirm_b.disconnect(cid_confirm)
