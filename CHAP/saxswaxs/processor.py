#!/usr/bin/env python

from CHAP.processor import Processor


class IntegrationProcessor(Processor):
    """Processor for integrating data with pyFAI in the context of a
    SAXS/WAXS workflow. Must be preceeded by
    `CHAP.saxswaxs.PreIntegrationReader` when used in a Pipeline.
    """
    def process(self, data, config, **kwargs):
        """Return integrated data.

        :param data: Preparatory pyFAI integration inputs returned
            directly from `saxswaxs.PreIntegrationReader`
        :type data: list[PipelineData]
        :param config: Integration parameters (will be validated by
            `saxswaxs.models.IntegrationConfig`)
        :type config: dict
        :returns: Integrated data & metadata associated with the
            processing.
        :rtype: nexusformat.nexus.NXprocess
        """
        from CHAP.saxswaxs.models import IntegrationConfig

        pipeline_data = self.unwrap_pipelinedata(data)[0]
        detector_data = [d['data'] for d in pipeline_data]
        integrators = [d['ai'] for d in pipeline_data]
        masks = [d['mask'] for d in pipeline_data]

        config = IntegrationConfig(**config)

        #return self.get_nxprocess(detector_data, integrators, masks, config)
        nxprocess = self.get_nxprocess(
            detector_data, integrators, masks, config)
        self.logger.debug(nxprocess.tree)
        return nxprocess

    def get_nxprocess(
            self, detector_data, integrators, masks, integration_config):
        """Return a NXprocess containing the integrated data and
        associated processing metadata.

        :param detector_data: List of NXfields containing raw data for
            one or more detectors.
        :type detector_data: list[nexusformat.nexus.NXfield]
        :param integrators: List of azimuthal integrators to use for
            each detector.
        :type integrators: list[pyFAI.azimuthalIntegrator.AzimuthalIntegrator]
        :param masks: List of mask arrays to use for each detector.
        :type masks: list[numpy.ndarray]
        :param integration_config: pyFAI integration parameters.
        :type integration_config: CHAP.saxswaxs.models.IntegrationConfig
        :returns: Integrated detector data & metadata
        :rtype: nexusformat.nexus.NXprocess
        """
        from CHAP import version
        from nexusformat.nexus import NXdata, NXfield, NXprocess

        if integration_config.error_model is not None:
            I, I_sigma = self.get_integrated_data(
                detector_data, integrators, masks, integration_config)
            I_sigma = NXfield(
                I_sigma, name='I_sigma',
                attrs={'units': 'a.u', 'long_name': 'Intensity (a.u)'})
        else:
            I = self.get_integrated_data(
                detector_data, integrators, masks, integration_config)
            I_sigma = None
        I = NXfield(
            I, name='I',
            attrs={'units': 'a.u', 'long_name': 'Intensity (a.u)'})
        return NXprocess(
            name=integration_config.title,
            data=NXdata(
                signal=I,
                axes=self.get_nxaxes(detector_data, integration_config),
                errors=I_sigma),
            program=f'{__name__}.{self.__class__.__name__}',
            version=version,
            attrs={'default': 'data'}
        )

    def get_nxaxes(self, detector_data, integration_config):
        """Return a tuple of NXfields representing the axes for the
        integrated dataset.

        :param detector_data: List of NXfields containing raw data for
            one or more detectors.
        :type detector_data: list[nexusformat.nexus.NXfield]
        :param integration_config: pyFAI integration parameters.
        :type integration_config: CHAP.saxswaxs.models.IntegrationConfig
        :return: Coordinate axes for the resulting integrated dataset
        :rtype: tuple[nexusformat.nexus.NXfield]
        """
        from nexusformat.nexus import NXfield
        from pyFAI.units import to_unit, RADIAL_UNITS, AZIMUTHAL_UNITS
        ANY_UNITS = {**RADIAL_UNITS, **AZIMUTHAL_UNITS}

        nxfield = detector_data[0]
        if not isinstance(nxfield, NXfield):
            raise TypeError('Cannot determine map axes')
        map_axes = nxfield.nxaxes

        integrated_data_coords = integration_config.integrated_data_coordinates
        integrated_data_axes = []
        for coord, values in integrated_data_coords.items():
            unit = to_unit(coord, type_=ANY_UNITS)
            if unit.short_name is None:
                name = unit.name.split('_', 1)[0]
            else:
                name = unit.short_name
            if unit.unit_symbol is None:
                unit_symbol = unit.name.rsplit('_', 1)[-1]
            else:
                unit_symbol = unit.unit_symbol
            integrated_data_axes.append(
                NXfield(
                    name=name,
                    value=values,
                    attrs={'units': unit_symbol,
                           'long_name': unit.label})
            )
        return (*map_axes, *integrated_data_axes)

    def get_azimuthal_integrators(self, integrators, integration_config):
        """Return a list of azimuthal integrators that have been
        artificially rotated about the axis of the beam so that the
        azimuthal integration ranges indicated in `integration_config`
        will result in integrated data ranges that are continuous in
        that direction.

        :param integrators: List of AzimuthalIntegrators to
            artificially rotate
        :type integrators: list[pyFAI.azimuthalIntegrator.AzimuthalIntegrator]
        :param integration_config: pyFAI integration parameters
        :type integration_config: CHAP.saxswaxs.models.IntegrationConfig
        :returns: List of atificially roated azimuthal integrators
        :rtype: list[pyFAI.azimuthalIntegrator.AzimuthalIntegrator]
        """
        from copy import deepcopy
        from numpy import pi

        chi_offset = integration_config.get_azimuthal_adjustment()
        ais = []
        for ai in integrators:
            _ai = deepcopy(ai)
            _ai.rot3 += chi_offset * pi/180
            ais.append(_ai)
        return ais

    def get_multi_geometry_integrator(self, integrators, integration_config):
        """Return a multi-geometry integrator appropriate for use with
        azimuthal or cake integration.

        :param integrators: List of AzimuthalIntegrators to use in the
            multi-geometry integrator
        :type integrators: list[pyFAI.azimuthalIntegrator.AzimuthalIntegrator]
        :param integration_config: pyFAI integration parameters
        :type integration_config: CHAP.saxswaxs.models.IntegrationConfig
        :returns: pyFAI multi-geometry integrator
        :rtype: pyFAI.multi_geometry.MultiGeometry
        """
        from pyFAI.multi_geometry import MultiGeometry
        ais = self.get_azimuthal_integrators(integrators, integration_config)
        return MultiGeometry(ais,
                             unit=integration_config.radial_units,
                             radial_range=(integration_config.radial_min,
                                           integration_config.radial_max),
                             azimuth_range=(integration_config.azimuthal_min,
                                            integration_config.azimuthal_max))

    def get_integrated_data(
            self, detector_data, integrators, masks, integration_config):
        """Return arrays of integrated data and the associated
        uncertainties (if applicable)

        :param detector_data: List of NXfields containing raw data for
            one or more detectors.
        :type detector_data: list[nexusformat.nexus.NXfield]
        :param integrators: List of azimuthal integrators to use for
            each detector.
        :type integrators: list[pyFAI.azimuthalIntegrator.AzimuthalIntegrator]
        :param masks: List of mask arrays to use for each detector.
        :type masks: list[numpy.ndarray]
        :param integration_config: pyFAI integration parameters.
        :type integration_config: CHAP.saxswaxs.models.IntegrationConfig
        :returns: Integrated data [, uncertainties]
        :rtype: numpy.ndarray [, Optional[numpy.ndarray]]
        """
        from multiprocessing.pool import ThreadPool
        import numpy as np
        map_shape = detector_data[0].shape[:-2]
        I = np.empty(
            (*map_shape, *integration_config.integrated_data_shape))
        if integration_config.error_model is not None:
            I_sigma = np.empty(
                (*map_shape, *integration_config.integrated_data_shape))

        if integration_config.integration_type == 'radial':
            chi_offset = integration_config.get_azimuthal_adjustment()
            chi_min = integration_config.azimuthal_min - chi_offset
            chi_max = integration_config.azimuthal_max - chi_offset
            def _get_integrated_data(data):
                """Return radially-integrated data for a single index in the
                map of raw data.

                :param data: A single frame of data from one or more
                    detectors.
                :type data: list[numpy.ndarray]
                :returns: Radially-integrated intensity
                :rtype: numpy.ndarray
                """
                import numpy as np
                intensities = []
                for _data, integrator, mask in zip(data, integrators, masks):
                    chi, intensity = integrator.integrate_radial(
                        _data, integration_config.azimuthal_npt,
                        unit=integration_config.azimuthal_units,
                        azimuth_range=(chi_min,chi_max),
                        radial_unit=integration_config.radial_units,
                        radial_range=(integration_config.radial_min,
                                      integration_config.radial_max),
                        mask=mask)
                    intensities.append(intensity)
                intensity = np.nansum(intensities, axis=0)
                intensity = np.where(intensity==0, np.nan, intensity)
                if integration_config.right_handed:
                    intensity = np.flip(intensity)
                return intensity
        if integration_config.integration_type == 'azimuthal':
            integrator = self.get_multi_geometry_integrator(
                integrators, integration_config)
            def _get_integrated_data(data):
                """Return azimuthally-integrated data for a single index
                in the map of raw data.

                :param data: A single frame of data from one or more
                    detectors.
                :type data: list[numpy.ndarray]
                :returns: Azimuthally-integrated intensity, and errors for
                    intensity if requested
                :rtype: numpy.ndarray
                """
                result = integrator.integrate1d(
                    data, lst_mask=masks,
                    error_model=integration_config.error_model,
                    npt=integration_config.radial_npt)
                if result.sigma is None:
                    return result.intensity
                return result.intensity, result.sigma
        if integration_config.integration_type == 'cake':
            integrator = self.get_multi_geometry_integrator(
                integrators, integration_config)
            def _get_integrated_data(data):
                """Return cake-integrated data for a single index in the
                map of raw data.

                :param data: A single frame of data from one or more
                    detectors.
                :type data: list[numpy.ndarray]
                :returns: Cake-integrated intensity, and errors for
                    intensity if requested
                :rtype: numpy.ndarray
                """
                result = integrator.integrate2d(
                    data, lst_mask=masks, method='bbox',
                    error_model=integration_config.error_model,
                    npt_rad=integration_config.radial_npt,
                    npt_azim=integration_config.azimuthal_npt)

                if integration_config.right_handed:
                    from pyFAI.containers import Integrate2dResult
                    import numpy as np
                    if integration_config.error_model is not None:
                        # Only works for versions of pyfai pyfai>=2023.*
                        result = Integrate2dResult(
                            np.flip(result.intensity, axis=0),
                            result.radial, result.azimuthal,
                            np.flip(result.sigma, axis=0))
                    else:
                        result = Integrate2dResult(
                            np.flip(result.intensity, axis=0),
                            result.radial, result.azimuthal)

                if result.sigma is None:
                    return result.intensity
                return result.intensity, result.sigma

        if integration_config.error_model is None:
            def fill_data(map_index):
                I[map_index] = _get_integrated_data(
                    [d[map_index] for d in detector_data])
        else:
            def fill_data(map_index):
                I[map_index], I_sigma[map_index] = _get_integrated_data(
                    [d[map_index] for d in detector_data])

        with ThreadPool(processes=4) as pool:
            pool.map(fill_data, np.ndindex(map_shape))

        if integration_config.error_model is None:
            return I
        else:
            return I, I_sigma


if __name__ == '__main__':
    from CHAP.processor import main
    main()
