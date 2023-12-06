#!/usr/bin/env python

from CHAP.reader import Reader


class PreIntegrationReader(Reader):
    """Read in lists of data from parameters required to perfom
    azimuthal integration with pyFAI on data from one or more
    detectors. Must be used immediatey prior to
    saxswaxs.IntegrationProcessor."""

    def read(self, data=[], detectors=[], **kwargs):
        """Return the raw dataset, AzimuthalIntegrator, and (optional)
        mask for one or more detectors.

        :param data: Nexus object containing raw datasets for the
            configured detectors.
        :type data: CHAP.pipeline.PipelineData
        :param detectors: List of dictionaries that each configure
            setup parameters for a single detector. The dictionaries
            contain 3 keys: `'nxfield_path'` (a string that specifies
            the path to the raw dataset for this detector in the input
            `data`), `'poni_file'` (a string that points towars a
            pyFAI PONI file to use on this detector's data), and
            `'mask_file'` (a string that points towards a n image file
            to be used as a mask on this detector's data -- this is
            the only optional entry).
        :type detectors: list[dict[str, str]]
        :raises ValueError: If the resulting data loaded from inputs
            is not valid for use with `saxswaxs.IntegrationProcessor`
        :return: Input data appropriate for use with
            saxswaxs.IntegrationProcessor
        """
        results = self.load_inputs(data, detectors, kwargs.get('inputdir'))
        self.validate(results)
        return results

    def load_inputs(self, data, detectors, inputdir):
        """Validate the arguments supplied by the Pipeline to the
        `read` method and load the inputs requested. Returned the
        loaded items.

        :param data: Nexus object containing raw datasets for the
            configured detectors.
        :type data: nexusformat.nexus.NXobject
        :param detectors:
        :param detectors: List of dictionaries that each configure
            setup parameters for a single detector. The dictionaries
            contain 3 keys: `'nxfield_path'` (a string that specifies
            the path to the raw dataset for this detector in the input
            `data`), `'poni_file'` (a string that points towars a
            pyFAI PONI file to use on this detector's data), and
            `'mask_file'` (a string that points towards a n image file
            to be used as a mask on this detector's data -- this is
            the only optional entry).
        :type detectors: list[dict[str, str]]
        :raise: ValueError if input parameters are not valid
        :return: objects loaded from the files and paths provided,
            ready for use by saxswaxs.IntegrationProcessor.
        """
        from nexusformat.nexus import NXfield, NXobject
        from numpy import ndarray
        import os
        from pyFAI import load as pyfai_load
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
        from typing import Optional

        # Validate input data from the Pipeline
        if len(data) == 0:
            msg = 'Must provide input NeXus data'
            self.logger.error(msg)
            raise ValueError(msg)
        data = self.unwrap_pipelinedata(data)[0]
        if not isinstance(data, NXobject):
            msg = 'Input data must be a NeXus object'
            self.logger.error(msg)
            raise TypeError(msg)

        # Validate and load input detector configurations
        results = []
        detector_schema = [
            {'key': 'nxfield_path', 'optional': False, 'isfile': False,
             'load_function': data.__getitem__,
             'results_key': 'data', 'results_type': NXfield},
            {'key': 'poni_file', 'optional': False, 'isfile': True,
             'load_function': pyfai_load,
             'results_key': 'ai', 'results_type': AzimuthalIntegrator},
            {'key': 'mask_file', 'optional': True, 'isfile': True,
             'load_function': load_image,
             'results_key': 'mask', 'results_type': Optional[ndarray]}
        ]
        for d in detectors:
            result = {}
            # Validate schema items
            for s in detector_schema:
                if s['key'] in d:
                    if s['isfile']:
                        filename = os.path.join(inputdir, d[s['key']])
                        if not os.path.isfile(filename):
                            msg = f'File {filename} does not exist'
                            self.logger.error(msg)
                            raise ValueError(msg)
                        d[s['key']] = filename
                    result[s['results_key']] = s['load_function'](d[s['key']])
                else:
                    if not s['optional']:
                        msg = f'Missing value for "{s["key"]}"'
                        self.logger.error(msg)
                        raise KeyError(msg)
                    result[s['results_key']] = None
                if not isinstance(result[s['results_key']], s['results_type']):
                    msg = (
                        f'Type for "{s["results_key"]}" must be '
                        + f'{s["results_type"]}, '
                        + f'not {type(result[s["results_key"]])}')
                    self.logger.error(msg)
                    raise TypeError(msg)

            results.append(result)
        return results

    def validate(self, results):
        """Assure that the shapes of masks (if used) match the shape
        of the corresponding detector data and that the map shapes of
        multiple detector datasets (if used) match each other.

        :param result: Loaded parameters to validate.
        :type result: list[dict]
        :raises ValueError: If there is a mismatch of array shapes in
            `result`.
        :returns: None
        """
        map_shape = results[0]['data'].shape[:-2]
        self.logger.debug(f'Validating loaded data (map shape: {map_shape})')
        for detector in results:
            _map_shape = detector['data'].shape[:-2]
            if map_shape != _map_shape:
                raise ValueError('Detector datasets have different map shapes')
            if detector.get('mask') is not None:
                detector_shape = detector['data'].shape[-2:]
                mask_shape = detector['mask'].shape
                if detector_shape != mask_shape:
                    raise ValueError(
                        f'Shape of mask array ({mask_shape}) does not match '
                        + f'shape of detector ({detector_shape})')


def load_image(filename):
    """Return a numpy array of data from an image file. File will be
    loaded with `PIL.Image.open`.

    :param filename: The image file
    :type filename: str
    :returns: A 2D array of image data from `filename`
    :rtype: numpy.ndarray
    """
    from PIL import Image
    from numpy import array
    return array(Image.open(filename))


if __name__ == '__main__':
    from CHAP.reader import main
    main()
