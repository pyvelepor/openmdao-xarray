import numpy as np
import openmdao.api as om
import openmdao.recorders.case
import openmdao.recorders.sqlite_reader as sqlite_reader
import xarray as xr
import xarray.backends as xr_backends

class BackendEntrypoint(xr_backends.BackendEntrypoint):
    """
    Backend for loading OpenMDAO case data into a xarray dataset.

    Lazy loading is not supported at this time.
    """
    open_dataset_parameters = [
        "filename_or_obj",
        "drop_variables",
        "source"
    ]

    def open_dataset(
        self,
        filename_or_obj,
        drop_variables=None,
        source="",
    ):
        """
        Returns case data for a given source,
        :param filename_or_obj: path to the OpenMDAO case file
        :param drop_variables: variables to exclude
        :param source: only load cases recorded by `source`. Defaults to "" to load cases recorded by all sources
        :return: dataset with case data, aligned to global iteration coordinates
        """
        case_reader: sqlite_reader.SqliteCaseReader = om.CaseReader(filename_or_obj)

        if drop_variables is None:
            drop_variables = []

        return xr.merge(
            collect_cases(
                case_reader.get_case(case_id)
                for case_id
                in case_reader.list_cases(
                    source=source,
                    out_stream=None
                )
            )
        ).drop_vars(drop_variables)

def get_input_absolute_name(case, promoted_name) -> dict:
    """
    Utility to extract OpenMDAO variable metadata from a case
    :param case: case object to extract metadata from
    :param promoted_name: name of OpenMDAO variable to extract metadata for
    :return: metadata for the OpenMDAO variable
    """

    return case._prom2abs['input'][promoted_name][0]

def get_output_absolute_name(case, promoted_name) -> dict:
    """
    Utility to extract OpenMDAO variable metadata from a case
    :param case: case object to extract metadata from
    :param promoted_name: name of OpenMDAO variable to extract metadata for
    :return: metadata for the OpenMDAO variable
    """

    if promoted_name in case._prom2abs["output"]:
        absolute_name = case._prom2abs["output"][promoted_name][0]
        return absolute_name

    # if variable name isn't in `case._prom2abs`, it's possible the promoted variable is
    # connected to an output variable from an independent variable component. In which case,
    # metadata needs to be looked up using metadata for input variables
    absolute_name = case._conns[get_input_absolute_name(case, promoted_name)]

    return absolute_name

def get_dimensions(variable_name, metadata):
    """
    Utility to determine names for xarray dimensions for an OpenMDAO variable based off the variable's shape
    :param variable_name: name of OpenMDAO variable
    :param metadata: metadata for OpenMDAO variable
    :return: names for xarray dimensions, formatted as {`name`}.dim_{n}
    """
    dims = []
    ndim = len(metadata["shape"])

    dims.extend(
        f"{variable_name}.dim_{i}"
        for i
        in range(ndim)
    )

    return dims

@xr.register_dataset_accessor("outputs")
class OutputAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj.filter_by_attrs(output=True)
        self._value = None
        self._residual = None

    @property
    def value(self):
        """Return value for output variables"""
        if self._value is None:
            self._value = self._obj.sel(output="value")

        return self._value

    @property
    def residual(self):
        """Return residual for output variables"""
        if self._residual is None:
            self._residual = self._obj.sel(output="residual")

        return self._residual

    @property
    def all(self):
        """Return value and residual for output variables"""
        return self._obj


@xr.register_dataset_accessor("driver")
class CaseAccessor:
    """
    xarray accessor to simplify querying a dataset for design variables, objectives, constraints, and responses
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._objectives = None
        self._design_vars = None
        self._constraints = None
        self._responses = None

    @property
    def design_vars(self):
        """Return design variables."""
        if self._design_vars is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            self._design_vars = self._obj.filter_by_attrs(design_var=True)

        return self._design_vars

    @property
    def objectives(self):
        """Return objective variables"""
        if self._objectives is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            self._objectives = self._obj.filter_by_attrs(objective=True)

        return self._objectives

    @property
    def constraints(self):
        """Return constraint variables"""
        if self._constraints is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            self._constraints = self._obj.filter_by_attrs(constraint=True)

        return self._constraints

    @property
    def responses(self):
        """Return response variables"""
        if self._responses is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            self._responses = self._obj.filter_by_attrs(response=True)

        return self._responses

    @property
    def all(self):
        """Return only objective, design, constraint, and response variables"""
        return xr.merge([
            self.objectives,
            self.design_vars,
            self.constraints,
            self.responses
        ])


@xr.register_dataset_accessor("source_vars")
class CaseAccessor:
    """
    xarray accessor to simplify querying a dataset for source variables
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._inputs = None
        self._outputs = None
        self._residuals = None
        self._derivatives = None

    @property
    def inputs(self):
        """Return values for input variables"""
        if self._inputs is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            self._inputs = self._obj.filter_by_attrs(input=True)

        return self._inputs

    @property
    def outputs(self):
        """Return values only for output variables"""
        if self._outputs is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            self._outputs = self._obj.filter_by_attrs(output=True).sel(output="value")

        return self._outputs

    @property
    def residuals(self):
        """Return residuals only for output variables"""
        if self._residuals is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            self._residuals = self._obj.filter_by_attrs(output=True).sel(output="residual")

        return self._residuals

    @property
    def derivatives(self):
        """Return derivatives only for output variables"""
        if self._derivatives is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            self._derivatives = self._obj.filter_by_attrs(derivative=True)

        return self._derivatives


def collect_inputs(case):
    """
    Loads data for each input variables in `case`
    :param case: `case` object to load data from
    :yield: xarray DataArray for each input variable
    """
    inputs = case.inputs

    if inputs is None:
        inputs = {}

    for var_name, data in inputs.items():
        metadata = case._abs2meta.get(var_name, None)

        if metadata is None:
            absolute_var_name = get_input_absolute_name(var_name)
            metadata = case._abs2meta[absolute_var_name]

        dimensions = ["iteration", *get_dimensions(var_name, metadata)]
        data = np.expand_dims(data, axis=0)

        yield xr.DataArray(
            data=data,
            dims=dimensions,
            coords={
                "iteration": [case.counter]
            },
            attrs={
                "desc": metadata["desc"],
                "shape": metadata["shape"],
                "units": metadata["units"],
                "input": True,
                "output": False,
                "derivative": False
            },
            name=var_name
        )


def collect_outputs(case):
    """
    Loads data for each output variables in `case`, including residuals
    :param case: `case` object to load data from
    :yield: xarray DataArray for each output variable

    Dimensions
    - `iteration`: global iteration coordinate (i.e. `case.counter`)
    - `output`: dimension for output value and residuals
    - `{variable name}.dim_{n}`: xarray dimension for each dimension in the output variable

    Non-dimensions
    - `source`: Source which recorded the case
    - `name`: source-specific name for the iteration coordinate

    Coordinates
    - `iteration`: [1, 2, ..., n] where n is total number of recorded cases.
    - `output`: ["value", "residual"]
    - `source`: [{case 1}.source, {case 2}.source, ..., {case n}.name]
    - `name`: [{case 1}.name, {case 2}.name, ..., {case n}.name]
    """
    outputs = case.outputs
    residuals = case.residuals

    if outputs is None:
        outputs = {}

    if residuals is None:
        residuals = {}

    for variable_name, value in outputs.items():
        metadata = case._abs2meta.get(variable_name, None)

        if metadata is None:
            absolute_variable_name = get_output_absolute_name(case, variable_name)
            metadata = case._abs2meta[absolute_variable_name]

        dimensions = ["iteration", "output", *get_dimensions(variable_name, metadata)]
        residual = residuals.get(variable_name, np.full(shape=metadata["shape"], fill_value=np.nan))

        data = np.expand_dims(
            np.stack((value, residual), axis=0),
            axis=0
        )

        yield xr.DataArray(
            data=data,
            dims=dimensions,
            coords={
                "iteration": [case.counter], },
            attrs={
                "desc": metadata["desc"],
                "units": metadata["units"],
                "shape": metadata["shape"],
                "res_units": metadata.get("res_units", None),
                "ref": metadata["ref"],
                "res_ref": metadata["res_ref"],
                "ref0": metadata["ref0"],
                "input": False,
                "output": True,
                "derivative": False,
                "objective": True if "objective" in metadata["type"] else False,
                "design_var": True if "desvar" in metadata["type"] else False,
                "constraint": True if "constraint" in metadata["type"] else False,
                "response": True if "response" in metadata["type"] else False,
            },
            name=variable_name
        )


def collect_derivatives(case):
    derivatives = case.derivatives

    if derivatives is None:
        derivatives = {}

    for variable_name, data in derivatives.items():
        of_variable_name, wrt_variable_name = variable_name

        of_variable_metadata = case._abs2meta.get(of_variable_name, None)
        wrt_variable_metadata = case._abs2meta.get(wrt_variable_name, None)

        if of_variable_metadata is None:
            absolute_variable_name = get_output_absolute_name(case, of_variable_name)
            of_variable_metadata = case._abs2meta[absolute_variable_name]

        if wrt_variable_metadata is None:
            absolute_variable_name = get_output_absolute_name(case, wrt_variable_name)
            wrt_variable_metadata = case._abs2meta[absolute_variable_name]

        dimensions = [
            "iteration",
            *get_dimensions(of_variable_name, of_variable_metadata),
            *get_dimensions(wrt_variable_name, wrt_variable_metadata)
        ]

        data = np.expand_dims(data, axis=0)

        yield xr.DataArray(
            data=data,
            dims=dimensions,
            coords={
                "iteration": [case.counter]
            },
            attrs={
                "desc": f"Derivative of {of_variable_name} w.r.t. {wrt_variable_name}",
                "input": False,
                "output": False,
                "derivative": True
            },
            name=variable_name
        )


def collect_cases(cases):
    """
    Loads data for input variables, output variables (including residuals) and derivatives into a xarray Data
    :param cases: iterable of cases to load data from
    :yield: xarray Dataset

    Variables
    - {variable name}: for each input, output, and derivative recorded by `case.source` for the iteration

    Dimensions
    - `iteration`: global iteration (i.e. `case.counter`)
    - `output`: dimension for output value and residuals
    - `{variable name}.dim_{n}`: xarray dimension per each variable per each dimension in the OpenMDAO variable

    Non-dimensions
    - `source`: Source which recorded the case (e.g. "root", "driver", etc.), aligned with `iteration`
    - `name`: source-specific iteration coordinate (e.g. "rank0:root|SLSQPDriver|0, etc.), aligned with `iteration`

    Coordinates
    - `iteration`: [`case.counter`]
    - `output`: ["value", "residual"]
    - `source`: [`case.source`]
    - `name`: [`case.name`]


    """
    for case in cases:
        yield xr.Dataset(
            data_vars={
                data_array.name: data_array
                for data_array
                in (
                    *collect_inputs(case),
                    *collect_outputs(case),
                    *collect_derivatives(case)
                )
            },
            coords={
                "iteration": [case.counter],
                "source": ("iteration", [case.source]),
                "name": ("iteration", [case.name]),
                "output": ["value", "residual"]
            }
        )