from abc import ABCMeta
from typing import Dict, Optional, Union

import copy
import timeit
import joblib
import inspect
import numpy as np
import scipy.sparse
from ConfigSpace import Configuration
from sklearn.pipeline import Pipeline

import autosklearn.pipeline.create_searchspace_util
from autosklearn.askl_typing import FEAT_TYPE_TYPE

from .components.base import AutoSklearnChoice, AutoSklearnComponent

DATASET_PROPERTIES_TYPE = Dict[str, Union[str, int, bool]]
PIPELINE_DATA_DTYPE = Union[
    np.ndarray,
    scipy.sparse.bsr_matrix,
    scipy.sparse.coo_matrix,
    scipy.sparse.csc_matrix,
    scipy.sparse.csr_matrix,
    scipy.sparse.dia_matrix,
    scipy.sparse.dok_matrix,
    scipy.sparse.lil_matrix,
]


class BasePipeline(Pipeline):
    """Base class for all pipeline objects.

    Notes
    -----
    This class should not be instantiated, only subclassed."""

    __metaclass__ = ABCMeta

    def __init__(
        self,
        config=None,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        steps=None,
        dataset_properties=None,
        include=None,
        exclude=None,
        random_state=None,
        init_params=None,
    ):

        self.init_params = init_params if init_params is not None else {}
        self.include = include if include is not None else {}
        self.exclude = exclude if exclude is not None else {}
        self.dataset_properties = (
            dataset_properties if dataset_properties is not None else {}
        )
        self.random_state = random_state
        self.feat_type = feat_type

        if steps is None:
            self.steps = self._get_pipeline_steps(
                feat_type=feat_type, dataset_properties=dataset_properties
            )
        else:
            self.steps = steps

        self._validate_include_exclude_params()

        self.config_space = self.get_hyperparameter_search_space(feat_type=feat_type)

        if config is None:
            self.config = self.config_space.get_default_configuration()
        else:
            if isinstance(config, dict):
                config = Configuration(self.config_space, config)
            if self.config_space != config.configuration_space:
                print(self.config_space._children)
                print(config.configuration_space._children)
                import difflib

                diff = difflib.unified_diff(
                    str(self.config_space).splitlines(),
                    str(config.configuration_space).splitlines(),
                )
                diff = "\n".join(diff)
                raise ValueError(
                    "Configuration passed does not come from the "
                    "same configuration space. Differences are: "
                    "%s" % diff
                )
            self.config = config

        self.set_hyperparameters(
            self.config, feat_type=feat_type, init_params=init_params
        )

        super().__init__(steps=self.steps)

        self._additional_run_info = {}

    def fit(self, X, y, **fit_params):
        """Fit the selected algorithm to the training data.

        Parameters
        ----------
        X : array-like or sparse, shape = (n_samples, n_features)
            Training data. The preferred type of the matrix (dense or sparse)
            depends on the estimator selected.

        y : array-like
            Targets

        fit_params : dict
            See the documentation of sklearn.pipeline.Pipeline for formatting
            instructions.

        Returns
        -------
        self : returns an instance of self.

        Raises
        ------
        NoModelException
            NoModelException is raised if fit() is called without specifying
            a classification algorithm first.
        """
        X, fit_params = self.fit_transformer(X, y, **fit_params)
        self.fit_estimator(X, y, **fit_params)
        return self

    def fit_transformer(self, X, y, fit_params=None):
        self.num_targets = 1 if len(y.shape) == 1 else y.shape[1]
        if fit_params is None:
            fit_params = {}
        fit_params = {
            key.replace(":", "__"): value for key, value in fit_params.items()
        }
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        return Xt, fit_params_steps[self.steps[-1][0]]

    def fit_estimator(self, X, y, **fit_params):
        fit_params = {
            key.replace(":", "__"): value for key, value in fit_params.items()
        }
        self._final_estimator.fit(X, y, **fit_params)
        return self

    def iterative_fit(self, X, y, n_iter=1, **fit_params):
        self._final_estimator.iterative_fit(X, y, n_iter=n_iter, **fit_params)

    def estimator_supports_iterative_fit(self):
        return self._final_estimator.estimator_supports_iterative_fit()

    def get_max_iter(self):
        if self.estimator_supports_iterative_fit():
            return self._final_estimator.get_max_iter()
        else:
            raise NotImplementedError()

    def configuration_fully_fitted(self):
        return self._final_estimator.configuration_fully_fitted()

    def get_current_iter(self):
        return self._final_estimator.get_current_iter()

    def predict(self, X, batch_size=None):
        """Predict the classes using the selected model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        batch_size: int or None, defaults to None
            batch_size controls whether the pipeline will be
            called on small chunks of the data. Useful when calling the
            predict method on the whole array X results in a MemoryError.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Returns the predicted values"""

        if batch_size is None:
            return super().predict(X).astype(self._output_dtype)
        else:
            if not isinstance(batch_size, int):
                raise ValueError(
                    "Argument 'batch_size' must be of type int, "
                    "but is '%s'" % type(batch_size)
                )
            if batch_size <= 0:
                raise ValueError(
                    "Argument 'batch_size' must be positive, " "but is %d" % batch_size
                )

            else:
                if self.num_targets == 1:
                    y = np.zeros((X.shape[0],), dtype=self._output_dtype)
                else:
                    y = np.zeros(
                        (X.shape[0], self.num_targets), dtype=self._output_dtype
                    )

                # Copied and adapted from the scikit-learn GP code
                for k in range(max(1, int(np.ceil(float(X.shape[0]) / batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    y[batch_from:batch_to] = self.predict(
                        X[batch_from:batch_to], batch_size=None
                    )

                return y

    def set_hyperparameters(
        self,
        configuration,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        init_params=None,
    ):
        self.config = configuration

        for node_idx, n_ in enumerate(self.steps):
            node_name, node = n_

            sub_configuration_space = node.get_hyperparameter_search_space(
                feat_type=feat_type, dataset_properties=self.dataset_properties
            )
            sub_config_dict = {}
            for param in configuration:
                if param.startswith("%s:" % node_name):
                    value = configuration[param]
                    new_name = param.replace("%s:" % node_name, "", 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(
                sub_configuration_space, values=sub_config_dict
            )

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith("%s:" % node_name):
                        value = init_params[param]
                        new_name = param.replace("%s:" % node_name, "", 1)
                        sub_init_params_dict[new_name] = value
            else:
                sub_init_params_dict = None

            if isinstance(
                node, (AutoSklearnChoice, AutoSklearnComponent, BasePipeline)
            ):
                node.set_hyperparameters(
                    feat_type=feat_type,
                    configuration=sub_configuration,
                    init_params=sub_init_params_dict,
                )
            else:
                raise NotImplementedError("Not supported yet!")

        # In-code check to make sure init params
        # is checked after pipeline creation
        self._check_init_params_honored(init_params)

        return self

    def get_hyperparameter_search_space(
        self, feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        """Return the configuration space for the CASH problem.

        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the AutoSklearnClassifier.

        """
        if not hasattr(self, "config_space") or self.config_space is None:
            self.config_space = self._get_hyperparameter_search_space(
                feat_type=feat_type,
                include=self.include,
                exclude=self.exclude,
                dataset_properties=self.dataset_properties,
            )
        return self.config_space

    def _get_hyperparameter_search_space(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        include=None,
        exclude=None,
        dataset_properties=None,
    ):
        """Return the configuration space for the CASH problem.

        This method should be called by the method
        get_hyperparameter_search_space of a subclass. After the subclass
        assembles a list of available estimators and preprocessor components,
        _get_hyperparameter_search_space can be called to do the work of
        creating the actual
        ConfigSpace.configuration_space.ConfigurationSpace object.

        Parameters
        ----------
        feat_type: dict
            python dictionary which maps the columns of the dataset to the data types

        estimator_name : str
            Name of the estimator hyperparameter which will be used in the
            configuration space. For a classification task, this would be
            'classifier'.

        estimator_components : dict {name: component}
            Dictionary with all estimator components to be included in the
            configuration space.

        preprocessor_components : dict {name: component}
            Dictionary with all preprocessor components to be included in the
            configuration space. .

        always_active : list of str
            A list of components which will always be active in the pipeline.
            This is useful for components like imputation which have
            hyperparameters to be configured, but which do not have any parent.

        default_estimator : str
            Default value for the estimator hyperparameter.

        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the AutoSklearnClassifier.
        """
        raise NotImplementedError()

    def _get_base_search_space(
        self,
        cs,
        dataset_properties,
        include,
        exclude,
        pipeline,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
    ):
        if include is None:
            if self.include is None:
                include = {}
            else:
                include = self.include

        keys = [pair[0] for pair in pipeline]
        for key in include:
            if key not in keys:
                raise ValueError(
                    "Invalid key in include: %s; should be one " "of %s" % (key, keys)
                )

        if exclude is None:
            if self.exclude is None:
                exclude = {}
            else:
                exclude = self.exclude

        keys = [pair[0] for pair in pipeline]
        for key in exclude:
            if key not in keys:
                raise ValueError(
                    "Invalid key in exclude: %s; should be one " "of %s" % (key, keys)
                )

        if "sparse" not in dataset_properties:
            # This dataset is probably dense
            dataset_properties["sparse"] = False
        if "signed" not in dataset_properties:
            # This dataset probably contains unsigned data
            dataset_properties["signed"] = False

        matches = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline=pipeline,
            dataset_properties=dataset_properties,
            include=include,
            exclude=exclude,
        )

        # Now we have only legal combinations at this step of the pipeline
        # Simple sanity checks
        assert np.sum(matches) != 0, "No valid pipeline found."

        assert np.sum(matches) <= np.size(
            matches
        ), "'matches' is not binary; %s <= %d, %s" % (
            str(np.sum(matches)),
            np.size(matches),
            str(matches.shape),
        )

        # Iterate each dimension of the matches array (each step of the
        # pipeline) to see if we can add a hyperparameter for that step
        for node_idx, n_ in enumerate(pipeline):
            node_name, node = n_

            is_choice = isinstance(node, AutoSklearnChoice)

            # if the node isn't a choice we can add it immediately because it
            #  must be active (if it wasn't, np.sum(matches) would be zero
            if not is_choice:
                cs.add_configuration_space(
                    node_name,
                    node.get_hyperparameter_search_space(
                        dataset_properties=dataset_properties, feat_type=feat_type
                    ),
                )
            # If the node is a choice, we have to figure out which of its
            #  choices are actually legal choices
            else:
                choices_list = (
                    autosklearn.pipeline.create_searchspace_util.find_active_choices(
                        matches,
                        node,
                        node_idx,
                        dataset_properties,
                        include.get(node_name),
                        exclude.get(node_name),
                    )
                )
                sub_config_space = node.get_hyperparameter_search_space(
                    feat_type=feat_type,
                    dataset_properties=dataset_properties,
                    include=choices_list,
                )
                cs.add_configuration_space(node_name, sub_config_space)

        # And now add forbidden parameter configurations
        # According to matches
        if np.sum(matches) < np.size(matches):
            cs = autosklearn.pipeline.create_searchspace_util.add_forbidden(
                conf_space=cs,
                pipeline=pipeline,
                matches=matches,
                dataset_properties=dataset_properties,
                include=include,
                exclude=exclude,
            )

        return cs

    def _check_init_params_honored(self, init_params):
        """
        Makes sure that init params is honored at the implementation level
        """
        if init_params is None or len(init_params) < 1:
            # None/empty dict, so no further check required
            return

        # There is the scenario, where instance is passed as an argument to the
        # init_params 'instance': '{"task_id": "73543c4a360aa24498c0967fbc2f926b"}'}
        # coming from smac instance. Remove this key to make the testing stricter
        init_params.pop("instance", None)

        for key, value in init_params.items():

            if ":" not in key:
                raise ValueError(
                    "Unsupported argument to init_params {}."
                    "When using init_params, a hierarchical format like "
                    "node_name:parameter must be provided.".format(key)
                )
            node_name = key.split(":", 1)[0]
            if node_name not in self.named_steps.keys():
                raise ValueError(
                    "The current node name specified via key={} of init_params "
                    "is not valid. Valid node names are {}".format(
                        key, self.named_steps.keys()
                    )
                )
                continue
            variable_name = key.split(":")[-1]
            node = self.named_steps[node_name]
            if isinstance(node, BasePipeline):
                # If dealing with a sub pipe,
                # Call the child _check_init_params_honored with the updated config
                node._check_init_params_honored(
                    {key.replace("%s:" % node_name, "", 1): value}
                )
                continue

            if isinstance(node, AutoSklearnComponent):
                node_dict = vars(node)
            elif isinstance(node, AutoSklearnChoice):
                node_dict = vars(node.choice)
            else:
                raise ValueError("Unsupported node type {}".format(type(node)))

            if variable_name not in node_dict or node_dict[variable_name] != value:
                raise ValueError(
                    "Cannot properly set the pair {}->{} via init_params"
                    "".format(key, value)
                )

    def __repr__(self):
        class_name = self.__class__.__name__

        configuration = {}
        self.config._populate_values()
        for hp_name in self.config:
            if self.config[hp_name] is not None:
                configuration[hp_name] = self.config[hp_name]

        configuration_string = "".join(
            [
                "configuration={\n  ",
                ",\n  ".join(
                    [
                        "'%s': %s" % (hp_name, repr(configuration[hp_name]))
                        for hp_name in sorted(configuration)
                    ]
                ),
                "}",
            ]
        )

        if len(self.dataset_properties) > 0:
            dataset_properties_string = []
            dataset_properties_string.append("dataset_properties={")
            for i, item in enumerate(self.dataset_properties.items()):
                if i != 0:
                    dataset_properties_string.append(",\n  ")
                else:
                    dataset_properties_string.append("\n  ")

                if isinstance(item[1], str):
                    dataset_properties_string.append("'%s': '%s'" % (item[0], item[1]))
                else:
                    dataset_properties_string.append("'%s': %s" % (item[0], item[1]))
            dataset_properties_string.append("}")
            dataset_properties_string = "".join(dataset_properties_string)

            return_value = "%s(%s,\n%s)" % (
                class_name,
                configuration,
                dataset_properties_string,
            )
        else:
            return_value = "%s(%s)" % (class_name, configuration_string)

        return return_value

    def _get_pipeline_steps(
        self, dataset_properties, feat_type: Optional[FEAT_TYPE_TYPE] = None
    ):
        raise NotImplementedError()

    def _get_estimator_hyperparameter_name(self):
        raise NotImplementedError()

    def get_additional_run_info(self):
        """Allows retrieving additional run information from the pipeline.

        Can be overridden by subclasses to return additional information to
        the optimization algorithm.
        """
        return self._additional_run_info

    def _validate_include_exclude_params(self):
        if self.include is not None and self.exclude is not None:
            for key in self.include.keys():
                if key in self.exclude.keys():
                    raise ValueError(
                        "Cannot specify include and exclude for same step '{}'.".format(
                            key
                        )
                    )

        supported_steps = {
            step[0]: step[1]
            for step in self.steps
            if isinstance(step[1], AutoSklearnChoice)
        }
        for arg in ["include", "exclude"]:
            argument = getattr(self, arg)
            if not argument:
                continue
            for key in list(argument.keys()):
                if key not in supported_steps:
                    raise ValueError(
                        "The provided key '{}' in the '{}' argument is not valid. The"
                        " only supported keys for this task are {}".format(
                            key, arg, list(supported_steps.keys())
                        )
                    )

                candidate_components = argument[key]
                if not (
                    isinstance(candidate_components, list) and candidate_components
                ):
                    raise ValueError(
                        "The provided value of the key '{}' in the '{}' argument is "
                        "not valid. The value must be a non-empty list.".format(
                            key, arg
                        )
                    )

                available_components = list(
                    supported_steps[key]
                    .get_available_components(
                        dataset_properties=self.dataset_properties
                    )
                    .keys()
                )
                for component in candidate_components:
                    if component not in available_components:
                        raise ValueError(
                            "The provided component '{}' for the key '{}' in the '{}'"
                            " argument is not valid. The supported components for the"
                            " step '{}' for this task are {}".format(
                                component, key, arg, key, available_components
                            )
                        )
    # BELOW FUNCTIONS ARE DIRECTLY COPIED FROM sklearn 0.24.X github repo
    # https://github.com/scikit-learn/scikit-learn/blob/0.24.X/sklearn/pipeline.py
    def _check_fit_params(self, **fit_params):
        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}
        for pname, pval in fit_params.items():
            if '__' not in pname:
                raise ValueError(
                    "Pipeline.fit does not accept the {} parameter. "
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipeline.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`.".format(pname))
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        return fit_params_steps

    # Estimator interface

    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = self.check_memory(self.memory)

        fit_transform_one_cached = memory.cache(self._fit_transform_one)

        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if (transformer is None or transformer == 'passthrough'):
                with self._print_elapsed_time('Pipeline',
                                         self._log_message(step_idx)):
                    continue

            if hasattr(memory, 'location'):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = self.clone(transformer)
            elif hasattr(memory, 'cachedir'):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = self.clone(transformer)
            else:
                cloned_transformer = self.clone(transformer)
            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer, X, y, None,
                message_clsname='Pipeline',
                message=self._log_message(step_idx),
                **fit_params_steps[name])
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X
    
    def _fit_transform_one(self, transformer,
                        X,
                        y,
                        weight,
                        message_clsname='',
                        message=None,
                        **fit_params):
        """
        Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
        with the fitted transformer. If ``weight`` is not ``None``, the result will
        be multiplied by ``weight``.
        """
        with self._print_elapsed_time(message_clsname, message):
            if hasattr(transformer, 'fit_transform'):
                res = transformer.fit_transform(X, y, **fit_params)
            else:
                res = transformer.fit(X, y, **fit_params).transform(X)

        if weight is None:
            return res, transformer
        return res * weight, transformer

    # from: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/validation.py#L414
    def check_memory(self, memory):
        """Check that ``memory`` is joblib.Memory-like.

        joblib.Memory-like means that ``memory`` can be converted into a
        joblib.Memory instance (typically a str denoting the ``location``)
        or has the same interface (has a ``cache`` method).

        Parameters
        ----------
        memory : None, str or object with the joblib.Memory interface
            - If string, the location where to create the `joblib.Memory` interface.
            - If None, no caching is done and the Memory object is completely transparent.

        Returns
        -------
        memory : object with the joblib.Memory interface
            A correct joblib.Memory object.

        Raises
        ------
        ValueError
            If ``memory`` is not joblib.Memory-like.

        Examples
        --------
        >>> from sklearn.utils.validation import check_memory
        >>> check_memory("caching_dir")
        Memory(location=caching_dir/joblib)
        """
        if memory is None or isinstance(memory, str):
            memory = joblib.Memory(location=memory, verbose=0)
        elif not hasattr(memory, "cache"):
            raise ValueError(
                "'memory' should be None, a string or have the same"
                " interface as joblib.Memory."
                " Got memory='{}' instead.".format(memory)
            )
        return memory

    # from: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_user_interface.py#L36
    def _message_with_time(self, source, message, time):
        """Create one line message for logging purposes.

        Parameters
        ----------
        source : str
            String indicating the source or the reference of the message.

        message : str
            Short message.

        time : int
            Time in seconds.
        """
        start_message = "[%s] " % source

        # adapted from joblib.logger.short_format_time without the Windows -.1s
        # adjustment
        if time > 60:
            time_str = "%4.1fmin" % (time / 60)
        else:
            time_str = " %5.1fs" % time
        end_message = " %s, total=%s" % (message, time_str)
        dots_len = 70 - len(start_message) - len(end_message)
        return "%s%s%s" % (start_message, dots_len * ".", end_message)  

    def _print_elapsed_time(self, source, message=None):
        """Log elapsed time to stdout when the context is exited.

        Parameters
        ----------
        source : str
            String indicating the source or the reference of the message.

        message : str, default=None
            Short message. If None, nothing will be printed.

        Returns
        -------
        context_manager
            Prints elapsed time upon exit if verbose.
        """
        if message is None:
            yield
        else:
            start = timeit.default_timer()
            yield
            print(self._message_with_time(source, message, timeit.default_timer() - start))

    def clone(self, estimator, *, safe=True):
        """Construct a new unfitted estimator with the same parameters.

        Clone does a deep copy of the model in an estimator
        without actually copying attached data. It returns a new estimator
        with the same parameters that has not been fitted on any data.

        .. versionchanged:: 1.3
            Delegates to `estimator.__sklearn_clone__` if the method exists.

        Parameters
        ----------
        estimator : {list, tuple, set} of estimator instance or a single \
                estimator instance
            The estimator or group of estimators to be cloned.
        safe : bool, default=True
            If safe is False, clone will fall back to a deep copy on objects
            that are not estimators. Ignored if `estimator.__sklearn_clone__`
            exists.

        Returns
        -------
        estimator : object
            The deep copy of the input, an estimator if input is an estimator.

        Notes
        -----
        If the estimator's `random_state` parameter is an integer (or if the
        estimator doesn't have a `random_state` parameter), an *exact clone* is
        returned: the clone and the original estimator will give the exact same
        results. Otherwise, *statistical clone* is returned: the clone might
        return different results from the original estimator. More details can be
        found in :ref:`randomness`.

        Examples
        --------
        >>> from sklearn.base import clone
        >>> from sklearn.linear_model import LogisticRegression
        >>> X = [[-1, 0], [0, 1], [0, -1], [1, 0]]
        >>> y = [0, 0, 1, 1]
        >>> classifier = LogisticRegression().fit(X, y)
        >>> cloned_classifier = clone(classifier)
        >>> hasattr(classifier, "classes_")
        True
        >>> hasattr(cloned_classifier, "classes_")
        False
        >>> classifier is cloned_classifier
        False
        """
        if hasattr(estimator, "__sklearn_clone__") and not inspect.isclass(estimator):
            return estimator.__sklearn_clone__()
        return self._clone_parametrized(estimator, safe=safe)

    def _clone_parametrized(self, estimator, *, safe=True):
        """Default implementation of clone. See :func:`sklearn.base.clone` for details."""

        estimator_type = type(estimator)
        if estimator_type is dict:
            return {k: self.clone(v, safe=safe) for k, v in estimator.items()}
        elif estimator_type in (list, tuple, set, frozenset):
            return estimator_type([self.clone(e, safe=safe) for e in estimator])
        elif not hasattr(estimator, "get_params") or isinstance(estimator, type):
            if not safe:
                return copy.deepcopy(estimator)
            else:
                if isinstance(estimator, type):
                    raise TypeError(
                        "Cannot clone object. "
                        + "You should provide an instance of "
                        + "scikit-learn estimator instead of a class."
                    )
                else:
                    raise TypeError(
                        "Cannot clone object '%s' (type %s): "
                        "it does not seem to be a scikit-learn "
                        "estimator as it does not implement a "
                        "'get_params' method." % (repr(estimator), type(estimator))
                    )

        klass = estimator.__class__
        new_object_params = estimator.get_params(deep=False)
        for name, param in new_object_params.items():
            new_object_params[name] = self.clone(param, safe=False)

        new_object = klass(**new_object_params)
        try:
            new_object._metadata_request = copy.deepcopy(estimator._metadata_request)
        except AttributeError:
            pass

        params_set = new_object.get_params(deep=False)

        # quick sanity check of the parameters of the clone
        for name in new_object_params:
            param1 = new_object_params[name]
            param2 = params_set[name]
            if param1 is not param2:
                raise RuntimeError(
                    "Cannot clone object %s, as the constructor "
                    "either does not set or modifies parameter %s" % (estimator, name)
                )

        # _sklearn_output_config is used by `set_output` to configure the output
        # container of an estimator.
        if hasattr(estimator, "_sklearn_output_config"):
            new_object._sklearn_output_config = copy.deepcopy(
                estimator._sklearn_output_config
            )
        return new_object

