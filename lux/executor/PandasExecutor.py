#  Copyright 2019-2020 The Lux Authors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pandas as pd
from lux.vis.VisList import VisList
from lux.vis.Vis import Vis
from lux.core.frame import LuxDataFrame
from lux.executor.Executor import Executor
from lux.utils import utils
from lux.utils.date_utils import is_datetime_series
from lux.utils.utils import check_import_lux_widget, check_if_id_like
import warnings
import lux


class PandasExecutor(Executor):
    """
    Given a Vis objects with complete specifications, fetch and process data using Pandas dataframe operations.
    """

    def __init__(self):
        self.name = "PandasExecutor"
        warnings.formatwarning = lux.warning_format

    def __repr__(self):
        return f"<PandasExecutor>"

    @staticmethod
    def execute_sampling(ldf: LuxDataFrame):
        # General Sampling for entire dataframe
        SAMPLE_FLAG = lux.config.sampling
        SAMPLE_START = lux.config.sampling_start
        SAMPLE_CAP = lux.config.sampling_cap
        SAMPLE_FRAC = 0.75

        if SAMPLE_FLAG and len(ldf) > SAMPLE_CAP:
            if ldf._sampled is None:  # memoize unfiltered sample df
                ldf._sampled = ldf.sample(n=SAMPLE_CAP, random_state=1)
            ldf._message.add_unique(
                f"Large dataframe detected: Lux is only visualizing a random sample capped at {SAMPLE_CAP} rows.",
                priority=99,
            )
        elif SAMPLE_FLAG and len(ldf) > SAMPLE_START:
            if ldf._sampled is None:  # memoize unfiltered sample df
                ldf._sampled = ldf.sample(frac=SAMPLE_FRAC, random_state=1)
            ldf._message.add_unique(
                f"Large dataframe detected: Lux is only visualizing a random sample of {len(ldf._sampled)} rows.",
                priority=99,
            )
        else:
            ldf._sampled = ldf

    @staticmethod
    def execute(vislist: VisList, ldf: LuxDataFrame):
        """
        Given a VisList, fetch the data required to render the vis.
        1) Apply filters
        2) Retrieve relevant attribute
        3) Perform vis-related processing (aggregation, binning)
        4) return a DataFrame with relevant results

        Parameters
        ----------
        vislist: list[lux.Vis]
            vis list that contains lux.Vis objects for visualization.
        ldf : lux.core.frame
            LuxDataFrame with specified intent.

        Returns
        -------
        None
        """
        PandasExecutor.execute_sampling(ldf)
        for vis in vislist:
            # The vis data starts off being original or sampled dataframe
            vis._vis_data = ldf._sampled
            filter_executed = PandasExecutor.execute_filter(vis)
            # Select relevant data based on attribute information
            attributes = set([])
            for clause in vis._inferred_intent:
                if clause.attribute != "Record":
                    attributes.add(clause.attribute)
            # TODO: Add some type of cap size on Nrows ?
            vis._vis_data = vis.data[list(attributes)]

            if vis.mark == "bar" or vis.mark == "line":
                PandasExecutor.execute_aggregate(vis, isFiltered=filter_executed)
            elif vis.mark == "histogram":
                PandasExecutor.execute_binning(vis)
            elif vis.mark == "scatter":
                HBIN_START = 5000
                if lux.config.heatmap and len(ldf) > HBIN_START:
                    vis._postbin = True
                    ldf._message.add_unique(
                        f"Large scatterplots detected: Lux is automatically binning scatterplots to heatmaps.",
                        priority=98,
                    )
                    # vis._mark = "heatmap"
                    # PandasExecutor.execute_2D_binning(vis) # Lazy Evaluation (Early pruning based on interestingness)

    @staticmethod
    def execute_aggregate(vis: Vis, isFiltered=True):
        """
        Aggregate data points on an axis for bar or line charts

        Parameters
        ----------
        vis: lux.Vis
            lux.Vis object that represents a visualization
        ldf : lux.core.frame
            LuxDataFrame with specified intent.

        Returns
        -------
        None
        """
        import numpy as np

        x_attr = vis.get_attr_by_channel("x")[0]
        y_attr = vis.get_attr_by_channel("y")[0]
        has_color = False
        groupby_attr = ""
        measure_attr = ""
        if x_attr.aggregation is None or y_attr.aggregation is None:
            return
        if y_attr.aggregation != "":
            groupby_attr = x_attr
            measure_attr = y_attr
            agg_func = y_attr.aggregation
        if x_attr.aggregation != "":
            groupby_attr = y_attr
            measure_attr = x_attr
            agg_func = x_attr.aggregation
        if groupby_attr.attribute in vis.data.unique_values.keys():
            attr_unique_vals = vis.data.unique_values[groupby_attr.attribute]
        # checks if color is specified in the Vis
        if len(vis.get_attr_by_channel("color")) == 1:
            color_attr = vis.get_attr_by_channel("color")[0]
            color_attr_vals = vis.data.unique_values[color_attr.attribute]
            color_cardinality = len(color_attr_vals)
            # NOTE: might want to have a check somewhere to not use categorical variables with greater than some number of categories as a Color variable----------------
            has_color = True
        else:
            color_cardinality = 1
        if measure_attr != "":
            if measure_attr.attribute == "Record":
                # need to get the index name so that we can rename the index column to "Record"
                # if there is no index, default to "index"
                index_name = vis.data.index.name
                if index_name == None:
                    index_name = "index"

                vis._vis_data = vis.data.reset_index()
                # if color is specified, need to group by groupby_attr and color_attr
                if has_color:
                    vis._vis_data = (
                        vis.data.groupby(
                            [groupby_attr.attribute, color_attr.attribute], dropna=False, history=False
                        )
                        .count()
                        .reset_index()
                        .rename(columns={index_name: "Record"})
                    )
                    vis._vis_data = vis.data[[groupby_attr.attribute, color_attr.attribute, "Record"]]
                else:
                    vis._vis_data = (
                        vis.data.groupby(groupby_attr.attribute, dropna=False, history=False)
                        .count()
                        .reset_index()
                        .rename(columns={index_name: "Record"})
                    )
                    vis._vis_data = vis.data[[groupby_attr.attribute, "Record"]]
            else:
                # if color is specified, need to group by groupby_attr and color_attr
                if has_color:
                    groupby_result = vis.data.groupby(
                        [groupby_attr.attribute, color_attr.attribute], dropna=False, history=False
                    )
                else:
                    groupby_result = vis.data.groupby(
                        groupby_attr.attribute, dropna=False, history=False
                    )
                groupby_result = groupby_result.agg(agg_func)
                intermediate = groupby_result.reset_index()
                vis._vis_data = intermediate.__finalize__(vis.data)
            result_vals = list(vis.data[groupby_attr.attribute])
            # create existing group by attribute combinations if color is specified
            # this is needed to check what combinations of group_by_attr and color_attr values have a non-zero number of elements in them
            if has_color:
                res_color_combi_vals = []
                result_color_vals = list(vis.data[color_attr.attribute])
                for i in range(0, len(result_vals)):
                    res_color_combi_vals.append([result_vals[i], result_color_vals[i]])
            # For filtered aggregation that have missing groupby-attribute values, set these aggregated value as 0, since no datapoints
            if isFiltered or has_color and attr_unique_vals:
                N_unique_vals = len(attr_unique_vals)
                if len(result_vals) != N_unique_vals * color_cardinality:
                    columns = vis.data.columns
                    if has_color:
                        df = pd.DataFrame(
                            {
                                columns[0]: attr_unique_vals * color_cardinality,
                                columns[1]: pd.Series(color_attr_vals).repeat(N_unique_vals),
                            }
                        )
                        vis._vis_data = vis.data.merge(
                            df,
                            on=[columns[0], columns[1]],
                            how="right",
                            suffixes=["", "_right"],
                        )
                        for col in columns[2:]:
                            vis.data[col] = vis.data[col].fillna(0)  # Triggers __setitem__
                        assert len(list(vis.data[groupby_attr.attribute])) == N_unique_vals * len(
                            color_attr_vals
                        ), f"Aggregated data missing values compared to original range of values of `{groupby_attr.attribute, color_attr.attribute}`."

                        # Keep only the three relevant columns not the *_right columns resulting from merge
                        vis._vis_data = vis.data.iloc[:, :3]

                    else:
                        df = pd.DataFrame({columns[0]: attr_unique_vals})

                        vis._vis_data = vis.data.merge(
                            df, on=columns[0], how="right", suffixes=["", "_right"]
                        )

                        for col in columns[1:]:
                            vis.data[col] = vis.data[col].fillna(0)
                        assert (
                            len(list(vis.data[groupby_attr.attribute])) == N_unique_vals
                        ), f"Aggregated data missing values compared to original range of values of `{groupby_attr.attribute}`."

            vis._vis_data = vis._vis_data.dropna(subset=[measure_attr.attribute])
            try:
                vis._vis_data = vis._vis_data.sort_values(by=groupby_attr.attribute, ascending=True)
            except TypeError:
                warnings.warn(
                    f"\nLux detects that the attribute '{groupby_attr.attribute}' maybe contain mixed type."
                    + f"\nTo visualize this attribute, you may want to convert the '{groupby_attr.attribute}' into a uniform type as follows:"
                    + f"\n\tdf['{groupby_attr.attribute}'] = df['{groupby_attr.attribute}'].astype(str)"
                )
                vis._vis_data[groupby_attr.attribute] = vis._vis_data[groupby_attr.attribute].astype(str)
                vis._vis_data = vis._vis_data.sort_values(by=groupby_attr.attribute, ascending=True)
            vis._vis_data = vis._vis_data.reset_index()
            vis._vis_data = vis._vis_data.drop(columns="index")

    @staticmethod
    def execute_binning(vis: Vis):
        """
        Binning of data points for generating histograms

        Parameters
        ----------
        vis: lux.Vis
            lux.Vis object that represents a visualization
        ldf : lux.core.frame
            LuxDataFrame with specified intent.

        Returns
        -------
        None
        """
        import numpy as np

        bin_attribute = list(filter(lambda x: x.bin_size != 0, vis._inferred_intent))[0]
        bin_attr = bin_attribute.attribute
        if not np.isnan(vis.data[bin_attr]).all():
            # np.histogram breaks if array contain NaN
            series = vis.data[bin_attr].dropna()
            # TODO:binning runs for name attribte. Name attribute has datatype quantitative which is wrong.
            counts, bin_edges = np.histogram(series, bins=bin_attribute.bin_size)
            # bin_edges of size N+1, so need to compute bin_center as the bin location
            bin_center = np.mean(np.vstack([bin_edges[0:-1], bin_edges[1:]]), axis=0)
            # TODO: Should vis.data be a LuxDataFrame or a Pandas DataFrame?
            binned_result = np.array([bin_center, counts]).T
            vis._vis_data = pd.DataFrame(binned_result, columns=[bin_attr, "Number of Records"])

    @staticmethod
    def execute_filter(vis: Vis):
        assert (
            vis.data is not None
        ), "execute_filter assumes input vis.data is populated (if not, populate with LuxDataFrame values)"
        filters = utils.get_filter_specs(vis._inferred_intent)

        if filters:
            # TODO: Need to handle OR logic
            for filter in filters:
                vis._vis_data = PandasExecutor.apply_filter(
                    vis.data, filter.attribute, filter.filter_op, filter.value
                )
            return True
        else:
            return False

    @staticmethod
    def apply_filter(df: pd.DataFrame, attribute: str, op: str, val: object) -> pd.DataFrame:
        """
        Helper function for applying filter to a dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe to filter on
        attribute : str
            Filter attribute
        op : str
            Filter operation, '=', '<', '>', '<=', '>=', '!='
        val : object
            Filter value

        Returns
        -------
        df: pandas.DataFrame
            Dataframe resulting from the filter operation
        """
        # Handling NaN filter values
        if utils.like_nan(val):
            if op != "=" and op != "!=":
                warnings.warn("Filter on NaN must be used with equality operations (i.e., `=` or `!=`)")
            else:
                if op == "=":
                    return df[df[attribute].isna()]
                elif op == "!=":
                    return df[~df[attribute].isna()]
        # Applying filter in regular, non-NaN cases
        if op == "=":
            return df[df[attribute] == val]
        elif op == "<":
            return df[df[attribute] < val]
        elif op == ">":
            return df[df[attribute] > val]
        elif op == "<=":
            return df[df[attribute] <= val]
        elif op == ">=":
            return df[df[attribute] >= val]
        elif op == "!=":
            return df[df[attribute] != val]
        return df

    @staticmethod
    def execute_2D_binning(vis: Vis):
        pd.reset_option("mode.chained_assignment")
        with pd.option_context("mode.chained_assignment", None):
            x_attr = vis.get_attr_by_channel("x")[0].attribute
            y_attr = vis.get_attr_by_channel("y")[0].attribute

            vis._vis_data["xBin"] = pd.cut(vis._vis_data[x_attr], bins=lux.config.heatmap_bin_size)
            vis._vis_data["yBin"] = pd.cut(vis._vis_data[y_attr], bins=lux.config.heatmap_bin_size)

            color_attr = vis.get_attr_by_channel("color")
            if len(color_attr) > 0:
                color_attr = color_attr[0]
                groups = vis._vis_data.groupby(["xBin", "yBin"], history=False)[color_attr.attribute]
                if color_attr.data_type == "nominal":
                    # Compute mode and count. Mode aggregates each cell by taking the majority vote for the category variable. In cases where there is ties across categories, pick the first item (.iat[0])
                    result = groups.agg(
                        [
                            ("count", "count"),
                            (color_attr.attribute, lambda x: pd.Series.mode(x).iat[0]),
                        ]
                    ).reset_index()
                elif color_attr.data_type == "quantitative":
                    # Compute the average of all values in the bin
                    result = groups.agg(
                        [("count", "count"), (color_attr.attribute, "mean")]
                    ).reset_index()
                result = result.dropna()
            else:
                groups = vis._vis_data.groupby(["xBin", "yBin"], history=False)[x_attr]
                result = groups.count().reset_index(name=x_attr)
                result = result.rename(columns={x_attr: "count"})
                result = result[result["count"] != 0]

            # convert type to facilitate weighted correlation interestingess calculation
            result["xBinStart"] = result["xBin"].apply(lambda x: x.left).astype("float")
            result["xBinEnd"] = result["xBin"].apply(lambda x: x.right)

            result["yBinStart"] = result["yBin"].apply(lambda x: x.left).astype("float")
            result["yBinEnd"] = result["yBin"].apply(lambda x: x.right)

            vis._vis_data = result.drop(columns=["xBin", "yBin"])

    #######################################################
    ############ Metadata: data type, model #############
    #######################################################
    def compute_dataset_metadata(self, ldf: LuxDataFrame):
        ldf._data_type = {}
        self.compute_data_type(ldf)

    def compute_data_type(self, ldf: LuxDataFrame):
        from pandas.api.types import is_datetime64_any_dtype as is_datetime

        for attr in list(ldf.columns):
            if attr in ldf._type_override:
                ldf._data_type[attr] = ldf._type_override[attr]
            else:
                temporal_var_list = ["month", "year", "day", "date", "time", "weekday"]
                if is_datetime(ldf[attr]):
                    ldf._data_type[attr] = "temporal"
                elif self._is_datetime_string(ldf[attr]):
                    ldf._data_type[attr] = "temporal"
                elif isinstance(attr, pd._libs.tslibs.timestamps.Timestamp):
                    ldf._data_type[attr] = "temporal"
                elif str(attr).lower() in temporal_var_list:
                    ldf._data_type[attr] = "temporal"
                elif self._is_datetime_number(ldf[attr]):
                    ldf._data_type[attr] = "temporal"
                elif pd.api.types.is_float_dtype(ldf.dtypes[attr]):
                    # int columns gets coerced into floats if contain NaN
                    convertible2int = pd.api.types.is_integer_dtype(ldf[attr].convert_dtypes())
                    if (
                        convertible2int
                        and ldf.cardinality[attr] != len(ldf)
                        and ldf.cardinality[attr] < 20
                    ):
                        ldf._data_type[attr] = "nominal"
                    else:
                        ldf._data_type[attr] = "quantitative"
                elif pd.api.types.is_integer_dtype(ldf.dtypes[attr]):
                    # See if integer value is quantitative or nominal by checking if the ratio of cardinality/data size is less than 0.4 and if there are less than 10 unique values
                    if ldf.pre_aggregated:
                        if ldf.cardinality[attr] == len(ldf):
                            ldf._data_type[attr] = "nominal"
                    if ldf.cardinality[attr] / len(ldf) < 0.4 and ldf.cardinality[attr] < 20:
                        ldf._data_type[attr] = "nominal"
                    else:
                        ldf._data_type[attr] = "quantitative"
                    if check_if_id_like(ldf, attr):
                        ldf._data_type[attr] = "id"
                # Eliminate this clause because a single NaN value can cause the dtype to be object
                elif pd.api.types.is_string_dtype(ldf.dtypes[attr]):
                    if check_if_id_like(ldf, attr):
                        ldf._data_type[attr] = "id"
                    else:
                        ldf._data_type[attr] = "nominal"
                # check if attribute is any type of datetime dtype
                elif is_datetime_series(ldf.dtypes[attr]):
                    ldf._data_type[attr] = "temporal"
                else:
                    ldf._data_type[attr] = "nominal"
        if not pd.api.types.is_integer_dtype(ldf.index) and ldf.index.name:
            ldf._data_type[ldf.index.name] = "nominal"

        non_datetime_attrs = []
        for attr in ldf.columns:
            if ldf._data_type[attr] == "temporal" and not is_datetime(ldf[attr]):
                non_datetime_attrs.append(attr)
        warn_msg = ""
        if len(non_datetime_attrs) == 1:
            warn_msg += f"\nLux detects that the attribute '{non_datetime_attrs[0]}' may be temporal.\n"
        elif len(non_datetime_attrs) > 1:
            warn_msg += f"\nLux detects that attributes {non_datetime_attrs} may be temporal.\n"
        if len(non_datetime_attrs) > 0:
            warn_msg += "To display visualizations for these attributes accurately, please convert temporal attributes to Pandas Datetime objects using the pd.to_datetime function and provide a 'format' parameter to specify the datetime format of the attribute.\nFor example, you can convert a year-only attribute (e.g., 1998, 1971, 1982) to Datetime type by specifying the `format` as '%Y'.\n\nHere is a starter template that you can use for converting the temporal fields:\n"
            for attr in non_datetime_attrs:
                warn_msg += f"\tdf['{attr}'] = pd.to_datetime(df['{attr}'], format='<replace-with-datetime-format>')\n"
            warn_msg += "\nSee more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html"
            warn_msg += f"\nIf {attr} is not a temporal attribute, please use override Lux's automatically detected type:"
            warn_msg += f"\n\tdf.set_data_type({{'{attr}':'quantitative'}})"
            warnings.warn(warn_msg, stacklevel=2)

    @staticmethod
    def _is_datetime_string(series):
        if series.dtype == object:
            not_numeric = False
            try:
                pd.to_numeric(series)
            except Exception as e:
                not_numeric = True

            datetime_col = None
            if not_numeric:
                try:
                    datetime_col = pd.to_datetime(series)
                except Exception as e:
                    return False
            if datetime_col is not None:
                return True
        return False

    @staticmethod
    def _is_datetime_number(series):
        if series.dtype == int:
            try:
                temp = series.astype(str)
                pd.to_datetime(temp)
                return True
            except Exception:
                return False
        return False

    def compute_stats(self, ldf: LuxDataFrame):
        # precompute statistics
        ldf.unique_values = {}
        ldf._min_max = {}
        ldf.cardinality = {}

        for attribute in ldf.columns:

            if isinstance(attribute, pd._libs.tslibs.timestamps.Timestamp):
                # If timestamp, make the dictionary keys the _repr_ (e.g., TimeStamp('2020-04-05 00.000')--> '2020-04-05')
                attribute_repr = str(attribute._date_repr)
            else:
                attribute_repr = attribute

            ldf.unique_values[attribute_repr] = list(ldf[attribute_repr].unique())
            ldf.cardinality[attribute_repr] = len(ldf.unique_values[attribute_repr])

            if pd.api.types.is_float_dtype(ldf.dtypes[attribute]) or pd.api.types.is_integer_dtype(
                ldf.dtypes[attribute]
            ):
                ldf._min_max[attribute_repr] = (
                    ldf[attribute].min(),
                    ldf[attribute].max(),
                )

        if not pd.api.types.is_integer_dtype(ldf.index):
            index_column_name = ldf.index.name
            ldf.unique_values[index_column_name] = list(ldf.index)
            ldf.cardinality[index_column_name] = len(ldf.index)
