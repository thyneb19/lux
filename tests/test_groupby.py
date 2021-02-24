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

from .context import lux
import pytest
import pandas as pd


def test_agg(global_var):
    df = pytest.car_df
    df._repr_html_()
    new_df = df[["Horsepower", "Brand"]].groupby("Brand").agg(sum)
    new_df._repr_html_()
    assert new_df.history[0].name == "groupby"


def test_shortcut_agg(global_var):
    df = pytest.car_df
    df._repr_html_()
    new_df = df[["MilesPerGal", "Brand"]].groupby("Brand").sum()
    new_df._repr_html_()
    assert new_df.history[0].name == "groupby"


def test_agg_mean(global_var):
    df = pytest.car_df
    df._repr_html_()
    new_df = df.groupby("Origin").mean()
    new_df._repr_html_()
    assert new_df.history[0].name == "groupby"


def test_agg_size(global_var):
    df = pytest.car_df
    df._repr_html_()
    new_df = df.groupby("Brand").size().to_frame()
    new_df._repr_html_()
    assert new_df.history[0].name == "groupby"


def test_filter(global_var):
    df = pytest.car_df
    df._repr_html_()
    new_df = df.groupby("Origin").filter(lambda x: x["Weight"].mean() > 3000)
    new_df._repr_html_()
    assert new_df.history[0].name == "groupby"
