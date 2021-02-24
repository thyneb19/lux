<p align="center"><a href="#"><img width=60% alt="" src="https://github.com/lux-org/lux-resources/blob/master/readme_img/logo.png?raw=true"></a></p>
<h2 align="center">A Python API for Intelligent Visual Discovery</h2>

<p align="center">
    <a href="https://travis-ci.com/lux-org/lux">
        <img alt="Build Status" src="https://travis-ci.com/lux-org/lux.svg?branch=master" align="center">
    </a>
    <a href="https://badge.fury.io/py/lux-api">
        <img src="https://badge.fury.io/py/lux-api.svg" alt="PyPI version" height="18" align="center">
    </a>
    <a href='https://lux-api.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/lux-api/badge/?version=latest' alt='Documentation Status'  align="center"/>
    </a>
    <a href='https://lux-project.slack.com/join/shared_invite/zt-lilu4e87-TM4EDTq9HWzlDRycFsrkLg'>
        <img src='https://img.shields.io/static/v1?label=chat&logo=slack&message=Slack&color=brightgreen' alt='Slack'  align="center"/>
    </a>
    <a href='https://forms.gle/XKv3ejrshkCi3FJE6'>
        <img src='https://img.shields.io/static/v1?label=email&message=signup&color=brightgreen' alt='Mailing List'  align="center"/>
    </a>
    <a href='https://mybinder.org/v2/gh/lux-org/lux-binder/master'>
        <img src='https://img.shields.io/badge/launch-binder-579ACA.svg?logo=data:image/png;base64,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' alt='Binder'  align="center"/>
    </a>
    <a href="https://codecov.io/gh/lux-org/lux">
        <img src="https://codecov.io/gh/lux-org/lux/branch/master/graph/badge.svg" align="center" alt='CodeCov'/>
    </a>
    <a href='https://twitter.com/intent/follow?original_referer=https%3A%2F%2Fpublish.twitter.com%2F&ref_src=twsrc%5Etfw&screen_name=lux_api&tw_p=followbutton'>
        <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/lux_api?label=%40lux_api&style=social" alt='Twitter' align="center"/>
    </a>
</p>

Lux is a Python library that makes data science easier by automating aspects of the data exploration process. Lux facilitate faster experimentation with data, even when the user does not have a clear idea of what they are looking for. Visualizations are displayed via [an interactive widget](https://github.com/lux-org/lux-widget) that allow users to quickly browse through large collections of visualizations directly within their Jupyter notebooks.

Here is a [1-min video](https://www.youtube.com/watch?v=qmnYP-LmbNU) introducing Lux, and [slides](http://dorisjunglinlee.com/files/Zillow_07_2020_Slide.pdf) from a more extended talk.

Try out Lux on your own in a live Jupyter Notebook [here](https://mybinder.org/v2/gh/lux-org/lux-binder/master?urlpath=tree/demo/employee_demo.ipynb)! 

# Getting Started

To start using Lux, simply add an extra import statement along with your Pandas import.

```python    
import lux
import pandas as pd
```

Then, Lux can be used as-is, without modifying any of your existing Pandas code. Here, we use Pandas's [read_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) command to load in a [dataset of colleges](https://github.com/lux-org/lux-datasets/blob/master/data/college.csv) and their properties.

```python    
df = pd.read_csv("https://raw.githubusercontent.com/lux-org/lux-datasets/master/data/college.csv")
df
```

<img src="https://github.com/lux-org/lux-resources/blob/master/readme_img/basicDemo.gif?raw=true"
     alt="Basic recommendations in Lux"
     style="width:900px" />

Voila! Here's a set of visualizations that you can now use to explore your dataset further!

<!-- # Features
Lux provides a suite of capabilities that enables users to effortlessly discover visual insights from their data. -->

<!-- Lux guides users to potential next-steps in their exploration. -->
### Next-step recommendations based on user intent: 

In addition to dataframe visualizations at every step in the exploration, you can specify in Lux the attributes and values you're interested in. Based on this intent, Lux guides users towards potential next-steps in their exploration.

For example, we might be interested in the attributes `AverageCost` and `SATAverage`.

```python
df.intent = ["AverageCost","SATAverage"]
df
```
<img src="https://github.com/lux-org/lux-resources/blob/master/readme_img/contextRec.gif?raw=true"
     alt="Next-step Recommendations Based on User Context"
     style="width:600px" />
 
 The left-hand side of the widget shows the current visualization, i.e., the current visualization generated based on what the user is interested in. On the right, Lux generates three sets of recommendations, organized as separate tabs on the widget:

 - `Enhance` adds an additional attribute to the current selection, essentially highlighting how additional variables affect the relationship of `AverageCost` and `SATAverage`. We see that if we breakdown the relationship by `FundingModel`, there is a clear separation between public colleges (shown in red) and private colleges (in blue), with public colleges being cheaper to attend and with SAT average of lower than 1400.
    <img src="https://github.com/lux-org/lux-resources/blob/master/readme_img/enhance.png?raw=true"
     alt="Enhance Recommendations"
     style="width:600px" />
 - `Filter` adds a filter to the current selection, while keeping attributes (on the X and Y axes) fixed. These visualizations show how the relationship of  `AverageCost` and `SATAverage` changes for different subsets of data. For instance, we see that colleges that offer Bachelor's degree as its highest degree show a roughly linear trend between the two variables.
    <img src="https://github.com/lux-org/lux-resources/blob/master/readme_img/filter.png?raw=true"
     alt="Filter Recommendations"
     style="width:600px" />
 - `Generalize` removes an attribute to display a more general trend, showing the distributions of `AverageCost` and `SATAverage` on its own. From the `AverageCost` histogram, we see that many colleges with average cost of around $20000 per year, corresponding to the bulge we see in the scatterplot view.
    <img src="https://github.com/lux-org/lux-resources/blob/master/readme_img/generalize.png?raw=true"
     alt="Generalize Recommendations"
     style="width:600px" />

 See [this page](https://lux-api.readthedocs.io/en/latest/source/guide/intent.html) for more information on additional ways for specifying the intent.

### Easy programmatic access and export of visualizations: 

Now that we have found some interesting visualizations through Lux, we might be interested in digging into these visualizations a bit more or sharing it with others. We can save the visualizations generated in Lux as a [static, shareable HTML](https://lux-api.readthedocs.io/en/latest/source/guide/export.html#exporting-widget-visualizations-as-static-html) or programmatically access these visualizations further in Jupyter. Selected `Vis` objects can be translated into [Altair](http://altair-viz.github.io/) or [Vega-Lite](https://vega.github.io/vega-lite/) code, so that they can be further edited.

<img src="https://github.com/lux-org/lux-resources/blob/master/readme_img/export.gif?raw=true"
     alt="Easily exportable visualization object"
     style="width:600px" />
     
Learn more about how to save and export visualizations [here](https://lux-api.readthedocs.io/en/latest/source/guide/export.html#exporting-widget-visualizations-as-static-html).

### Quick, on-demand visualizations with the help of automatic encoding: 
We've seen how `Vis`s are automatically generated as part of the recommendations. Users can also create their own Vis via the same syntax as specifying the intent. Lux is built on the philosophy that users should always be able to visualize anything they want, without having to think about *how* the visualization should look like. Lux automatically determines the mark and channel mappings based on a set of [best practices](http://hosteddocs.ittoolbox.com/fourshowmeautomaticpresentations.pdf). The visualizations are rendered via [Altair](https://github.com/altair-viz/altair/tree/master/altair) into [Vega-Lite](https://github.com/vega/vega-lite) specifications.

```python    
from lux.vis.Vis import Vis
Vis(["Region=New England","MedianEarnings"],df)
```    

<img src="https://github.com/lux-org/lux-resources/blob/master/readme_img/view.png?raw=true"
     alt="Specified Visualization"
     width="200px" />

### Powerful language for working with collections of visualizations:

Lux provides a powerful abstraction for working with collections of visualizations based on a partially specified queries. Users can provide a list or a wildcard to iterate over combinations of filter or attribute values and quickly browse through large numbers of visualizations. The partial specification is inspired by existing work on visualization query languages, including [ZQL](https://github.com/vega/compassql) and [CompassQL](https://github.com/vega/compassql).

For example, we might be interested in looking at how the `AverageCost` distribution differs across different `Region`s.

```python    
from lux.vis.VisList import VisList
VisList(["Region=?","AverageCost"],df)
```    

<img src="https://github.com/lux-org/lux-resources/blob/master/readme_img/visList.gif?raw=true"
     alt="Example Vis List"
     style="width:600px" />


To find out more about other features in Lux, see the complete documentation on [ReadTheDocs](https://lux-api.readthedocs.io/).

# Installation 

To get started, Lux can be installed through [PyPI](https://pypi.org/project/lux-api/). 

```bash
pip install lux-api
```

If you use [conda](https://docs.conda.io/en/latest/), you can install Lux via:

```bash
conda install -c conda-forge lux-api
```

Both the PyPI and conda installation include includes the Lux Jupyter widget frontend, [lux-widget](https://pypi.org/project/lux-widget/). 

## Setup in Jupyter Notebook, VSCode

To use Lux in [Jupyter notebook](https://github.com/jupyter/notebook) or [VSCode](https://code.visualstudio.com/docs/python/jupyter-support), activate the notebook extension:

```bash
jupyter nbextension install --py luxwidget
jupyter nbextension enable --py luxwidget
```

If the installation happens correctly, you should see two `- Validating: OK` after executing the two lines above.

## Setup in Jupyter Lab

To use Lux in [Jupyter Lab](https://github.com/jupyterlab/jupyterlab), activate the lab extension:

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install luxwidget
```
Lux is only compatible with Jupyter Lab version 2.2.9 and below. Support for the recent [JupyterLab 3](https://blog.jupyter.org/jupyterlab-3-0-is-out-4f58385e25bb) will come soon. Note that JupyterLab and VSCode is supported only for lux-widget version >=0.1.2, if you have an earlier version, please upgrade to the latest version of [lux-widget](https://pypi.org/project/lux-widget/). Lux currently only works with the Chrome browser. 

If you encounter issues with the installation, please refer to [this page](https://lux-api.readthedocs.io/en/latest/source/guide/FAQ.html#troubleshooting-tips) to troubleshoot the installation. Follow [these instructions](https://lux-api.readthedocs.io/en/latest/source/getting_started/installation.html#manual-installation-dev-setup) to set up Lux for development purposes.

# Support and Resources

Lux is undergoing active development. If you are interested in using Lux, we would love to hear from you. Any feedback, suggestions, and contributions for improving Lux are welcome.

Other additional resources:

- Sign up for the early-user [mailing list](https://forms.gle/XKv3ejrshkCi3FJE6) to stay tuned for upcoming releases, updates, or user studies. 
- Visit [ReadTheDoc](https://lux-api.readthedocs.io/en/latest/) for more detailed documentation.
- Try out these hands-on [exercises](https://mybinder.org/v2/gh/lux-org/lux-binder/master?urlpath=tree/exercise) or [tutorials](https://mybinder.org/v2/gh/lux-org/lux-binder/master?urlpath=tree/tutorial) on [Binder](https://mybinder.org/v2/gh/lux-org/lux-binder/master). Or clone and run [lux-binder](https://github.com/lux-org/lux-binder) locally.
- Join our community [Slack](https://lux-project.slack.com/join/shared_invite/zt-lilu4e87-TM4EDTq9HWzlDRycFsrkLg) to discuss and ask questions.
- Report any bugs, issues, or requests through [Github Issues](https://github.com/lux-org/lux/issues). 
