# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Golf Putting Models with PyMC

    This notebook demonstrates four different probabilistic models for golf putting data, using PyMC.

    - **Logistic Regression**: Simple probability model
    - **Angle Model**: Considers the angle to the hole
    - **Distance & Angle**: Adds distance tolerance
    - **Distance, Angle & Dispersion**: Adds extra dispersion

    This notebook is adapted from the notebook [here](https://www.pymc.io/projects/examples/en/latest/case_studies/putting_workflow.html).
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
    import pytensor
    from pytensor import function
    from pytensor.compile import Function
    from pytensor.tensor.variable import TensorVariable
    import scipy.stats as st
    import xarray as xr
    import io
    from pymc.model.fgraph import clone_model

    az.style.use("arviz-darkgrid")

    # Constants for golf putting geometry
    BALL_RADIUS = (1.68 / 2) / 12  # feet
    CUP_RADIUS = (4.25 / 2) / 12  # feet
    OVERSHOT = 1.0  # feet
    DISTANCE_TOLERANCE = 3.0  # feet

    # Utility for standard normal CDF
    def phi(x):
        return 0.5 + 0.5 * pt.erf(x / pt.sqrt(2.0))
    return (
        Axes,
        BALL_RADIUS,
        CUP_RADIUS,
        DISTANCE_TOLERANCE,
        Function,
        OVERSHOT,
        TensorVariable,
        az,
        clone_model,
        function,
        io,
        np,
        pd,
        phi,
        plt,
        pm,
        pt,
        pytensor,
        st,
        xr,
    )


@app.cell
def _(io, pd):
    # Berry (1996) golf putting data
    golf_data_lines = """distance tries successes
    2 1443 1346
    3 694 577
    4 455 337
    5 353 208
    6 272 149
    7 256 136
    8 240 111
    9 217 69
    10 200 67
    11 237 75
    12 202 52
    13 192 46
    14 174 54
    15 167 28
    16 201 27
    17 195 31
    18 191 33
    19 147 20
    20 152 24"""

    def read_golf_data(golf_data: str) -> pd.DataFrame:
        return pd.read_csv(
            io.StringIO(golf_data),
            sep=" ",
            dtype={"distance": "float"},
        )

    golf_data = read_golf_data(golf_data_lines)
    return golf_data, read_golf_data


@app.cell
def _(read_golf_data):
    # Broadie (2018) golf putting data
    new_golf_data_lines = """distance tries successes
    0.28 45198 45183
    0.97 183020 182899
    1.93 169503 168594
    2.92 113094 108953
    3.93 73855 64740
    4.94 53659 41106
    5.94 42991 28205
    6.95 37050 21334
    7.95 33275 16615
    8.95 30836 13503
    9.95 28637 11060
    10.95 26239 9032
    11.95 24636 7687
    12.95 22876 6432
    14.43 41267 9813
    16.43 35712 7196
    18.44 31573 5290
    20.44 28280 4086
    21.95 13238 1642
    24.39 46570 4767
    28.40 38422 2980
    32.39 31641 1996
    36.39 25604 1327
    40.37 20366 834
    44.38 15977 559
    48.37 11770 311
    52.36 8708 231
    57.25 8878 204
    63.23 5492 103
    69.18 3087 35
    75.19 1742 24"""

    new_golf_data = read_golf_data(new_golf_data_lines)
    return (new_golf_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Data
    """)
    return


@app.cell
def _(mo):
    # UI for dataset and model selection
    dataset_selector = mo.ui.dropdown(
        options={"Berry (1996)": "berry", "Broadie (2018)": "broadie"},
        value="Berry (1996)",
        label="Select Dataset",
    )

    dataset_selector
    return (dataset_selector,)


@app.cell
def _(dataset_selector, mo):
    mo.md(f"""
    ```python
    data: pd.DataFrame = read_{dataset_selector.value}_dataset()
    ```
    """)
    return


@app.cell
def _(dataset_selector, golf_data, new_golf_data):
    # Select data based on UI
    data = golf_data if dataset_selector.value == "berry" else new_golf_data
    return (data,)


@app.cell
def _(data):
    data
    return


@app.cell
def _(data, dataset_selector, mo):
    mo.md(f"""
    The {dataset_selector.value} dataset has {data["tries"].sum().item():,} shots across {len(data)} different distances ranging from {data["distance"].min()} ft to {data["distance"].max()} ft.

    The marginal success rate is {data["successes"].sum() / data["tries"].sum():.2%}
    """)
    return


@app.cell
def _(data, plot_golf_data):
    plot_golf_data(data)
    return


@app.cell
def _(BALL_RADIUS, CUP_RADIUS, DISTANCE_TOLERANCE, OVERSHOT, pd, phi, pm, pt):
    # Model definitions
    def initialize_model(golf_data: pd.DataFrame) -> pm.Model:
        coords = {"dist": golf_data["distance"].values}
        with pm.Model(coords=coords) as model:
            pm.Data("distance", golf_data["distance"], dims="dist")
            pm.Data("tries", golf_data["tries"], dims="dist")
            pm.Data("successes", golf_data["successes"], dims="dist")
        return model

    def define_logit_model(golf_data: pd.DataFrame) -> pm.Model:
        with initialize_model(golf_data) as model:
            a = pm.Normal("intercept")
            b = pm.Normal("slope")
            p = pm.Deterministic(
                "p_make",
                pm.math.invlogit(a * model["distance"] + b),
                dims="dist",
            )
            pm.Binomial(
                "success",
                n=model["tries"],
                p=p,
                observed=model["successes"],
                dims="dist",
            )
        return model

    def define_angle_model(golf_data: pd.DataFrame) -> pm.Model:
        with initialize_model(golf_data) as model:
            variance_of_shot = pm.HalfNormal("variance_of_shot")
            p_goes_in = pm.Deterministic(
                "p_make",
                2
                * phi(
                    pt.arcsin((CUP_RADIUS - BALL_RADIUS) / model["distance"])
                    / variance_of_shot
                )
                - 1,
                dims="dist",
            )
            pm.Binomial(
                "success",
                n=model["tries"],
                p=p_goes_in,
                observed=model["successes"],
                dims="dist",
            )
        return model

    def define_distance_angle_model(golf_data: pd.DataFrame) -> pm.Model:
        with initialize_model(golf_data) as model:
            variance_of_shot = pm.HalfNormal("variance_of_shot")
            variance_of_distance = pm.HalfNormal("variance_of_distance")
            p_good_angle = (
                2
                * phi(
                    pt.arcsin((CUP_RADIUS - BALL_RADIUS) / model["distance"])
                    / variance_of_shot
                )
                - 1
            )
            p_good_distance = phi(
                (DISTANCE_TOLERANCE - OVERSHOT)
                / ((model["distance"] + OVERSHOT) * variance_of_distance)
            ) - phi(-OVERSHOT / ((model["distance"] + OVERSHOT) * variance_of_distance))
            p = pm.Deterministic(
                "p_make",
                p_good_angle * p_good_distance,
                dims="dist",
            )
            pm.Binomial(
                "success",
                n=model["tries"],
                p=p,
                observed=model["successes"],
                dims="dist",
            )
        return model

    def define_disp_distance_angle_model(golf_data: pd.DataFrame) -> pm.Model:
        with initialize_model(golf_data) as model:
            variance_of_shot = pm.HalfNormal("variance_of_shot")
            variance_of_distance = pm.HalfNormal("variance_of_distance")
            dispersion = pm.HalfNormal("dispersion")
            p_good_angle = (
                2
                * phi(
                    pt.arcsin((CUP_RADIUS - BALL_RADIUS) / model["distance"])
                    / variance_of_shot
                )
                - 1
            )
            p_good_distance = phi(
                (DISTANCE_TOLERANCE - OVERSHOT)
                / ((model["distance"] + OVERSHOT) * variance_of_distance)
            ) - phi(-OVERSHOT / ((model["distance"] + OVERSHOT) * variance_of_distance))
            p = pm.Deterministic(
                "p_make",
                p_good_angle * p_good_distance,
                dims="dist",
            )
            pm.Normal(
                "p_success",
                mu=p,
                sigma=pt.sqrt(((p * (1 - p)) / model["tries"]) + dispersion**2),
                observed=model["successes"] / model["tries"],
                dims="dist",
            )
        return model
    return (
        define_angle_model,
        define_disp_distance_angle_model,
        define_distance_angle_model,
        define_logit_model,
    )


@app.cell
def _(mo):
    model_selector = mo.ui.dropdown(
        options={
            "Logistic Regression": "logit",
            "Angle Model": "angle",
            "Distance & Angle": "distance_angle",
            "Distance, Angle & Dispersion": "disp_distance_angle",
        },
        value="Logistic Regression",
        label="Select Model",
    )
    return (model_selector,)


@app.cell(hide_code=True)
def _(mo, model_selector):
    mo.md(rf"""
    ## Displaying PyMC Models

    The `pm.Model` object implements the `_display_` method which makes viewing your model in marimo very easy!

    ```python
    # In general
    coords = {{ ... }}
    with pm.Model(coords=coords) as model:
        ...

    # In our case
    model = define_{model_selector.value}_model(data)

    model
    ```
    """)
    return


@app.cell
def _(model_selector):
    model_selector
    return


@app.cell
def _(
    data,
    define_angle_model,
    define_disp_distance_angle_model,
    define_distance_angle_model,
    define_logit_model,
    model_selector,
):
    model = {
        "logit": define_logit_model,
        "angle": define_angle_model,
        "distance_angle": define_distance_angle_model,
        "disp_distance_angle": define_disp_distance_angle_model,
    }[model_selector.value](data)

    model
    return (model,)


@app.cell(hide_code=True)
def _(mo, parameter_ui):
    fn_inputs = ", ".join(f"{key}={value}" for key, value in parameter_ui.value.items())
    mo.md(rf"""
    ## Model

    ```python
    import pytensor

    inputs = [v for v in pytensor.graph.ancestors([model["p_make"]])  if v in model.free_RVs]
    fn = pytensor.function([inputs], model["p_make"])

    fn({fn_inputs})
    ```
    """)
    return


@app.cell
def _(Function, function, pm, pytensor):
    def free_RVs_into_p_make(model: pm.Model):
        return [
            v
            for v in pytensor.graph.ancestors([model["p_make"]])
            if v in model.free_RVs
        ]

    def compile_p_make_function(model: pm.Model) -> Function:
        inputs = free_RVs_into_p_make(model)
        return function(inputs, model["p_make"], mode="FAST_COMPILE")
    return compile_p_make_function, free_RVs_into_p_make


@app.cell
def _(TensorVariable, free_RVs_into_p_make, mo, pm):
    def get_distribution_name(var: TensorVariable) -> str:
        return var.owner.op.__class__.__name__

    def get_parameter_ui(name: str):
        kwargs = {"show_value": True}

        if name == "HalfNormalRV":
            return mo.ui.slider(start=0.01, value=0.1, step=0.01, stop=0.25, **kwargs)

        return mo.ui.slider(start=-10, value=0.0, stop=10, step=0.01, **kwargs)

    def create_marimo_model_inputs(model: pm.Model):
        inputs = free_RVs_into_p_make(model)
        names = [get_distribution_name(var) for var in inputs]

        return {var.name: get_parameter_ui(name) for var, name in zip(inputs, names)}
    return (create_marimo_model_inputs,)


@app.cell
def _(create_marimo_model_inputs, mo, model):
    parameter_ui = mo.ui.dictionary(create_marimo_model_inputs(model))
    return (parameter_ui,)


@app.cell
def _(compile_p_make_function, model):
    fn = compile_p_make_function(model)
    return (fn,)


@app.cell
def _(data, fn, mo, parameter_ui, plot_golf_data):
    _ax = plot_golf_data(data)
    _ax.plot(
        data["distance"],
        fn(**parameter_ui.value),
        linestyle="--",
        color="grey",
        label="Guessed Parameters",
    )
    _ax.set_title("")
    _ax.legend()

    mo.hstack(
        [
            _ax,
            parameter_ui,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sampling Model

    The speedy `nutpie` sampler provides a beautiful progressbar

    ```python
    with model:
        idata = pm.sample(nuts_sampler="nutpie")
    ```

    Though it can be quite, using a `mo.ui.run_button` can be helpful to trigger execution.
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="click to sample")

    run_button
    return (run_button,)


@app.cell
def _(az):
    idatas: dict[tuple[str, str], az.InferenceData] = {}
    return (idatas,)


@app.cell
def _(
    dataset_selector,
    idatas: "dict[tuple[str, str], az.InferenceData]",
    mo,
    model,
    model_selector,
    pm,
    run_button,
):
    # Sample from the selected model

    has_cached_idata = (dataset_selector.value, model_selector.value) in idatas


    callout = None
    if has_cached_idata:
        idata = idatas[(dataset_selector.value, model_selector.value)]
    elif run_button.value:
        idata = pm.sample(
            model=model,
            nuts_sampler="nutpie",
        )
        idatas[(dataset_selector.value, model_selector.value)] = idata
    else: 
        callout = mo.callout(f"The {model_selector.selected_key} has not been sampled yet on {dataset_selector.selected_key} dataset. Click the button above to sample.")

    callout
    return (idata,)


@app.cell
def _(clone_model, data, idata, model, np, pm):
    # Prediction utility
    def get_predictions(model, idata, new_distances):
        with clone_model(model):
            pm.set_data(
                {
                    "distance": new_distances,
                    "tries": np.ones_like(new_distances, dtype=int),
                    "successes": np.ones_like(new_distances, dtype=int),
                },
                coords={"dist": new_distances},
            )
            predictions = pm.sample_posterior_predictive(
                idata,
                var_names=["p_make"],
                predictions=True,
                progressbar=False,
            )
        return predictions.predictions["p_make"]

    new_distances = np.linspace(1, data["distance"].max(), 100)
    predictions = get_predictions(model, idata, new_distances)
    return (predictions,)


@app.cell
def _(Axes, az, np, plt, st):
    # Plotting utilities

    def format_percentage(ax: Axes) -> Axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
        # ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%'))
        ax.set_yticks(np.arange(0, 1.0001, 0.05), minor=True)

        return ax

    def plot_golf_data(golf_data, ax=None, color="C0"):
        if ax is None:
            fig, ax = plt.subplots()
        bg_color = ax.get_facecolor()
        rv = st.beta(golf_data.successes, golf_data.tries - golf_data.successes)
        ax.vlines(golf_data.distance, *rv.interval(0.68), label=None, color=color)
        ax.plot(
            golf_data.distance,
            golf_data.successes / golf_data.tries,
            "o",
            mec=color,
            mfc=bg_color,
            label=None,
        )
        ax.set_xlabel("Distance from hole (ft)")
        ax.set_ylabel("Percent of putts made")
        ax.set_ylim(bottom=0, top=1)
        format_percentage(ax)
        ax.set_xlim(left=0)
        ax.grid(True, axis="y", alpha=0.7)
        return ax

    def plot_hdi(predictions, ax, hdi_prob=0.68):
        hdi = az.hdi(predictions, hdi_prob=hdi_prob)[predictions.name]
        ax.fill_between(
            predictions["dist"],
            hdi.sel(hdi="lower"),
            hdi.sel(hdi="higher"),
            alpha=0.3,
            label=f"{int(hdi_prob * 100)}% HDI",
            color="C1",
        )
        return ax

    def plot_predictions(predictions, ax=None):
        ax = ax or plt.gca()
        plot_hdi(predictions, ax=ax, hdi_prob=0.94)
        plot_hdi(predictions, ax=ax, hdi_prob=0.68)
        return ax
    return format_percentage, plot_golf_data, plot_predictions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameter Estimates

    [The Arviz Project](https://www.arviz.org/en/latest/) provides tools to analysis Bayesian models.

    ```python
    az.summary(idata, var_names=[rv.name for rv in model.free_RVs])
    ```
    """)
    return


@app.cell
def _(az, idata, model):
    az.summary(idata.posterior[[rv.name for rv in model.free_RVs]]).sort_index()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Fit

    We are able to see how the curve fits (or doesn't fit) our data.
    """)
    return


@app.cell
def _(
    data,
    dataset_selector,
    model_selector,
    plot_golf_data,
    plot_predictions,
    plt,
    predictions,
):
    # Show data and model fit
    _, _ax = plt.subplots(figsize=(8, 5))
    plot_golf_data(data, ax=_ax)
    plot_predictions(predictions, ax=_ax)
    # if reference_line.value:
    #     _ax.axvline(
    #         sim_distance_slider.value,
    #         color="C3",
    #         linestyle="--",
    #         linewidth=1.5,
    #         label=f"Sim distance: {sim_distance_slider.value} ft",
    #     )
    #     _ax.scatter(
    #         sim_distance_slider.value,
    #         sim_made_mean.item(),
    #         color="C3",
    #         label="Sim mean",
    #     )

    _ax.set_title(
        f"Model fit: {model_selector.value.replace('_', ' ').title()} on {dataset_selector.value.title()} data"
    )
    _ax.legend()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## More than just predictions
    """)
    return


@app.cell
def _(model_selector):
    model_selector
    return


@app.cell
def _(mo, sim_distance_slider):
    mo.md(f"""
    Because we assumed a generative model for the putting process, we can simulate putts from any distance using the inferred parameters. 

    For example: **Putting from {sim_distance_slider.value} ft.**
    """)
    return


@app.cell
def _(data, mo, model_selector):
    reference_line = mo.ui.checkbox(label="Show reference line above")

    # UI for simulation distance
    if model_selector.value in [
        "angle",
        "distance_angle",
        "disp_distance_angle",
    ]:
        sim_distance_slider = mo.ui.slider(
            3,
            int(data["distance"].max() * 1.2),
            value=10,
            label="Distance to hole (ft)",
        )
        display = mo.hstack(
            [
                sim_distance_slider,
                # reference_line,
            ],
            justify="start",
        )
    else:
        display = mo.Html(
            "No putting simulation available for Logistic Regression model. Select another model."
        )

    display
    return (sim_distance_slider,)


@app.cell
def _(
    BALL_RADIUS,
    CUP_RADIUS,
    DISTANCE_TOLERANCE,
    OVERSHOT,
    idata,
    np,
    sim_distance_slider,
    xr,
):
    # Simulate putts from a given distance
    def simulate_from_distance(parameters, distance_to_hole, trials=100):
        parameters = (
            parameters if isinstance(parameters, xr.Dataset) else parameters.posterior
        )
        variance_of_shot = parameters["variance_of_shot"]
        theta = np.random.normal(0, variance_of_shot.mean().item(), size=trials)
        if "variance_of_distance" in parameters:
            variance_of_distance = parameters["variance_of_distance"]
            distance = np.random.normal(
                distance_to_hole + OVERSHOT,
                ((distance_to_hole + OVERSHOT) * variance_of_distance.mean().item()),
                size=trials,
            )
        else:
            distance = np.full(trials, distance_to_hole + OVERSHOT)

        x = distance * np.cos(theta)
        y = distance * np.sin(theta)
        made_it = (
            (np.abs(theta) < np.arcsin((CUP_RADIUS - BALL_RADIUS) / distance_to_hole))
            & (x > distance_to_hole)
            & (x < distance_to_hole + DISTANCE_TOLERANCE)
        )
        return x, y, made_it

    sim_x, sim_y, sim_made = simulate_from_distance(
        idata,
        sim_distance_slider.value,
        trials=5000,
    )
    return sim_made, sim_x, sim_y


@app.cell
def _(plt, sim_distance_slider, sim_made, sim_x, sim_y):
    # Plot simulated putts
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(0, 0, "k.", lw=1, mfc="black", ms=250 / sim_distance_slider.value)
    ax.plot(
        sim_x[~sim_made],
        sim_y[~sim_made],
        ".",
        alpha=0.1,
        mfc="r",
        ms=500 / sim_distance_slider.value,
        mew=0.5,
    )
    ax.plot(
        sim_x[sim_made],
        sim_y[sim_made],
        ".",
        alpha=0.1,
        mfc="g",
        ms=500 / sim_distance_slider.value,
        mew=0.5,
    )
    ax.plot(
        sim_distance_slider.value,
        0,
        "ko",
        lw=1,
        mfc="black",
        ms=350 / sim_distance_slider.value,
    )
    ax.set_facecolor("#e6ffdb")
    sim_made_mean = sim_made.mean()
    ax.set_title(
        f"Final position of {len(sim_x)} putts from {sim_distance_slider.value} ft.\n({100 * sim_made_mean:.1f}% made)"
    )
    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")

    first_stroke = ax
    return (first_stroke,)


@app.cell
def _(
    expected_num_putts,
    format_percentage,
    idata,
    np,
    plt,
    sim_distance_slider,
):
    _fig, _ax = plt.subplots(figsize=(6, 6))

    made = expected_num_putts(idata, sim_distance_slider.value)
    x = np.arange(1, 1 + len(made), dtype=int)
    _ax.vlines(np.arange(1, 1 + len(made)), 0, made, linewidths=50)
    _ax.set_title(f"{sim_distance_slider.value} feet")
    _ax.set_ylabel("Percent of attempts")
    _ax.set_xlabel("Number of putts")
    _ax.set_xticks(range(1, 6))
    _ax.set_ylim(0, 1)
    _ax.set_xlim(0, 6)
    _ax.set_title(f"Total strokes needed from {sim_distance_slider.value} ft.")
    format_percentage(_ax)

    follow_through = _ax
    return (follow_through,)


@app.cell
def _(first_stroke, follow_through, mo):
    mo.hstack([first_stroke, follow_through])
    return


@app.cell
def _(BALL_RADIUS, CUP_RADIUS, DISTANCE_TOLERANCE, OVERSHOT, az, np):
    def expected_num_putts(trace, distance_to_hole, trials=100_000):
        distance_to_hole = distance_to_hole * np.ones(trials)

        combined_trace = az.extract(trace)

        n_samples = combined_trace.sizes["sample"]

        idxs = np.random.randint(0, n_samples, trials)
        variance_of_shot = combined_trace["variance_of_shot"].isel(sample=idxs)
        if "variance_of_distance" in combined_trace:
            variance_of_distance = combined_trace["variance_of_distance"].isel(
                sample=idxs
            )

            def new_distance(distance_to_hole):
                return np.random.normal(
                    distance_to_hole + OVERSHOT,
                    (distance_to_hole + OVERSHOT) * variance_of_distance,
                )
        else:
            variance_of_distance = np.zeros(trials)

            def new_distance(distance_to_hole):
                return distance_to_hole + OVERSHOT

        n_shots = []
        while distance_to_hole.size > 0:
            theta = np.random.normal(0, variance_of_shot)

            distance = new_distance(distance_to_hole)

            final_position = np.array(
                [distance * np.cos(theta), distance * np.sin(theta)]
            )

            made_it = np.abs(theta) < np.arcsin(
                (CUP_RADIUS - BALL_RADIUS)
                / distance_to_hole.clip(min=CUP_RADIUS - BALL_RADIUS)
            )
            made_it = (
                made_it
                * (final_position[0] > distance_to_hole)
                * (final_position[0] < distance_to_hole + DISTANCE_TOLERANCE)
            )

            distance_to_hole = np.sqrt(
                (final_position[0] - distance_to_hole) ** 2 + final_position[1] ** 2
            )[~made_it].copy()
            variance_of_shot = variance_of_shot[~made_it]
            variance_of_distance = variance_of_distance[~made_it]
            n_shots.append(made_it.sum())
        return np.array(n_shots) / trials
    return (expected_num_putts,)


@app.cell
def _(mo):
    mo.md("""
    <br>
    <br>
    <br>
    <br>
    ---
    ## Resources

    This was adapted from one of the many PyMC-examples. Find the original notebook [here](https://www.pymc.io/projects/examples/en/latest/case_studies/putting_workflow.html).

    Read the blog post by Andrew Gelman [here](https://mc-stan.org/learn-stan/case-studies/golf.html).

    Find the source code for this notebook on [GitHub (`williambdean/the-best-pymc-example`)](https://github.com/williambdean/the-best-pymc-example).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
