"""
helpers.py — Darwin's Ark Behavioral Analysis
==============================================
Reusable functions for the BILD 62 final project:
  - summarize_factor_regression
  - plot_sterilization_bars
  - plot_aging_trends

Imported by the main Jupyter notebook to keep it clean and readable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.formula.api as smf


def summarize_factor_regression(df, factor_num, formula, factor_names):
    """
    Fit an OLS regression for a single behavioral factor and print a summary.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame (df_clean) in long format with columns
        ['factor', 'score', 'age', 'sex_bin', 'sterilized_bin'].
    factor_num : int
        Integer 1–8 identifying which behavioral factor to model.
    formula : str
        Patsy/OLS formula string, e.g. 'score ~ age + sex_bin + sterilized_bin'.
    factor_names : dict
        Mapping of factor number → human-readable name.

    Returns
    -------
    model : statsmodels RegressionResultsWrapper
        Fitted OLS result object.

    Raises
    ------
    AssertionError
        If factor_num is not in factor_names, required columns are missing,
        or no rows exist for the given factor.
    """
    assert factor_num in factor_names, (
        f"factor_num {factor_num!r} not found in factor_names. "
        f"Valid options: {sorted(factor_names.keys())}"
    )
    required_cols = {'factor', 'score'}
    assert required_cols.issubset(df.columns), (
        f"DataFrame is missing required columns: {required_cols - set(df.columns)}"
    )

    factor_df = df[df['factor'] == factor_num].copy()
    assert len(factor_df) > 0, f"No rows found for factor {factor_num}."

    factor_name = factor_names.get(factor_num, f'Factor {factor_num}')
    model = smf.ols(formula, data=factor_df).fit()

    print(f"=== {factor_name} (Factor {factor_num}) ===")
    print(f"R²: {model.rsquared:.4f}   Adj. R²: {model.rsquared_adj:.4f}   N: {int(model.nobs)}")
    print(model.params.to_string())
    print(f"p-values:\n{model.pvalues.to_string()}")
    print()

    return model


def plot_sterilization_bars(df_clean, factor_names, save_path=None):
    """
    Plot mean behavioral factor scores grouped by sex and sterilization status.

    For each of the 8 behavioral factors, draws a bar chart of group means
    (female/intact, female/sterilized, male/intact, male/sterilized) with
    95% confidence interval error bars.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned long-format DataFrame with columns
        ['factor', 'factor_name', 'sex', 'sterilized', 'score'].
    factor_names : dict
        Mapping of factor number → human-readable name.
    save_path : str or None
        If provided, saves the figure to this file path.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    AssertionError
        If required columns are missing from df_clean.
    """
    required = {'factor', 'factor_name', 'sex', 'sterilized', 'score'}
    assert required.issubset(df_clean.columns), (
        f"df_clean is missing columns: {required - set(df_clean.columns)}"
    )

    # Compute group means and 95% CI per factor × sex × sterilized
    summary = (
        df_clean
        .groupby(['factor_name', 'factor', 'sex', 'sterilized'])['score']
        .agg(['mean', 'sem', 'count'])
        .reset_index()
    )
    summary.columns = ['factor_name', 'factor', 'sex', 'sterilized', 'mean', 'sem', 'n']
    summary = summary.sort_values('factor')

    # Combine sex + sterilized into a readable x-axis label
    summary['group'] = summary['sex'] + '\n(' + summary['sterilized'] + ')'

    palette = {
        'female\n(no)':  '#f4a7b9',  # light pink — intact female
        'female\n(yes)': '#c2185b',  # dark pink  — sterilized female
        'male\n(no)':    '#90caf9',  # light blue — intact male
        'male\n(yes)':   '#1565c0'   # dark blue  — sterilized male
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    axes = axes.flatten()

    for i, factor_num in enumerate(range(1, 9)):
        ax = axes[i]
        data = summary[summary['factor'] == factor_num]

        ax.bar(
            data['group'], data['mean'],
            color=[palette[g] for g in data['group']],
            width=0.6, edgecolor='white'
        )
        # Error bars represent 95% CI (mean ± 1.96 × SE)
        ax.errorbar(
            x=range(len(data)), y=data['mean'],
            yerr=data['sem'] * 1.96,
            fmt='none', color='black', capsize=4, linewidth=1
        )

        ax.set_title(data['factor_name'].iloc[0], fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean Score', fontsize=8)
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=8)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    plt.suptitle(
        'Q1: Behavioral Factor Scores by Sex and Sterilization Status\n'
        '(error bars = 95% CI)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_aging_trends(df_clean, age_df, factor_names, save_path=None):
    """
    Plot behavioral factor scores across age with quadratic fitted curves.

    For each of the 8 behavioral factors, draws binned group means (± 95% CI)
    overlaid with a quadratic OLS fit. Uses pre-computed models stored in age_df
    to avoid redundant model fitting.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned long-format DataFrame with columns
        ['factor', 'factor_name', 'age', 'age_bin', 'score',
         'sex_bin', 'sterilized_bin'].
    age_df : pd.DataFrame
        Output from the age modeling loop; must contain columns
        ['factor', 'r2_gain', 'model'] where 'model' holds fitted OLS objects.
    factor_names : dict
        Mapping of factor number → human-readable name.
    save_path : str or None
        If provided, saves the figure to this file path.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    AssertionError
        If 'model' column is missing from age_df, or 'age_bin' is missing
        from df_clean (run pd.cut() before calling this function).
    """
    assert 'model' in age_df.columns, (
        "'age_df' must contain a 'model' column with fitted OLS objects. "
        "Make sure to store model objects in the age modeling loop."
    )
    assert 'age_bin' in df_clean.columns, (
        "df_clean must contain 'age_bin'. Run pd.cut() to create age bins first."
    )

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)
    axes = axes.flatten()
    colors = cm.tab10.colors

    # Build a single age sequence for predictions — held at female, sterilized
    # (the majority group) so curves reflect the typical dog in the dataset
    age_seq = pd.DataFrame({
        'age':            np.linspace(0.5, 18, 200),
        'sex_bin':        0,   # female
        'sterilized_bin': 1    # sterilized
    })

    for i, factor_num in enumerate(range(1, 9)):
        ax = axes[i]
        factor_df = df_clean[df_clean['factor'] == factor_num].copy()
        factor_name = factor_df['factor_name'].iloc[0]

        # Compute binned means and 95% CI for scatter points
        binned = (
            factor_df.groupby('age_bin', observed=True)['score']
            .agg(['mean', 'sem', 'count'])
            .reset_index()
        )
        # Use the left edge of each bin label as the x midpoint
        bin_mids = [float(str(b).split('–')[0]) + 1 for b in binned['age_bin']]

        ax.errorbar(
            bin_mids, binned['mean'],
            yerr=binned['sem'] * 1.96,
            fmt='o', color=colors[i], markersize=5,
            capsize=3, linewidth=1, label='Mean ± 95% CI'
        )

        # Retrieve the pre-fitted quadratic model — no redundant re-fitting
        model = age_df.loc[age_df['factor'] == factor_num, 'model'].values[0]
        fitted = model.predict(age_seq)

        ax.plot(age_seq['age'], fitted, color=colors[i],
                linewidth=2, linestyle='-', label='Quadratic fit')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        # Annotate with ΔR² — how much variance age terms add over baseline
        r2g = age_df.loc[age_df['factor'] == factor_num, 'r2_gain'].values[0]
        ax.text(0.97, 0.05, f'ΔR²={r2g:.3f}',
                transform=ax.transAxes, ha='right',
                fontsize=8, color='dimgray')

        ax.set_title(factor_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Age (years)', fontsize=8)
        ax.set_ylabel('Mean Score', fontsize=8)
        ax.set_xlim(0, 19)

    plt.suptitle(
        'Q2: Behavioral Factor Scores Across Age\n'
        '(holding sex=female, sterilized=yes; dots = binned means ± 95% CI)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
