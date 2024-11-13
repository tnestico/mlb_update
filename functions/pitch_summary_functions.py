import pandas as pd
import numpy as np
import json
from  matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import math
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.colors
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import polars as pl
from PIL import Image
import requests
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import PIL


### PITCH COLOURS ###

# Dictionary to map pitch types to their corresponding colors and names
pitch_colours = {
    ## Fastballs ##
    'FF': {'colour': '#FF007D', 'name': '4-Seam Fastball'},
    'FA': {'colour': '#FF007D', 'name': 'Fastball'},
    'SI': {'colour': '#98165D', 'name': 'Sinker'},
    'FC': {'colour': '#BE5FA0', 'name': 'Cutter'},

    ## Offspeed ##
    'CH': {'colour': '#F79E70', 'name': 'Changeup'},
    'FS': {'colour': '#FE6100', 'name': 'Splitter'},
    'SC': {'colour': '#F08223', 'name': 'Screwball'},
    'FO': {'colour': '#FFB000', 'name': 'Forkball'},

    ## Sliders ##
    'SL': {'colour': '#67E18D', 'name': 'Slider'},
    'ST': {'colour': '#1BB999', 'name': 'Sweeper'},
    'SV': {'colour': '#376748', 'name': 'Slurve'},

    ## Curveballs ##
    'KC': {'colour': '#311D8B', 'name': 'Knuckle Curve'},
    'CU': {'colour': '#3025CE', 'name': 'Curveball'},
    'CS': {'colour': '#274BFC', 'name': 'Slow Curve'},
    'EP': {'colour': '#648FFF', 'name': 'Eephus'},

    ## Others ##
    'KN': {'colour': '#867A08', 'name': 'Knuckleball'},
    'PO': {'colour': '#472C30', 'name': 'Pitch Out'},
    'UN': {'colour': '#9C8975', 'name': 'Unknown'},
}

# Create dictionaries for pitch types and their attributes
dict_colour = {key: value['colour'] for key, value in pitch_colours.items()}
dict_pitch = {key: value['name'] for key, value in pitch_colours.items()}
dict_pitch_desc_type = {value['name']: key for key, value in pitch_colours.items()}
dict_pitch_desc_type.update({'Four-Seam Fastball':'FF'})
dict_pitch_desc_type.update({'All':'All'})
dict_pitch_name = {value['name']: value['colour'] for key, value in pitch_colours.items()}
dict_pitch_name.update({'Four-Seam Fastball':'#FF007D'})

font_properties = {'family': 'calibi', 'size': 12}
font_properties_titles = {'family': 'calibi', 'size': 20}
font_properties_axes = {'family': 'calibi', 'size': 16}
     
cmap_sum = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#648FFF','#FFFFFF','#FFB000',])

### FANGRAPHS STATS DICT ###
fangraphs_stats_dict = {'IP':{'table_header':'$\\bf{IP}$','format':'.1f',} ,
 'TBF':{'table_header':'$\\bf{PA}$','format':'.0f',} ,
 'AVG':{'table_header':'$\\bf{AVG}$','format':'.3f',} ,
 'K/9':{'table_header':'$\\bf{K\/9}$','format':'.2f',} ,
 'BB/9':{'table_header':'$\\bf{BB\/9}$','format':'.2f',} ,
 'K/BB':{'table_header':'$\\bf{K\/BB}$','format':'.2f',} ,
 'HR/9':{'table_header':'$\\bf{HR\/9}$','format':'.2f',} ,
 'K%':{'table_header':'$\\bf{K\%}$','format':'.1%',} ,
 'BB%':{'table_header':'$\\bf{BB\%}$','format':'.1%',} ,
 'K-BB%':{'table_header':'$\\bf{K-BB\%}$','format':'.1%',} ,
 'WHIP':{'table_header':'$\\bf{WHIP}$','format':'.2f',} ,
 'BABIP':{'table_header':'$\\bf{BABIP}$','format':'.3f',} ,
 'LOB%':{'table_header':'$\\bf{LOB\%}$','format':'.1%',} ,
 'xFIP':{'table_header':'$\\bf{xFIP}$','format':'.2f',} ,
 'FIP':{'table_header':'$\\bf{FIP}$','format':'.2f',} ,
 'H':{'table_header':'$\\bf{H}$','format':'.0f',} ,
 '2B':{'table_header':'$\\bf{2B}$','format':'.0f',} ,
 '3B':{'table_header':'$\\bf{3B}$','format':'.0f',} ,
 'R':{'table_header':'$\\bf{R}$','format':'.0f',} ,
 'ER':{'table_header':'$\\bf{ER}$','format':'.0f',} ,
 'HR':{'table_header':'$\\bf{HR}$','format':'.0f',} ,
 'BB':{'table_header':'$\\bf{BB}$','format':'.0f',} ,
 'IBB':{'table_header':'$\\bf{IBB}$','format':'.0f',} ,
 'HBP':{'table_header':'$\\bf{HBP}$','format':'.0f',} ,
 'SO':{'table_header':'$\\bf{SO}$','format':'.0f',} ,
 'OBP':{'table_header':'$\\bf{OBP}$','format':'.0f',} ,
 'SLG':{'table_header':'$\\bf{SLG}$','format':'.0f',} ,
 'ERA':{'table_header':'$\\bf{ERA}$','format':'.2f',} ,
 'wOBA':{'table_header':'$\\bf{wOBA}$','format':'.3f',} ,
 'G':{'table_header':'$\\bf{G}$','format':'.0f',}, 
  'strikePercentage':{'table_header':'$\\bf{Strike\%}$','format':'.1%'} }

colour_palette = ['#FFB000','#648FFF','#785EF0',
                  '#DC267F','#FE6100','#3D1EB2','#894D80','#16AA02','#B5592B','#A3C1ED']

### GET COLOURS ###
def get_color(value, normalize, cmap_sum):
    """
    Get the color corresponding to a value based on a colormap and normalization.

    Parameters
    ----------
    value : float
        The value to be mapped to a color.
    normalize : matplotlib.colors.Normalize
        The normalization function to scale the value.
    cmap_sum : matplotlib.colors.Colormap
        The colormap to use for mapping the value to a color.

    Returns
    -------
    str
        The hexadecimal color code corresponding to the value.
    """
    color = cmap_sum(normalize(value))
    return mcolors.to_hex(color)

### PITCH ELLIPSE ###
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    
    if len(x) != len(y):
        raise ValueError("x and y must be the same size")
    try:
        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        facecolor=facecolor,linewidth=2,linestyle='--', **kwargs)
        

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = x.mean()
        

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = y.mean()
        

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
        

        ellipse.set_transform(transf + ax.transData)
    except ValueError:
         return    
        
    return ax.add_patch(ellipse)     
### VELOCITY KDES ###
def velocity_kdes(df: pl.DataFrame, ax: plt.Axes, gs: gridspec.GridSpec, gs_x: list, gs_y: list, fig: plt.Figure):
    """
    Plot the velocity KDEs for different pitch types.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing pitch data.
    ax : plt.Axes
        The axis to plot on.
    gs : GridSpec
        The GridSpec for the subplot layout.
    gs_x : list
        The x-coordinates for the GridSpec.
    gs_y : list
        The y-coordinates for the GridSpec.
    fig : plt.Figure
        The figure to plot on.
    """
    # Get unique pitch types sorted by pitch count
    items_in_order = df.sort("pitch_count", descending=True)['pitch_type'].unique(maintain_order=True).to_numpy()

    # Create the inner subplot inside the outer subplot
    ax.axis('off')
    ax.set_title('Pitch Velocity Distribution', fontdict={'family': 'calibi', 'size': 20})
    inner_grid_1 = gridspec.GridSpecFromSubplotSpec(len(items_in_order), 1, subplot_spec=gs[gs_x[0]:gs_x[-1], gs_y[0]:gs_y[-1]])
    ax_top = [fig.add_subplot(inner) for inner in inner_grid_1]

    for idx, i in enumerate(items_in_order):
        pitch_data = df.filter(pl.col('pitch_type') == i)['start_speed']
        if np.unique(pitch_data).size == 1:  # Check if all values are the same
            ax_top[idx].plot([np.unique(pitch_data), np.unique(pitch_data)], [0, 1], linewidth=4, color=dict_colour[i], zorder=20)
        else:
            sns.kdeplot(pitch_data, ax=ax_top[idx], fill=True, clip=(pitch_data.min(), pitch_data.max()), color=dict_colour[i])

        # Plot the mean release speed for the current data
        df_average = df.filter(df['pitch_type'] == i)['start_speed']
        ax_top[idx].plot([df_average.mean(), df_average.mean()], [ax_top[idx].get_ylim()[0], ax_top[idx].get_ylim()[1]], color=dict_colour[i], linestyle='--')

        # Plot the mean release speed for the statcast group data
        df_statcast_group = pl.read_csv('functions/statcast_2024_grouped.csv')
        df_average = df_statcast_group.filter(df_statcast_group['pitch_type'] == i)['release_speed']
        ax_top[idx].plot([df_average.mean(), df_average.mean()], [ax_top[idx].get_ylim()[0], ax_top[idx].get_ylim()[1]], color=dict_colour[i], linestyle=':')

        ax_top[idx].set_xlim(math.floor(df['start_speed'].min() / 5) * 5, math.ceil(df['start_speed'].max() / 5) * 5)
        ax_top[idx].set_xlabel('')
        ax_top[idx].set_ylabel('')
        if idx < len(items_in_order) - 1:
            ax_top[idx].spines['top'].set_visible(False)
            ax_top[idx].spines['right'].set_visible(False)
            ax_top[idx].spines['left'].set_visible(False)
            ax_top[idx].tick_params(axis='x', colors='none')

        ax_top[idx].set_xticks(range(math.floor(df['start_speed'].min() / 5) * 5, math.ceil(df['start_speed'].max() / 5) * 5, 5))
        ax_top[idx].set_yticks([])
        ax_top[idx].grid(axis='x', linestyle='--')
        ax_top[idx].text(-0.01, 0.5, i, transform=ax_top[idx].transAxes, fontsize=14, va='center', ha='right')

    ax_top[-1].spines['top'].set_visible(False)
    ax_top[-1].spines['right'].set_visible(False)
    ax_top[-1].spines['left'].set_visible(False)
    ax_top[-1].set_xticks(list(range(math.floor(df['start_speed'].min() / 5) * 5, math.ceil(df['start_speed'].max() / 5) * 5, 5)))
    ax_top[-1].set_xlabel('Velocity (mph)')

### TJ STUFF+ ROLLING ###
def tj_stuff_roling(df: pl.DataFrame, window: int, ax: plt.Axes):
    """
    Plot the rolling average of tjStuff+ for different pitch types.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing pitch data.
    window : int
        The window size for calculating the rolling average.
    ax : plt.Axes
        The axis to plot on.
    """
    # Get unique pitch types sorted by pitch count
    items_in_order = df.sort("pitch_count", descending=True)['pitch_type'].unique(maintain_order=True).to_numpy()
    
    # Plot the rolling average for each pitch type
    for i in items_in_order:
        pitch_data = df.filter(pl.col('pitch_type') == i)
        if pitch_data['pitch_count'].max() >= window:
            sns.lineplot(
                x=range(1, pitch_data['pitch_count'].max() + 1),
                y=pitch_data['tj_stuff_plus'].rolling_mean(window),
                color=dict_colour[i],
                ax=ax,
                linewidth=3
            )

    # Adjust x-axis limits to start from 1
    ax.set_xlim(window, df['pitch_count'].max())
    ax.set_ylim(70, 130)
    ax.set_xlabel('Pitches', fontdict=font_properties_axes)
    ax.set_ylabel('tjStuff+', fontdict=font_properties_axes)
    ax.set_title(f"{window} Pitch Rolling tjStuff+", fontdict=font_properties_titles)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

### TJ STUFF+ ROLLING ###
def tj_stuff_roling_game(df: pl.DataFrame, window: int, ax: plt.Axes):
    """
    Plot the rolling average of tjStuff+ for different pitch types over games.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing pitch data.
    window : int
        The window size for calculating the rolling average.
    ax : plt.Axes
        The axis to plot on.
    """
    # Map game_id to sequential numbers
    date_to_number = {date: i + 1 for i, date in enumerate(df['game_id'].unique(maintain_order=True))}

    # Add a column with the sequential game numbers
    df_plot = df.with_columns(
        pl.col("game_id").map_elements(lambda x: date_to_number.get(x, x)).alias("start_number")
    )

    # Group by relevant columns and calculate mean tj_stuff_plus
    plot_game_roll = df_plot.group_by(['start_number', 'game_id', 'game_date', 'pitch_type', 'pitch_description']).agg(
        pl.col('tj_stuff_plus').mean().alias('tj_stuff_plus')
    ).sort('start_number', descending=False)

    # Get the list of pitch types ordered by frequency
    sorted_value_counts = df['pitch_type'].value_counts().sort('count', descending=True)
    items_in_order = sorted_value_counts['pitch_type'].to_list()

    # Plot the rolling average for each pitch type
    for i in items_in_order:
        df_item = plot_game_roll.filter(pl.col('pitch_type') == i)
        df_item = df_item.with_columns(
            pl.col("start_number").cast(pl.Int64)
        ).join(
            pl.DataFrame({"start_number": list(date_to_number.values())}),
            on="start_number",
            how="outer"
        ).sort("start_number_right").with_columns([
            pl.col("start_number").fill_null(strategy="forward").fill_null(strategy="backward"),
            pl.col("tj_stuff_plus").fill_null(strategy="forward").fill_null(strategy="backward"),
            pl.col("pitch_type").fill_null(strategy="forward").fill_null(strategy="backward"),
            pl.col("pitch_description").fill_null(strategy="forward").fill_null(strategy="backward")
        ])

        sns.lineplot(x=range(1, max(df_item['start_number_right']) + 1),
                        y=df_item.filter(pl.col('pitch_type') == i)['tj_stuff_plus'].rolling_mean(window,min_periods=1),
                        color=dict_colour[i],
                        ax=ax, linewidth=3)

        # Highlight missing game data points
        for n in range(len(df_item)):
            if df_item['game_id'].is_null()[n]:
                sns.scatterplot(x=[df_item['start_number_right'][n]],
                                y=[df_item['tj_stuff_plus'].rolling_mean(window,min_periods=1)[n]],
                                color='white',
                                ec=dict_colour[i],
                                ax=ax,
                                zorder=100)

    # Adjust x-axis limits to start from 1
    ax.set_xlim(1, max(df_item['start_number']))
    ax.set_ylim(70, 130)
    ax.set_xlabel('Games', fontdict=font_properties_axes)
    ax.set_ylabel('tjStuff+', fontdict=font_properties_axes)
    ax.set_title(f"{window} Game Rolling tjStuff+", fontdict=font_properties_titles)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def break_plot(df: pl.DataFrame, ax: plt.Axes):
    """
    Plot the pitch breaks for different pitch types.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing pitch data.
    ax : plt.Axes
        The axis to plot on.
    """
    # Get unique pitch types sorted by pitch count
    label_labels = df.sort(by=['pitch_count', 'pitch_type'], descending=[False, True])['pitch_type'].unique(maintain_order=True).to_numpy()

    # Plot confidence ellipses for each pitch type
    for idx, label in enumerate(label_labels):
        subset = df.filter(pl.col('pitch_type') == label)
        if len(subset) > 4:
            try:
                confidence_ellipse(subset['hb'], subset['ivb'], ax=ax, edgecolor=dict_colour[label], n_std=2, facecolor=dict_colour[label], alpha=0.2)
            except ValueError:
                return

    # Plot scatter plot for pitch breaks
    if df['pitcher_hand'][0] == 'R':
        sns.scatterplot(ax=ax, x=df['hb'], y=df['ivb'], hue=df['pitch_type'], palette=dict_colour, ec='black', alpha=1, zorder=2)
    elif df['pitcher_hand'][0] == 'L':
        sns.scatterplot(ax=ax, x=df['hb'], y=df['ivb'], hue=df['pitch_type'], palette=dict_colour, ec='black', alpha=1, zorder=2)

    # Set axis limits
    ax.set_xlim((-25, 25))
    ax.set_ylim((-25, 25))

    # Add horizontal and vertical lines
    ax.hlines(y=0, xmin=-50, xmax=50, color=colour_palette[8], alpha=0.5, linestyles='--', zorder=1)
    ax.vlines(x=0, ymin=-50, ymax=50, color=colour_palette[8], alpha=0.5, linestyles='--', zorder=1)

    # Set axis labels and title
    ax.set_xlabel('Horizontal Break (in)', fontdict=font_properties_axes)
    ax.set_ylabel('Induced Vertical Break (in)', fontdict=font_properties_axes)
    ax.set_title("Pitch Breaks", fontdict=font_properties_titles)

    # Remove legend
    ax.get_legend().remove()

    # Set tick labels
    ax.set_xticklabels(ax.get_xticks(), fontdict=font_properties)
    ax.set_yticklabels(ax.get_yticks(), fontdict=font_properties)

    # Add text annotations for glove side and arm side
    if df['pitcher_hand'][0] == 'R':
        ax.text(-24.5, -24.5, s='← Glove Side', fontstyle='italic', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), fontsize=12, zorder=3)
        ax.text(24.5, -24.5, s='Arm Side →', fontstyle='italic', ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), fontsize=12, zorder=3)
    elif df['pitcher_hand'][0] == 'L':
        ax.invert_xaxis()
        ax.text(24.5, -24.5, s='← Arm Side', fontstyle='italic', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), fontsize=12, zorder=3)
        ax.text(-24.5, -24.5, s='Glove Side →', fontstyle='italic', ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), fontsize=12, zorder=3)

    # Set aspect ratio and format axis ticks
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

# DEFINE STRIKE ZONE
strike_zone = pl.DataFrame({
    'PlateLocSide': [-0.9, -0.9, 0.9, 0.9, -0.9],
    'PlateLocHeight': [1.5, 3.5, 3.5, 1.5, 1.5]
})

### STRIKE ZONE ###
def draw_line(axis, alpha_spot=1, catcher_p=True):
    """
    Draw the strike zone and home plate on the given axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axis to draw the strike zone on.
    alpha_spot : float, optional
        The transparency level of the lines (default is 1).
    catcher_p : bool, optional
        Whether to draw the catcher's perspective (default is True).
    """
    # Draw the strike zone
    axis.plot(strike_zone['PlateLocSide'].to_list(), strike_zone['PlateLocHeight'].to_list(), 
              color='black', linewidth=1.3, zorder=3, alpha=alpha_spot)

    if catcher_p:
        # Draw home plate from catcher's perspective
        axis.plot([-0.708, 0.708], [0.15, 0.15], color='black', linewidth=1, alpha=alpha_spot, zorder=1)
        axis.plot([-0.708, -0.708], [0.15, 0.3], color='black', linewidth=1, alpha=alpha_spot, zorder=1)
        axis.plot([-0.708, 0], [0.3, 0.5], color='black', linewidth=1, alpha=alpha_spot, zorder=1)
        axis.plot([0, 0.708], [0.5, 0.3], color='black', linewidth=1, alpha=alpha_spot, zorder=1)
        axis.plot([0.708, 0.708], [0.3, 0.15], color='black', linewidth=1, alpha=alpha_spot, zorder=1)
    else:
        # Draw home plate from pitcher's perspective
        axis.plot([-0.708, 0.708], [0.4, 0.4], color='black', linewidth=1, alpha=alpha_spot, zorder=1)
        axis.plot([-0.708, -0.9], [0.4, -0.1], color='black', linewidth=1, alpha=alpha_spot, zorder=1)
        axis.plot([-0.9, 0], [-0.1, -0.35], color='black', linewidth=1, alpha=alpha_spot, zorder=1)
        axis.plot([0, 0.9], [-0.35, -0.1], color='black', linewidth=1, alpha=alpha_spot, zorder=1)
        axis.plot([0.9, 0.708], [-0.1, 0.4], color='black', linewidth=1, alpha=alpha_spot, zorder=1)

def location_plot(df: pl.DataFrame, ax: plt.Axes, hand: str):
    """
    Plot the pitch locations for different pitch types against a specific batter hand.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing pitch data.
    ax : plt.Axes
        The axis to plot on.
    hand : str
        The batter hand ('L' for left-handed, 'R' for right-handed).
    """
    # Get unique pitch types sorted by pitch count
    label_labels = df.sort(by=['pitch_count', 'pitch_type'], descending=[False, True])['pitch_type'].unique(maintain_order=True).to_numpy()

    # Plot confidence ellipses for each pitch type
    for label in label_labels:
        subset = df.filter((pl.col('pitch_type') == label) & (pl.col('batter_hand') == hand))
        if len(subset) >= 5:
            confidence_ellipse(subset['px'], subset['pz'], ax=ax, edgecolor=dict_colour[label], n_std=1.5, facecolor=dict_colour[label], alpha=0.3)

    # Group pitch locations by pitch type and calculate mean values
    pitch_location_group = (
        df.filter(pl.col("batter_hand") == hand)
        .group_by("pitch_type")
        .agg([
            pl.col("start_speed").count().alias("pitches"),
            pl.col("px").mean().alias("px"),
            pl.col("pz").mean().alias("pz")
        ])
    )

    # Calculate pitch percentages
    total_pitches = pitch_location_group['pitches'].sum()
    pitch_location_group = pitch_location_group.with_columns(
        (pl.col("pitches") / total_pitches).alias("pitch_percent")
    )

    # Plot pitch locations
    sns.scatterplot(ax=ax, x=pitch_location_group['px'], y=pitch_location_group['pz'],
                    hue=pitch_location_group['pitch_type'], palette=dict_colour, ec='black',
                    s=pitch_location_group['pitch_percent'] * 750, linewidth=2, zorder=2)

    # Customize plot appearance
    ax.axis('square')
    draw_line(ax, alpha_spot=0.75, catcher_p=False)
    ax.axis('off')
    ax.set_xlim((-2.75, 2.75))
    ax.set_ylim((-0.5, 5))
    if len(pitch_location_group['px']) > 0:
        ax.get_legend().remove()
    ax.grid(False)
    ax.set_title(f"Pitch Locations vs {hand}HB\n{pitch_location_group['pitches'].sum()} Pitches", fontdict=font_properties_titles)


def summary_table(df: pl.DataFrame, ax: plt.Axes):
    """
    Create a summary table of pitch data.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing pitch data.
    ax : plt.Axes
        The axis to plot the table on.
    """
    # Aggregate pitch data by pitch description
    df_agg = df.group_by("pitch_description").agg(
        pl.col('is_pitch').sum().alias('count'),
        (pl.col('is_pitch').sum() / df.select(pl.col('is_pitch').sum())).alias('count_percent'),
        pl.col('start_speed').mean().alias('start_speed'),
        pl.col('ivb').mean().alias('ivb'),
        pl.col('hb').mean().alias('hb'),
        pl.col('spin_rate').mean().alias('spin_rate'),
        pl.col('vaa').mean().alias('vaa'),
        pl.col('haa').mean().alias('haa'),
        pl.col('z0').mean().alias('z0'),
        pl.col('x0').mean().alias('x0'),
        pl.col('extension').mean().alias('extension'),
        (((pl.col('spin_direction').mean() + 180) % 360 // 30) + 
        (((pl.col('spin_direction').mean() + 180) % 360 % 30 / 30 / 100 * 60).round(2) * 10).round(0) // 1.5 / 4)
        .cast(pl.Float64).map_elements(lambda x: f"{int(x)}:{int((x % 1) * 60):02d}", return_dtype=pl.Utf8).alias('clock_time'),
        pl.col('tj_stuff_plus').mean().alias('tj_stuff_plus'),
        pl.col('pitch_grade').mean().alias('pitch_grade'),
        (pl.col('in_zone').sum() / pl.col('is_pitch').sum()).alias('zone_percent'),
        (pl.col('ozone_swing').sum() / pl.col('out_zone').sum()).alias('chase_percent'),
        (pl.col('whiffs').sum() / pl.col('swings').sum()).alias('whiff_percent'),
        (pl.col('woba_pred_contact').sum() / pl.col('bip').sum()).alias('xwobacon')
    ).sort("count", descending=True)

    # Aggregate all pitch data
    df_agg_all = df.group_by(pl.lit("All").alias("pitch_description")).agg(
        pl.col('is_pitch').sum().alias('count'),
        (pl.col('is_pitch').sum() / df.select(pl.col('is_pitch').sum())).alias('count_percent'),
        pl.lit(None).alias('start_speed'),
        pl.lit(None).alias('ivb'),
        pl.lit(None).alias('hb'),
        pl.lit(None).alias('spin_rate'),
        pl.lit(None).alias('vaa'),
        pl.lit(None).alias('haa'),
        pl.lit(None).alias('z0'),
        pl.lit(None).alias('x0'),
        pl.col('extension').mean().alias('extension'),
        pl.lit(None).alias('clock_time'),
        pl.col('tj_stuff_plus').mean().alias('tj_stuff_plus'),
        pl.lit(None).alias('pitch_grade'),
        (pl.col('in_zone').sum() / pl.col('is_pitch').sum()).alias('zone_percent'),
        (pl.col('ozone_swing').sum() / pl.col('out_zone').sum()).alias('chase_percent'),
        (pl.col('whiffs').sum() / pl.col('swings').sum()).alias('whiff_percent'),
        (pl.col('woba_pred_contact').sum() / pl.col('bip').sum()).alias('xwobacon')
    )

    # Concatenate aggregated data
    df_agg = pl.concat([df_agg, df_agg_all]).fill_nan(None)

    # Load statcast pitch summary data
    statcast_pitch_summary = pl.read_csv('functions/statcast_2024_grouped.csv')

    # Create table
    table = ax.table(cellText=df_agg.fill_nan('—').fill_null('—').to_numpy(), colLabels=df_agg.columns, cellLoc='center',
                     colWidths=[2.3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], bbox=[0.0, 0, 1, 0.8])

    # Set table properties
    min_font_size = 14
    table.auto_set_font_size(False)
    table.set_fontsize(min_font_size)
    table.scale(1, 0.5)

    # Set font size for values
    min_font_size = 18
    for i in range(len(df_agg) + 1):
        for j in range(len(df_agg.columns)):
            if i > 0:  # Skip the header row
                cell = table.get_celld()[i, j]
                cell.set_fontsize(min_font_size)

    # Define color maps
    cmap_sum = mcolors.LinearSegmentedColormap.from_list("", ['#648FFF', '#FFFFFF', '#FFB000'])
    cmap_sum_r = mcolors.LinearSegmentedColormap.from_list("", ['#FFB000', '#FFFFFF', '#648FFF'])

    # Update table cells with colors and text properties
    for i in range(len(df_agg)):
        pitch_check = dict_pitch_desc_type[df_agg['pitch_description'][i]]
        cell_text = table.get_celld()[(i + 1, 0)].get_text().get_text()

        if cell_text != 'All':
            table.get_celld()[(i + 1, 0)].set_facecolor(dict_pitch_name[cell_text])
            text_props = {'color': '#000000', 'fontweight': 'bold'} if cell_text in ['Split-Finger', 'Slider', 'Changeup'] else {'color': '#ffffff', 'fontweight': 'bold'}
            table.get_celld()[(i + 1, 0)].set_text_props(**text_props)
            if cell_text == 'Four-Seam Fastball':
                table.get_celld()[(i + 1, 0)].get_text().set_text('4-Seam')

        select_df = statcast_pitch_summary.filter(statcast_pitch_summary['pitch_type'] == pitch_check)

        # Apply color to specific columns based on normalized values
        columns_to_color = [(3, 'release_speed', 0.95, 1.05), (11, 'release_extension', 0.9, 1.1), (13, None, 80, 120), 
                            (14, None, 30, 70), (15, 'in_zone_rate', 0.7, 1.3), (16, 'chase_rate', 0.7, 1.3), 
                            (17, 'whiff_rate', 0.7, 1.3), (18, 'xwoba', 0.7, 1.3)]

        for col, stat, vmin_factor, vmax_factor in columns_to_color:
            cell_value = table.get_celld()[(i + 1, col)].get_text().get_text()
            if cell_value != '—':
                vmin = select_df[stat].mean() * vmin_factor if stat else vmin_factor
                vmax = select_df[stat].mean() * vmax_factor if stat else vmax_factor
                normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = cmap_sum if col != 18 else cmap_sum_r
                table.get_celld()[(i + 1, col)].set_facecolor(get_color(float(cell_value.strip('%')), normalize, cmap))

    # Set header text properties
    table.get_celld()[(len(df_agg), 0)].set_text_props(color='#000000', fontweight='bold')

    # Update column names
    new_column_names = ['$\\bf{Pitch\\ Name}$', '$\\bf{Count}$', '$\\bf{Pitch\\%}$', '$\\bf{Velocity}$', '$\\bf{iVB}$', 
                        '$\\bf{HB}$', '$\\bf{Spin}$', '$\\bf{VAA}$', '$\\bf{HAA}$', '$\\bf{vRel}$', '$\\bf{hRel}$', 
                        '$\\bf{Ext.}$', '$\\bf{Axis}$', '$\\bf{tjStuff+}$', '$\\bf{Grade}$', '$\\bf{Zone\\%}$', 
                        '$\\bf{Chase\\%}$', '$\\bf{Whiff\\%}$', '$\\bf{xwOBA}$\n$\\bf{Contact}$']

    for i, col_name in enumerate(new_column_names):
        table.get_celld()[(0, i)].get_text().set_text(col_name)

    # Format cell values
    def format_cells(columns, fmt):
        for col in columns:
            col_idx = df_agg.columns.index(col)
            for row in range(1, len(df_agg) + 1):
                cell_value = table.get_celld()[(row, col_idx)].get_text().get_text()
                if cell_value != '—':
                    table.get_celld()[(row, col_idx)].get_text().set_text(fmt.format(float(cell_value.strip('%'))))

    format_cells(['start_speed', 'ivb', 'hb', 'vaa', 'haa', 'z0', 'x0', 'extension'], '{:,.1f}')
    format_cells(['xwobacon'], '{:,.3f}')
    format_cells(['count_percent', 'zone_percent', 'chase_percent', 'whiff_percent'], '{:,.1%}')
    format_cells(['tj_stuff_plus', 'pitch_grade', 'spin_rate'], '{:,.0f}')

    # Create legend for pitch types
    items_in_order = (df.sort("pitch_count", descending=True)['pitch_type'].unique(maintain_order=True).to_numpy())
    colour_pitches = [dict_colour[x] for x in items_in_order]
    label = [dict_pitch[x] for x in items_in_order]
    handles = [plt.scatter([], [], color=color, marker='o', s=100) for color in colour_pitches]
    if len(label) > 5:
        ax.legend(handles, label, bbox_to_anchor=(0.1, 0.81, 0.8, 0.14), ncol=5,
                  fancybox=True, loc='lower center', fontsize=16, framealpha=1.0, markerscale=1.7, prop={'family': 'calibi', 'size': 16})
    else:
        ax.legend(handles, label, bbox_to_anchor=(0.1, 0.81, 0.8, 0.14), ncol=5,
                  fancybox=True, loc='lower center', fontsize=20, framealpha=1.0, markerscale=2, prop={'family': 'calibi', 'size': 20})
    ax.axis('off')

def plot_footer(ax: plt.Axes):
    """
    Add footer text to the plot.

    Parameters
    ----------
    ax : plt.Axes
        The axis to add the footer text to.
    """
    # Add footer text
    ax.text(0, 1, 'By: @TJStats', ha='left', va='top', fontsize=24)
    ax.text(0.5, 0.25, 
            '''
            Colour Coding Compares to League Average By Pitch
            tjStuff+ calculates the Expected Run Value (xRV) of a pitch regardless of type
            tjStuff+ is normally distributed, where 100 is the mean and Standard Deviation is 10
            Pitch Grade scales tjStuff+ to the traditional 20-80 Scouting Scale for a given pitch type
            ''', 
            ha='center', va='bottom', fontsize=12)
    ax.text(1, 1, 'Data: MLB, Fangraphs\nImages: MLB, ESPN', ha='right', va='top', fontsize=24)
    ax.axis('off')

# Function to get an image from a URL and display it on the given axis
def player_headshot(player_input: str, ax: plt.Axes, sport_id: int, season: int):
    """
    Display the player's headshot image on the given axis.

    Parameters
    ----------
    player_input : str
        The player's ID.
    ax : plt.Axes
        The axis to display the image on.
    sport_id : int
        The sport ID (1 for MLB, other for minor leagues).
    season : int
        The season year.
    """
    try:
        # Construct the URL for the player's headshot image based on sport ID
        if int(sport_id) == 1:
            url = f'https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{player_input}/headshot/silo/current.png'
        else:
            url = f'https://img.mlbstatic.com/mlb-photos/image/upload/c_fill,g_auto/w_640/v1/people/{player_input}/headshot/milb/current.png'

        # Send a GET request to the URL and open the image from the response content
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # Display the image on the axis
        ax.set_xlim(0, 1.3)
        ax.set_ylim(0, 1)
        ax.imshow(img, extent=[0, 1, 0, 1] if sport_id == 1 else [1/6, 5/6, 0, 1], origin='upper')
    except PIL.UnidentifiedImageError:
        ax.axis('off')
        return

    # Turn off the axis
    ax.axis('off')

def player_bio(pitcher_id: str, ax: plt.Axes, sport_id: int, year_input: int):
    """
    Display the player's bio information on the given axis.

    Parameters
    ----------
    pitcher_id : str
        The player's ID.
    ax : plt.Axes
        The axis to display the bio information on.
    sport_id : int
        The sport ID (1 for MLB, other for minor leagues).
    year_input : int
        The season year.
    """
    # Construct the URL to fetch player data
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}&hydrate=currentTeam"

    # Send a GET request to the URL and parse the JSON response
    data = requests.get(url).json()

    # Extract player information from the JSON data
    player_name = data['people'][0]['fullName']
    pitcher_hand = data['people'][0]['pitchHand']['code']
    age = data['people'][0]['currentAge']
    height = data['people'][0]['height']
    weight = data['people'][0]['weight']

    # Display the player's name, handedness, age, height, and weight on the axis
    ax.text(0.5, 1, f'{player_name}', va='top', ha='center', fontsize=56)
    ax.text(0.5, 0.7, f'{pitcher_hand}HP, Age:{age}, {height}/{weight}', va='top', ha='center', fontsize=30)
    ax.text(0.5, 0.45, f'Season Pitching Summary', va='top', ha='center', fontsize=40)

    # Make API call to retrieve sports information
    response = requests.get(url='https://statsapi.mlb.com/api/v1/sports').json()
    
    # Convert the JSON response into a Polars DataFrame
    df_sport_id = pl.DataFrame(response['sports'])    
    abb = df_sport_id.filter(pl.col('id') == sport_id)['abbreviation'][0]

    # Display the season and sport abbreviation
    ax.text(0.5, 0.20, f'{year_input} {abb} Season', va='top', ha='center', fontsize=30, fontstyle='italic')

    # Turn off the axis
    ax.axis('off')

def plot_logo(pitcher_id: str, ax: plt.Axes, df_team: pl.DataFrame, df_players: pl.DataFrame):
    """
    Display the team logo for the given pitcher on the specified axis.

    Parameters
    ----------
    pitcher_id : str
        The ID of the pitcher.
    ax : plt.Axes
        The axis to display the logo on.
    df_team : pl.DataFrame
        The DataFrame containing team data.
    df_players : pl.DataFrame
        The DataFrame containing player data.
    """
    # List of MLB teams and their corresponding ESPN logo URLs
    mlb_teams = [
        {"team": "AZ", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/ari.png&h=500&w=500"},
        {"team": "ATL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/atl.png&h=500&w=500"},
        {"team": "BAL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bal.png&h=500&w=500"},
        {"team": "BOS", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bos.png&h=500&w=500"},
        {"team": "CHC", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chc.png&h=500&w=500"},
        {"team": "CWS", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chw.png&h=500&w=500"},
        {"team": "CIN", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cin.png&h=500&w=500"},
        {"team": "CLE", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cle.png&h=500&w=500"},
        {"team": "COL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/col.png&h=500&w=500"},
        {"team": "DET", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/det.png&h=500&w=500"},
        {"team": "HOU", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/hou.png&h=500&w=500"},
        {"team": "KC", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/kc.png&h=500&w=500"},
        {"team": "LAA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/laa.png&h=500&w=500"},
        {"team": "LAD", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/lad.png&h=500&w=500"},
        {"team": "MIA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/mia.png&h=500&w=500"},
        {"team": "MIL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/mil.png&h=500&w=500"},
        {"team": "MIN", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/min.png&h=500&w=500"},
        {"team": "NYM", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nym.png&h=500&w=500"},
        {"team": "NYY", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nyy.png&h=500&w=500"},
        {"team": "OAK", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/oak.png&h=500&w=500"},
        {"team": "PHI", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/phi.png&h=500&w=500"},
        {"team": "PIT", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/pit.png&h=500&w=500"},
        {"team": "SD", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sd.png&h=500&w=500"},
        {"team": "SF", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sf.png&h=500&w=500"},
        {"team": "SEA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sea.png&h=500&w=500"},
        {"team": "STL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/stl.png&h=500&w=500"},
        {"team": "TB", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tb.png&h=500&w=500"},
        {"team": "TEX", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tex.png&h=500&w=500"},
        {"team": "TOR", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tor.png&h=500&w=500"},
        {"team": "WSH", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/wsh.png&h=500&w=500"},
        {"team": "ATH", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/oak.png&h=500&w=500"},
    ]

    try:
        # Create a DataFrame from the list of dictionaries
        df_image = pd.DataFrame(mlb_teams)
        image_dict = df_image.set_index('team')['logo_url'].to_dict()

        # Get the team ID for the given pitcher
        team_id = df_players.filter(pl.col('player_id') == pitcher_id)['team'][0]

        # Construct the URL to fetch team data
        url_team = f'https://statsapi.mlb.com/api/v1/teams/{team_id}'

        # Send a GET request to the team URL and parse the JSON response
        data_team = requests.get(url_team).json()

        # Extract the team abbreviation
        if data_team['teams'][0]['id'] in df_team['parent_org_id']:
            team_abb = df_team.filter(pl.col('team_id') == data_team['teams'][0]['id'])['parent_org_abbreviation'][0]
        else:
            team_abb = df_team.filter(pl.col('parent_org_id') == data_team['teams'][0]['parentOrgId'])['parent_org_abbreviation'][0]

        # Get the logo URL from the image dictionary using the team abbreviation
        logo_url = image_dict[team_abb]

        # Send a GET request to the logo URL
        response = requests.get(logo_url)

        # Open the image from the response content
        img = Image.open(BytesIO(response.content))

        # Display the image on the axis
        ax.set_xlim(0, 1.3)
        ax.set_ylim(0, 1)
        ax.imshow(img, extent=[0.3, 1.3, 0, 1], origin='upper')

        # Turn off the axis
        ax.axis('off')
    except KeyError:
        ax.axis('off')
        return

splits = {
    'all':0,
    'left':13,
    'right':14,
}

splits_title = {

    'all':'',
    'left':' vs LHH',
    'right':' vs RHH',

}


def fangraphs_pitching_leaderboards(season: int,
                                    split: str,
                                    start_date: str = '2024-01-01',
                                    end_date: str = '2024-12-31'):
    """
    Fetch pitching leaderboards data from Fangraphs.

    Parameters
    ----------
    season : int
        The season year.
    split : str
        The split type (e.g., 'All', 'LHH', 'RHH').
    start_date : str, optional
        The start date for the data (default is '2024-01-01').
    end_date : str, optional
        The end date for the data (default is '2024-12-31').

    Returns
    -------
    pl.DataFrame
        The DataFrame containing the pitching leaderboards data.
    """
    url = f"""
           https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=pit&lg=all&season={season}&season1={season}
           &startdate={start_date}&enddate={end_date}&ind=0&qual=0&type=8&month=1000&pageitems=500000
           """

    data = requests.get(url).json()
    df = pl.DataFrame(data=data['data'], infer_schema_length=1000)
    return df

def fangraphs_splits_scrape(player_input: str, year_input: int, start_date: str, end_date: str, split: str) -> pl.DataFrame:
    """
    Scrape Fangraphs splits data for a specific player.

    Parameters
    ----------
    player_input : str
        The player's ID.
    year_input : int
        The season year.
    start_date : str
        The start date for the data.
    end_date : str
        The end date for the data.
    split : str
        The split type (e.g., 'all', 'left', 'right').

    Returns
    -------
    pl.DataFrame
        The DataFrame containing the splits data.
    """
    split_dict = {
        'all': [],
        'left': ['5'],
        'right': ['6']
    }

    

    url = "https://www.fangraphs.com/api/leaders/splits/splits-leaders"

    # Get Fangraphs player ID
    fg_id = str(fangraphs_pitching_leaderboards(
        year_input,
        split='All',
        start_date=f'{year_input}-01-01',
        end_date=f'{year_input}-12-31'
    ).filter(pl.col('xMLBAMID') == player_input)['playerid'][0])

    # Payload for basic stats
    payload = {
        "strPlayerId": fg_id,
        "strSplitArr": split_dict[split],
        "strGroup": "season",
        "strPosition": "P",
        "strType": "2",
        "strStartDate": pd.to_datetime(start_date).strftime('%Y-%m-%d'),
        "strEndDate": pd.to_datetime(end_date).strftime('%Y-%m-%d'),
        "strSplitTeams": False,
        "dctFilters": [],
        "strStatType": "player",
        "strAutoPt": False,
        "arrPlayerId": [],
        "strSplitArrPitch": [],
        "arrWxTemperature": None,
        "arrWxPressure": None,
        "arrWxAirDensity": None,
        "arrWxElevation": None,
        "arrWxWindSpeed": None
    }

    # Fetch basic stats
    response = requests.post(url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
    data_pull = response.json()['data'][0]

    # Payload for advanced stats
    payload_advanced = payload.copy()
    payload_advanced["strType"] = "1"

    # Fetch advanced stats
    response_advanced = requests.post(url, data=json.dumps(payload_advanced), headers={'Content-Type': 'application/json'})
    data_pull_advanced = response_advanced.json()['data'][0]

    # Combine basic and advanced stats
    data_pull.update(data_pull_advanced)
    df_pull = pl.DataFrame(data_pull)

    return df_pull

                                   
def fangraphs_table(df: pl.DataFrame,
                    ax: plt.Axes,
                    player_input: str,
                    season: int,
                    split: str):
    """
    Create a table of Fangraphs pitching leaderboards data for a specific player.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot the table on.
    season : int
        The season year.
    split : str
        The split type (e.g., 'All', 'LHH', 'RHH').
    """

    start_date = df['game_date'][0]
    end_date = df['game_date'][-1]

    # Fetch Fangraphs splits data
    df_fangraphs = fangraphs_splits_scrape(player_input=player_input, 
                                           year_input=season,
                                           start_date=start_date,
                                           end_date=end_date,
                                           split=split)

    # Select relevant columns for the table
    plot_table = df_fangraphs.select(['IP', 'WHIP', 'ERA', 'TBF', 'FIP', 'K%', 'BB%', 'K-BB%'])

    # Format table values
    plot_table_values = [format(plot_table[x][0], fangraphs_stats_dict[x]['format']) if plot_table[x][0] != '---' else '---' for x in plot_table.columns]
    
    # Create the table
    table_fg = ax.table(cellText=[plot_table_values], colLabels=plot_table.columns, cellLoc='center',
                        bbox=[0.0, 0.1, 1, 0.7])

    # Set font size for the table
    min_font_size = 20
    table_fg.set_fontsize(min_font_size)

    # Update column names with formatted headers
    new_column_names = [fangraphs_stats_dict[col]['table_header'] for col in plot_table.columns]
    for i, col_name in enumerate(new_column_names):
        table_fg.get_celld()[(0, i)].get_text().set_text(col_name)

    # Set header text properties
    ax.text(0.5, 0.9, f'{start_date} to {end_date}{splits_title[split]}', va='bottom', ha='center',
            fontsize=36, fontstyle='italic')
    ax.axis('off')


def stat_summary_table(df: pl.DataFrame, 
                        player_input: int, 
                        sport_id: int, 
                        ax: plt.Axes,
                        split: str = 'All'):
    """
    Create a summary table of player statistics.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing pitch data.
    player_input : int
        The player's ID.
    sport_id : int
        The sport ID (1 for MLB, other for minor leagues).
    ax : plt.Axes
        The axis to plot the table on.
    split : str, optional
        The split type (default is 'All').
    """
    # Format start and end dates
    start_date_format = str(pd.to_datetime(df['game_date'][0]).strftime('%m/%d/%Y'))
    end_date_format = str(pd.to_datetime(df['game_date'][-1]).strftime('%m/%d/%Y'))

    # Determine app context based on sport ID
    appContext = 'majorLeague' if sport_id == 1 else 'minorLeague'

    # Fetch player stats from MLB API
    pitcher_stats_call = requests.get(
        f'https://statsapi.mlb.com/api/v1/people/{player_input}?appContext={appContext}&hydrate=stats(group=[pitching],type=[byDateRange],sportId={sport_id},startDate={start_date_format},endDate={end_date_format})'
    ).json()

    # Extract stats and create DataFrame
    pitcher_stats_call_header = [x for x in pitcher_stats_call['people'][0]['stats'][0]['splits'][-1]['stat']]
    pitcher_stats_call_values = [pitcher_stats_call['people'][0]['stats'][0]['splits'][-1]['stat'][x] for x in pitcher_stats_call['people'][0]['stats'][0]['splits'][-1]['stat']]
    pitcher_stats_call_df = pl.DataFrame(data=dict(zip(pitcher_stats_call_header, pitcher_stats_call_values)))

    # Add additional calculated columns
    pitcher_stats_call_df = pitcher_stats_call_df.with_columns(
        pl.lit(df['is_whiff'].sum()).alias('whiffs'),
        (pl.col('strikeOuts') / pl.col('battersFaced') * 100).round(1).cast(pl.Utf8).str.concat('%').alias('k_percent'),
        (pl.col('baseOnBalls') / pl.col('battersFaced') * 100).round(1).cast(pl.Utf8).str.concat('%').alias('bb_percent'),
        ((pl.col('strikeOuts') - pl.col('baseOnBalls')) / pl.col('battersFaced') * 100).round(1).cast(pl.Utf8).str.concat('%').alias('k_bb_percent'),
        (((pl.col('homeRuns') * 13 + 3 * ((pl.col('baseOnBalls')) + (pl.col('hitByPitch'))) - 2 * (pl.col('strikeOuts')))) / ((pl.col('outs')) / 3) + 3.15).round(2).map_elements(lambda x: f"{x:.2f}").alias('fip'),
        ((pl.col('strikes') / pl.col('numberOfPitches') * 100)).round(1).cast(pl.Utf8).str.concat('%').alias('strikePercentage'),
    )

    # Determine columns and title based on game count and sport ID
    if df['game_id'][0] == df['game_id'][-1]:
        pitcher_stats_call_df_small = pitcher_stats_call_df.select(['inningsPitched', 'battersFaced', 'earnedRuns', 'hits', 'strikeOuts', 'baseOnBalls', 'hitByPitch', 'homeRuns', 'strikePercentage', 'whiffs'])
        new_column_names = ['$\\bf{IP}$', '$\\bf{PA}$', '$\\bf{ER}$', '$\\bf{H}$', '$\\bf{K}$', '$\\bf{BB}$', '$\\bf{HBP}$', '$\\bf{HR}$', '$\\bf{Strike\%}$', '$\\bf{Whiffs}$']
        title = f'{df["game_date"][0]} vs {df["batter_team"][0]}'
    elif sport_id != 1:
        pitcher_stats_call_df_small = pitcher_stats_call_df.select(['inningsPitched', 'battersFaced', 'whip', 'era', 'fip', 'k_percent', 'bb_percent', 'k_bb_percent', 'strikePercentage'])
        new_column_names = ['$\\bf{IP}$', '$\\bf{PA}$', '$\\bf{WHIP}$', '$\\bf{ERA}$', '$\\bf{FIP}$', '$\\bf{K\%}$', '$\\bf{BB\%}$', '$\\bf{K-BB\%}$', '$\\bf{Strike\%}$']
        title = f'{df["game_date"][0]} to {df["game_date"][-1]}'
    else:
        fangraphs_table(df=df, ax=ax, player_input=player_input, season=int(df['game_date'][0][0:4]), split=split)
        return

    # Create and format the table
    table_fg = ax.table(cellText=pitcher_stats_call_df_small.to_numpy(), colLabels=pitcher_stats_call_df_small.columns, cellLoc='center', bbox=[0.0, 0.1, 1, 0.7])
    table_fg.set_fontsize(20)
    for i, col_name in enumerate(new_column_names):
        table_fg.get_celld()[(0, i)].get_text().set_text(col_name)

    # Add title to the plot
    ax.text(0.5, 0.9, title, va='bottom', ha='center', fontsize=36, fontstyle='italic')
    ax.axis('off')
