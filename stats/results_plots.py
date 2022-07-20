#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def boxplot(df, size=(8, 5), scale=1.5):
    """
    :param df: dataframe to extract the data to hideplot
    :param size: a tuple with figure width and height
    :param scale: a flot to increase or decrease the size of the font
    :return: boxplot figures
    """
    sns.set(style='whitegrid', font_scale=scale)
    fig = plt.figure(figsize=size)
    df.boxplot(vert=True, patch_artist=True,
               boxprops={'facecolor': 'white', 'edgecolor': 'blue',
                         'linewidth': 1},
               whiskerprops={'color': 'blue'},
               medianprops={'color': 'green', 'linewidth': 0.5},
               capprops={'color': 'blue'})
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)

    return fig


def save_plot(fig, outdir, plot_name, hideplot=True):
    """
    :param fig: the figure to save
    :param outdir: a string for the directory to save the figure
    :param plot_name: a string with the name to save the figure
    :param hideplot: a boolean to display or not the figure
    :return: if hideplot it closes the figure
    """
    plt.savefig(os.path.join(outdir, plot_name), bbox_inches='tight',
                format='eps', dpi=1200)
    print('...Plot {} saved...'.format(plot_name))

    if hideplot:
        return plt.close(fig)
    else:
        return plt.show(fig)


def multiple_comparisons(stats, outdir, group, hideplot, title=True):
    """
    :param stats: a dictionary containing the p-values from the
        multi-comparison post-hoc test of the metrics
    :param outdir: output directory to save th metrics figures
    :param group: a string to group the data by (name of the column)
    :param hideplot: boolean to display or not the figures
    :param title: a boolean indicating whether to add a title to the plot
    :return: a print statement to indicate the plots are saved
    """

    heatmap = {'linewidths': 0.5, 'linecolor': 'black', 'clip_on': False,
               'square': True}

    sns.set(font_scale=1.3)

    if group == 'Architecture':
        size = (4, 4)
        fname = 'net_'
    else:
        size = (6, 6)
        fname = 'exp_'

    for k in stats.keys():
        fig = plt.figure(figsize=size)
        if title:
            plt.title(label=k, loc='center')
        sign_plot(stats[k][2], **heatmap)
        plot_name = fname + k.lower() + '.eps'
        save_plot(fig, outdir, plot_name, hideplot)

    print('...Pairwise comparison completed...')


def probs_heatmap(probs, outdir, group, label='Maximum accuracy', name='acc',
                  hideplot=False):
    """
    :param probs: a dictionary containing the p-values from the
        multi-comparison post-hoc test of the maximum accuracy or number of training epochs
    :param outdir: directory to save the images.
    :param group: column name to group by
    :param label: a string for the figure title
    :param name: name of the column to extract the data
    :param hideplot: boolean to show the plots
    :return: comparison plots
    """
    if group == 'Architecture':
        size = (4, 4)
        fname = 'net_'
    else:
        size = (6, 6)
        fname = 'exp_'

    sns.set(font_scale=1.3)
    heatmap = {'linewidths': 0.5, 'linecolor': 'black', 'clip_on': False,
               'square': True}

    fig = plt.figure(figsize=size)
    if label is not None:
        plt.title(label=label, loc='center')
    sign_plot(probs, **heatmap)
    plot_name = fname + 'hm_' + name + '.eps'

    return save_plot(fig, outdir, plot_name, hideplot)


def prob_bar(outdir, cmap=None, hideplot=True):
    """
    :param outdir: directory to save the figure
    :param cmap: list of  5 strings color for the heatmap
    :param hideplot: a boolean to display or not the hideplot
    :return: save the probability bar for the post-hoc comparison
    Note: This function is adapted from scikit-posthocs
    """

    if not cmap:
        cmap = ['1', '#a1d99b', '#005a32', '#238b45', '#15B01A']
        # cmap = ['1', '#c2e699', '#006837', '#31a354', '#78c679']

    if len(cmap) != 5:
        raise ValueError('cmap list must contain 5 items')

    fig = plt.figure(figsize=(6, 0.2))
    g = ['NS', 'p < 0.05', 'p < 0.01', 'p < 0.001']
    a = [[0, 3, 2, 1]]
    res = sns.heatmap(a, vmin=-1, vmax=3, cbar=False,
                      cmap=ListedColormap(cmap), xticklabels=g, yticklabels='')
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=15)
    plot_name = 'prob_bar.eps'

    return save_plot(fig, outdir, plot_name, hideplot)


def sign_plot(x, g=None, labels=True, cmap=None, **kwargs):
    """
    Significance hideplot, a heatmap of p values (based on Seaborn).
    Note: This function is adapted from scikit-posthocs
    Parameters
    ----------
    x : x must be an array, any object exposing
        the array interface, containing p values.
    
    labels : bool
        Plot axes labels (default) or not.
        
    g : Union[List, np.ndarray]
         An array, any object exposing the array interface, containing
         group names.

    cmap : List consisting of five elements, that will be exported to
        ListedColormap method of matplotlib. First is for diagonal
        elements, second is for non-significant elements, third is for
        p < 0.001, fourth is for p < 0.01, fifth is for p < 0.05.

    kwargs
        Keyword arguments to be passed to seaborn heatmap method. These
        keyword params cannot be used: cbar, vmin, vmax, center.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object with the heatmap.
    """

    for key in ['cbar', 'vmin', 'vmax', 'center']:
        if key in kwargs:
            del kwargs[key]

    if isinstance(x, pd.DataFrame):
        df = x.copy()
    else:
        x = np.array(x)
        g = g or np.arange(x.shape[0])
        df = pd.DataFrame(np.copy(x), index=g, columns=g)

    if not cmap:
        cmap = ['1', '#a1d99b', '#005a32', '#238b45', '#15B01A']
        # cmap = ['1', '#c2e699', '#006837', '#31a354', '#78c679']

    df[(x < 0.001) & (x >= 0)] = 1
    df[(x < 0.01) & (x >= 0.001)] = 2
    df[(x < 0.05) & (x >= 0.01)] = 3
    df[(x >= 0.05)] = 0

    np.fill_diagonal(df.values, -1)

    if len(cmap) != 5:
        raise ValueError('cmap list must contain 5 items')

    hax = sns.heatmap(
        df, vmin=-1, vmax=3, cmap=ListedColormap(cmap), center=1,
        cbar=False, **kwargs)
    hax.set_xticklabels(hax.get_xmajorticklabels(), fontsize=15)
    hax.set_yticklabels(hax.get_ymajorticklabels(), fontsize=15)

    if not labels:
        hax.set_xlabel('')
        hax.set_ylabel('')

    return hax


def max_acc_epoch_plot(df, group, col_name='max_acc', outdir=None, hideplot=False):
    """
    :param df: a dataframe with raw data
    :param group: the name of the column to group by
    :param col_name: name of the column with the values to hideplot
    :param outdir: a string, directory to save the plots
    :param hideplot: a boolean to display or not the figures
    :return: a hideplot of  acc_ per group/treatment

    """

    size = (2, 2)
    if group == 'Architecture':
        fname = 'net_'
        n_cols = 5
        sns.set(style='whitegrid', font_scale=1.3)
    else:
        fname = 'exp_'
        n_cols = 4
        sns.set(style='whitegrid', font_scale=1.3)

    fig = plt.figure(figsize=size)
    f = sns.FacetGrid(data=df[[group, col_name]], col=group, hue=group,
                      col_wrap=n_cols)
    # density plots
    g = f.map(sns.kdeplot, col_name, fill=True, common_norm=False, alpha=1.0,
              legend=False, warn_singular=False)
    # control the title of each facet
    g.set_titles('{col_name}')

    # save the hideplot
    if 'max' in col_name:
        col_name = 'max_acc'
    # save the hideplot
    plot_name = fname + col_name + '_g.eps'
    return save_plot(fig, outdir, plot_name, hideplot)

    # print('...Distribution plots saved...')


def boxplot_epochs(acc_df, ep_df, group, outdir, hideplot=False):
    """
    :param acc_df: dataframe with maximum validation accuracy
    :param ep_df: dataframe with number of epochs at which the max validation accuracy was achieved
    :param group: string to group by the data
    :param outdir: directory to save the plots
    :param hideplot: shows or hides figures/plots
    :return: save plots
    """

    size = (8, 5)
    if group == 'Architecture':
        fname = 'net_'
    else:
        fname = 'exp_'
    sns.set(style='whitegrid', font_scale=1.6)

    for df in [acc_df, ep_df]:
        fig = boxplot(df, size)
        if df is acc_df:
            plt.ylabel('Max accuracy (%)')
            plot_name = fname + 'box_max_acc' + '.eps'
        else:
            plt.ylabel('Epochs')
            plot_name = fname + 'box_epochs' + '.eps'
        save_plot(fig, outdir, plot_name, hideplot)

    print('...Max val accuracy and epochs boxplots saved...')
