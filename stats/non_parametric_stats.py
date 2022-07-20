#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
import statistics as stat
from tqdm import tqdm
import matplotlib.pyplot as plt
from stats.results_plots import *
from stats.utils import rank_confidence
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# seed the random number generator
np.random.seed(123)


def acc_summary(group, accs, df, outdir, csv_dir, plot=True, hideplot=False):
    """
    :param group: name of the column to group the data either Architecture or Experiment
        (i.e. exp number, architecture, loss, optimizer)
    :param accs: list of accuracy strings for training, test and validation
    :param df: a dataframe with the data runs
    :param outdir: the directory to save the figures/plots
    :param csv_dir: a string containing the path to save the results in csv format
    :param plot: a boolean to generate or not a boxplot
    :param hideplot: a boolean to display or not a hideplot
    :return: a dataframe containing the average accuracy and std per each item
        in accs list by the group
    """

    df_acc = df.groupby(group)[accs].agg(['mean', 'std'])
    for acc in accs:
        df_acc[acc] = np.round(df_acc[acc] * 100, 2)
        df_ = df[['Run', group, acc]].pivot_table(index=['Run'], columns=group)
        df_[acc] = round(df_[acc] * 100, 4)
        df_.columns = df_.columns.droplevel().rename(None)
        if plot:
            fig = boxplot(df_)
            plt.xlabel(group)
            plt.ylabel('Accuracy (%)')
            plot_name = group[0:3].lower() + '_' + acc.replace(' ', '_').lower() + '.eps'
            # Save the figure
            save_plot(fig, outdir, plot_name, hideplot)
    # Saving the dataframe
    fname = 'acc_summ_' + group.lower() + '_.csv'
    df_acc.to_csv(os.path.join(csv_dir, fname))
    df_acc.to_csv(csv_dir + fname)

    return df_acc


def stats_summary(group, metrics, df, csv_dir, formatting=2):
    """
    :param group: the column name to sort the data either Architecture or Experiment
    :param metrics: list of strings containing the performance metrics measured
        in the test dataset before bootstrapping (columns of the dataframe)
    :param df: a dataframe with the results
    :param csv_dir: a string containing the path to save the results in csv format
    :param formatting: an integer to round the metrics output
    :return: a data frame containing the architecture/experiment metrics, average and std
    """
    stats_list = list()
    for m in metrics:
        stats = df.groupby(group)[m].agg(['min', 'max', 'mean', 'std'])
        stats.insert(0, 'metric', m)
        stats_list.append(stats)

    df_ = pd.concat(stats_list)

    df_['min'] = np.round(df_['min'] * 100, 2)
    df_['max'] = np.round(df_['max'] * 100, 2)
    df_['mean'] = np.round(df_['mean'] * 100, 2)
    df_['std'] = np.round(df_['std'] * 100, 2)

    df_ = df_.sort_values(by=[group, 'metric'])
    df_mean = df_.copy()

    # Save results
    fname = '/pre_bs_' + group.lower() + '_summary.csv'
    df_.to_csv(csv_dir + fname)

    string = '{0[mean]:.' + str(formatting) + 'f} ± {0[std]:.' + str(formatting) + 'f}'
    df_mean['avg + std'] = df_mean.agg(string.format, axis=1)
    df_mean = df_mean.pivot(values='avg + std', columns='metric')

    return df_mean


def bootstrapping(n_boots, df, metric, group):
    """
    :param n_boots: number of bootstraps
    :param df: dataframe with runs results data
    :param metric: name of column metric to evaluate
    :param group: the dataframe column to group by
    :return: a dictionary with the mean of samples retrieved for 
        each bootstrap by group
    """

    # Extracting values and groups
    data = [df.loc[ids, metric].values
            for ids in df.groupby(group).groups.values()]
    keys = list(df.groupby(group).groups.keys())
    boots_ = dict()
    n = len(keys)
    for i in range(n):
        k = keys[i]
        v = data[i]
        exp_ = np.zeros(n_boots)
        # size of the sample equal to the original size
        n_ = len(v)
        # bootstrap cycle
        for b in range(n_boots):
            idx = np.random.randint(0, n_, n_)
            # bootstrap sample
            sample = np.take(v, idx)
            stat_mean = round(np.mean(sample), 4)
            exp_[b] = stat_mean
        boots_[k] = exp_

    return boots_


def np_confidence_interval(bootstraps, metric, alpha=5.0, ascending=False):
    """
    :param bootstraps: dictionary with the bootstrapping results
    :param metric: a string metric to compute the confidence interval
    :param alpha: critical values, significance level to build the 
        confidence interval
    :param ascending: order the ranks in ascending or descending order
    :return: a dictionary containing the confidence boundaries and
    median for each of the metrics
    """
    interval = dict()
    lower_per = alpha / 2.0
    upper_per = 100 - lower_per

    rank_df = pd.DataFrame.from_dict(bootstraps)
    # if ascending is false higher values are given lower ranks
    ranks = rank_df.rank(axis='columns', ascending=ascending)
    ranks_ = round(ranks.mean(), 2)

    for k, v in bootstraps.items():
        v = np.sort(v)
        median = round(np.median(v), 4) * 100
        lower = round(np.percentile(v, lower_per), 4) * 100
        upper = round(np.percentile(v, upper_per), 4) * 100
        interval[k] = median, lower, upper

    ci = pd.DataFrame.from_dict(interval, orient='index')
    ci['metric'] = metric
    ci['rank'] = ranks_

    return ci


def bootstrap_stats_summary(group, metrics, df, boots, out_dir, csv_dir, alpha=5.0, hideplot=False,
                            nemenyi=True, title=True):
    """
    :param group: column name to group by, either Architecture or Experiment
    :param metrics: list of metrics to compute the statistics
    :param df: dataframe of the raw data.
    :param boots: number of bootstrap cycles to achieve alpha confidence level
    :param out_dir: a string, directory to save figures/plots
    :param csv_dir: a string, directory to save the csv files
    :param alpha: critical values, significance level
    :param hideplot: boolean displays or not
    plots and figures.
    :param nemenyi: if true Nemenyi pairwise comparison, else compute Conover comparison.
    :param title: a boolean to add or not a title to the plot
    :return: dataframes for the rankings, confidence intervals
    and a dictionary with the post-hoc probabilities
    """

    summary = list()
    post_hoc = dict()

    for m in metrics:
        bootstraps = bootstrapping(boots, df, m, group)
        df_ = np_confidence_interval(bootstraps, m, alpha)
        st, p, fn = friedman_comparing_test(bootstraps, nemenyi)
        print('........ {0} and {1}.........'.format(group, m))
        summary.append(df_)
        post_hoc[m] = [st, p, fn]
    multiple_comparisons(post_hoc, out_dir, group, hideplot, title)
    results = pd.concat(summary)
    results = results.rename(columns={0: 'Median', 1: 'CI - lower',
                                      2: 'CI - upper', 'metric': 'Metric',
                                      'rank': 'Rank'})
    results = results[['Metric', 'Rank', 'Median', 'CI - lower', 'CI - upper']]
    results = results.rename_axis(group)
    res_b = results.sort_values(by=[group, 'Metric'])
    boots_summ = '/boots_' + group.lower() + '_summary.csv'
    res_b.to_csv(csv_dir + boots_summ)
    ranks, c_int = rank_confidence(res_b)

    return ranks, c_int, post_hoc


def friedman_comparing_test(bootstraps, nemenyi=True):
    """    
    :param bootstraps: a dictionary with bootstraps means.
    :param nemenyi: a boolean for Nemenyi or Conover comparison post-hoc test.
    :return:  a dataframe with the comparison of the bootstrapped groups, p-values
    and Nemenyi or Conover statistic.
    """

    b_list = list(bootstraps.values())
    stats, p = friedmanchisquare(*b_list)

    # post-hoc multiple comparison using nemenyi-friedman distance
    if nemenyi:
        f_n = sp.posthoc_nemenyi_friedman(np.array(b_list).T)
    else:
        f_n = sp.posthoc_conover_friedman(np.array(b_list).T)

    f_n.columns = list(bootstraps.keys())
    f_n.index = list(bootstraps.keys())

    return stats, p, f_n


def bootstrapping_epochs(n_boots, df, epoch_col, acc_col, group):
    """
    :param n_boots: number of bootstraps
    :param df: dataframe with the metrics
    :param epoch_col: name of the column with the num of epochs
    :param acc_col: name of the column with the val acc
    :param group: name of the column to group by
    :return: two dictionaries with the mean of samples retrieved for 
        each bootstrap for the maximum validation accuracy and no of epochs
    """

    # Extracting values and groups
    data = [df.loc[ids, epoch_col].values
            for ids in df.groupby(group).groups.values()]

    data1 = [df.loc[ids, acc_col].values
             for ids in df.groupby(group).groups.values()]

    keys = list(df.groupby(group).groups.keys())
    boots_ = dict()
    boots_ac = dict()

    n = len(keys)
    np.random.seed(123)
    for i in tqdm(range(n)):
        k = keys[i]
        # epochs data
        v = data[i]
        # accuracy data
        v_ = data1[i]
        ep = np.zeros(n_boots)
        acc = np.zeros(n_boots)
        # size of the sample equal to the original size
        n_ = len(v)
        # bootstrap cycle
        for b in range(n_boots):
            idx = np.random.randint(0, n_, n_)
            # bootstrap sample
            sample = np.take(v, idx)
            sample_ = np.take(v_, idx)
            mode_ = stat.mode(sample)
            mean = round(np.mean(sample_), 4)
            ep[b] = mode_
            acc[b] = mean
        boots_[k] = ep
        boots_ac[k] = acc

    return boots_, boots_ac


def boots_epochs_df(ep_dic, acc_dic, group, out_dir, csv_dir,
                    box_plot=True, hideplot=False, nemenyi=True):
    """
    :param ep_dic: dictionary containing the bootstraps results  for the no of epoch
    :param acc_dic: dictionary containing the bootstraps results  for the no of epoch
    :param group: string of the name of the column to group by
    :param out_dir: directory to save the dataframe
    :param csv_dir: directory to save the csv files
    :param box_plot: a boolean to create or not a boxplot
    :param hideplot: a boolean to display all figures produced
    :param nemenyi: a boolean, if true compare groups by using Nemenyi stats else computes Conover
    :return: a dataframe with bootstrapped validation accuracy and number of epochs, a dictionary
    with probabilities, and a dataframe with the confidence intervals
    """
    lower_per = 5.0 / 2.0
    upper_per = 100 - lower_per
    ep_prob = dict()

    for k, v in acc_dic.items():
        acc_dic[k] = acc_dic[k] * 100
        lower = round(np.percentile(v, lower_per), 4) * 100
        upper = round(np.percentile(v, upper_per), 4) * 100
        ep_prob[k, 'ac-ci'] = lower, upper

    for k, v in ep_dic.items():
        v = np.sort(v)
        lower = np.percentile(v, lower_per)
        upper = np.percentile(v, upper_per)
        ep_prob[k, 'ep-ci'] = lower, upper

    st_acc, p_acc, fn_acc, = friedman_comparing_test(acc_dic, nemenyi)
    st_ep, p_ep, fn_ep, = friedman_comparing_test(ep_dic, nemenyi)

    acc_df = pd.DataFrame.from_dict(acc_dic)
    ep_df = pd.DataFrame.from_dict(ep_dic)

    if box_plot:
        boxplot_epochs(acc_df, ep_df, group, out_dir, hideplot)

    ranks_ep = ep_df.rank(axis='columns', ascending=True)
    ranks_acc = acc_df.rank(axis='columns', ascending=False)

    # Median for epochs and max val accuracy
    m_ep = ep_df.median()
    m_acc = acc_df.median()
    # Get ranks
    ranks_ep = round(ranks_ep.mean(), 2)
    ranks_acc = round(ranks_acc.mean(), 2)

    ranks = pd.concat([ranks_acc, m_acc, ranks_ep, m_ep], axis=1)
    ranks = ranks.rename(columns={0: 'Rank Accuracy', 1: 'Median acc',
                                  2: 'Rank Epoch', 3: 'Median epoch'})
    ranks['Median epoch'] = pd.to_numeric(ranks['Median epoch'],
                                          downcast='integer')
    # Pairwise comparison
    probs_heatmap(fn_ep, out_dir, group, 'Epochs', 'epochs', hideplot=hideplot)
    probs_heatmap(fn_acc, out_dir, group, 'Maximum accuracy', 'max_acc', hideplot=hideplot)

    fname = '/max_' + group.lower() + '_summary.csv'
    ranks.to_csv(csv_dir + fname)

    # Save probabilities and statistic
    ep_prob['Max accuracy'] = [st_acc, p_acc]
    ep_prob['Epochs'] = [st_ep, p_ep]

    b_acc = acc_df.melt(var_name=group, value_name='max acc')
    b_ep = ep_df.melt(var_name=group, value_name='epochs')
    ep_sum = pd.concat([b_ep, b_acc['max acc']], axis=1)

    return ranks, ep_prob, ep_sum
