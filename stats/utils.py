
import os


def rank_confidence(df, formatting=2):
    """
    :param df: a data frame summarising the result of the bootstrapping
        for all metrics, the median and the rank
    :param formatting: number of digits/decimal to display
    :return: two dataframes one containing the confidence intervals per metric
        and another with the ranking and media per metric
    """
    ranks = df[['Metric', 'Rank', 'Median']].copy()
    ci = df[['Metric', 'CI - lower', 'CI - upper']].copy()
    ranks = ranks.pivot(values=['Rank', 'Median'], columns=['Metric'])
    format_string = '{0[CI - lower]:.' + str(formatting) + 'f} - {0[CI - upper]:.' + str(
        formatting) + 'f}'
    ci['CI'] = ci.agg(format_string.format, axis=1)
    ci = ci.pivot(values='CI', columns='Metric')

    return ranks, ci


def df_to_latex(filename, caption, label, df, outdir=None):
    """
    :param filename: a string to write the latex table
    :param caption: table caption - string
    :param label: label to identify the table - string
    :param df: the dataframe to convert to latex
    :param outdir: a string with the directory to save the latext table
    :return: returns a latex table
    """

    outdir = os.path.join(os.getcwd(), outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print('Directory created')

    n_columns = len(df.keys()) + 1
    file_name = filename + '.tex'
    with open(os.path.join(os.getcwd(), 'tables', file_name), 'w') as tf:
        tf.write(df.reset_index().to_latex(index=False, caption=caption,
                                           label=label, escape=False,
                                           column_format='l' * n_columns))
    return file_name
