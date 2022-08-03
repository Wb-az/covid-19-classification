from stats.non_parametric_stats import *
from stats.results_plots import *


def main(**params):
    """
    :param params: a dictionary to access the data runs, group the data by
    experiment or by network architecture, num of bootstrap samples, plot and
    save plots and dataframes in csv format. Besides, it provides a summary
    of the accuracies at training, validation and test. It also summarizes of
    all performance metrics, before and after bootstrapping. The post-hoc
    friedman-nemenyi test stats also are in the dictionary for all performance
    metrics at test(accuracy, balanced accuracy, F1, F2, MCC, sensibility, specificity).
    :return: a dictionary with the summary of the accuracy during train, validation and test
    results, evaluation metrics and probabilities.
    """
    # Raise an assertion if incorrect grouping
    assert params['group'] in ['Architecture', 'Experiment'], 'Group by Architecture or Experiment'

    # Save output in a dictionary
    outputs = {}

    # Read data
    df = pd.read_csv(os.path.join(params['root'], params['data']))

    # Sorting by run architecture, loss and optimizer
    df = df.sort_values(['Run', 'Architecture', 'Loss', 'Optimizer'])

    # Abbreviate architecture name for plotting purpose
    df = df.replace({'MobileNet-v3-large': 'MobileNet-v3'})

    # Rename columns
    df = df.rename(columns={'Exp': 'Experiment', 'Sensitivity': 'Sens',
                            'F1 macro': 'F1', 'Specificity': 'Spec', 'Max epoch': 'Epoch'})
    # Re-number experiment as Exp-xx
    df['Experiment'] = ['Exp-' + str(x).zfill(2) for x in df['Experiment']]

    #  Create the experimental set-up
    df_set = df[['Experiment', 'Architecture', 'Loss', 'Optimizer']][0:20].set_index('Experiment')
    df_set = df_set.sort_values(by='Experiment')
    outputs['exp_setup'] = df_set

    # Filter data by training, validation and test accuracies
    df_acc = df.rename(columns={'Accuracy': 'Test acc', 'val_acc': 'Val acc',
                                'tr_acc': 'Train acc'})

    # Create directories to save figures and csv files
    outdir = os.path.join(params['root'], 'figures', params['group'].lower())
    csv_dir = os.path.join(params['root'], 'csv_files', params['group'].lower())

    # Create directories if they don't exist
    dirs = [outdir, csv_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print('Directory created')

    # Dataframe summarising train,test and validation accuracies
    acc_sum = acc_summary(params['group'], params['acc'], df_acc, outdir, csv_dir,
                          params['boxplot'], params['hideplot'])
    # Save results in a  dictionary
    outputs['accuracies'] = acc_sum

    # Filtering to get data only for the metrics to be compared
    df_covid = df.filter(['Experiment', 'Architecture', 'Loss', 'Optimizer', 'Accuracy', 'BA',
                          'MCC', 'F1', 'F2', 'Sens', 'Spec', 'Max acc', 'Epoch'])
    df_covid = df_covid.rename(columns={'Accuracy': 'Acc'})

    # Provide a summary of all evaluation metrics before bootstrapping
    if params['stats_sum']:
        st_s = stats_summary(params['group'], params['metrics'], df_covid, csv_dir)
        outputs['pre_boots'] = st_s

    # Bootstrapping summary and post-hoc test
    ranks, intervals, post_hoc = bootstrap_stats_summary(params['group'], params['metrics'],
                                                         df_covid, params['n_bootstraps'],
                                                         outdir, csv_dir, params['alpha'],
                                                         params['hideplot'], params['nemenyi'])
    # Save into the dictionary
    outputs['ranks'], outputs['ci'] = ranks, intervals
    outputs['stats_comp'] = post_hoc

    # Bootstrap for maximum acc and epoch - training
    epochs, max_acc = bootstrapping_epochs(params['n_bootstraps'], df_covid,
                                           'Epoch', 'Max acc', params['group'])

    # boxplot for bootstrapped epoch with max accuracy and accuracy during validation
    mx_rank, ci_prob, epochs_acc = boots_epochs_df(epochs, max_acc, params['group'], outdir,
                                                   csv_dir, params['boxplot'],

                                                   params['hideplot'], nemenyi=params['nemenyi'])

    # Save into the dictionary max accuracy ranking, p-values and confidence intervals
    outputs['max_rank'], outputs['max_acc_stats'] = mx_rank, ci_prob

    # density distribution for bootstrapped max accuracy and number of epochs during validation
    if params['dist_plot']:
        for a in ['epochs', 'max acc']:
            max_acc_epoch_plot(epochs_acc, params['group'], col_name=a, outdir=outdir,
                               hideplot=params['hideplot'])

    # Extract the friedman statistic and associated p-value
    friedman_test = {k: v[0:2] for (k, v) in post_hoc.items()}

    # Update dictionary to include Friedman-Nemenyi stats for the maximum accuracy and the number
    # of training epochs
    friedman_test.update({k: v for k, v in ci_prob.items() if k == 'Max accuracy' or k == 'Epochs'})
    pval = pd.DataFrame.from_dict(friedman_test).T.reset_index()
    pval.columns = ['Metric', 'Friedman', 'p-value']

    # Put together all post-hoc stats and p-value into the dictionary
    outputs['pval'] = pval

    # save dataframe
    pval.to_csv(os.path.join(csv_dir, params['group'].lower() + '_pval.csv'))

    return outputs


if __name__ == '__main__':
    acc = ['Train acc', 'Val acc', 'Test acc']
    metrics = ['Acc', 'BA', 'F1', 'F2', 'MCC', 'Sens', 'Spec']

    args = {'root': '/Users/aze_ace/Documents/pythonProject/covid_project',
            'data': 'covid_10_runs.csv', 'acc': acc, 'stats_sum': True, 'group': 'Architecture',
            'metrics': metrics, 'n_bootstraps': 1000, 'alpha': 5.0, 'nemenyi': True,
            'hideplot': True, 'boxplot': True, 'dist_plot': False}

    sum_net = main(args**)

    args['group'] = 'Experiment'

    sum_exp = main(args**)

    prob_bar(os.path.join(args['root'], 'figures'))
