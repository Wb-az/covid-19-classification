import os
import torch
import torchvision
import time
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from covid_class_dataset import *
from train_class import get_classification_model, main
from plotting import training_stats, show_predictions, show_misses
from evaluation_metrics import predictions, Metrics, misclassification, train_metrics


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

f_path = '/content/drive/MyDrive/INM363_classification/classification/curated_data.zip'

dest_path = '/content/drive/MyDrive/INM363_classification/classification'

with zipfile.ZipFile(f_path, 'r') as zip_ref:
    zip_ref.extractall(dest_path)


def covid_data(data_path, classes):
    data = []
    data_id = {}
    paths = {}
    for ii, name, in enumerate(classes):
        path = os.path.join(data_path, name)
        l_dir = sorted(os.listdir(path))
        data_id[name] = l_dir
        paths[name] = path
        for file in l_dir:
            file_path = os.path.join(path, file)
            data.append([name, file_path, file])

    return data, data_id


def test_evaluation(test_df, classes, **params_dict):

    if params_dict['model'] != 'cait':

        dataset = CovidDataset(test_df, 'image_path', 'image_label', get_transform(train=False))
    else:
        dataset = CovidDataset(test_df, 'image_path', 'image_label', get_vt_transform(train=False))

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=params_dict['batch_size'],
                                              shuffle=True)
                                              
    w_name = 'model_final_' + str(params['epochs']) + '.pth'                                          
    checkpoint = torch.load(os.path.join(params_dict['output_dir'], w_name))
    train_stats = checkpoint['stats']
    tr_acc, val_acc, max_acc, max_index, val_loss, min_loss, loss_idx = train_metrics(**train_stats)
    training_stats(params_dict['model'], hideplot=False, **train_stats)
    
    model_ = get_classification_model(params_dict['model'], params_dict['num_classes'],
                                      pretrained=False)
    model_.load_state_dict(checkpoint['model'], strict=False)
    
    y_test, y_pred, y_prob, y_prob_tensor = predictions(model_.to(device), test_loader, device)
    metrics = Metrics(y_test, y_pred, y_prob_tensor, classes)

    print('Model:', params_dict['model'], '|', 'run:', run, '|', 'loss:', params_dict['loss'], '|',
          'Optimizer:',
          params_dict['optimizer'])
    print('Classification report', metrics.classification_rep(), sep='\n')

    macro_prec, recall, f1, f2, f1_class, f2_class = metrics.macro_average()
    micro_prec, micro_rec, micro_f1 = metrics.micro_average()
    sens, spec, (tp, tn, fp, fn) = metrics.sensitivity_specificity()

    metrics_ = [['Date', date.today()], ['Exp', params_dict['exp']], ['Run', params_dict['run']],
                ['Architecture', params_dict['model']], ['Loss', params_dict['loss']],
                ['Optimizer', params_dict['optimizer']], ['Accuracy', micro_f1],
                ['BA', metrics.balanced_acc()],
                ['MCC', metrics.mcc()],
                ['F1 macro', f1],
                ['Sensitivity', recall],
                ['Precision', macro_prec],
                ['Specificity', round(np.mean(spec), 4)],
                ['F2', f2], ['AUROC', metrics.auroc()],
                ['F1 per class', f1_class],
                ['F2 per class', f2_class],
                ['Sensitivity per class', sens], ['Specificity per class', spec],
                ['Misses', fp + fn], ['FN', fn], ['FP', fp],
                ['tr_acc', tr_acc], ['val_acc', val_acc],
                ['Max acc', max_acc], ['Max epoch', max_index + 1],
                ['val_loss', val_loss], ['best loss', float(min_loss)],
                ['loss_idx', loss_idx + 1]]

    df_metrics = pd.DataFrame(metrics_, columns=['Metric', 'Value'],
                              index=[ii for ii in range(len(metrics_))])

    save_f = 'exp_' + str(params_dict['exp']) + '_run_' + run + '.csv'
    outdir = params_dict['output_dir']
    df_metrics.to_csv(os.path.join(outdir, save_f))
    
    dft = df_metrics.T
    dftm = dft.rename(columns=dft.iloc[0])
    fname = '/dftranspose.csv'
    outdir = params['output_dir']
    dftm.to_csv(outdir + fname)

    return model_, test_loader, y_test, y_pred, y_prob


if __name__ == '__main__':

    data_dir = os.path.join(os.getcwd(), 'curated_data')
    class_names = ['cap', 'covid', 'non_covid']

    data_dict, counts_dict = covid_data(data_dir, class_names)

    for k, v in counts_dict.items():
        print(f'Directory {k}, contains {len(counts_dict[k])}, instances')

    data_table = pd.DataFrame(data_dict, columns=['image_label', 'image_path', 'image_id'])

    exps = [i for i in range(1, 21)]

    exp_models = sorted(['resnet50', 'resnet50r', 'densenet121', 'mobilenet_v3_l',
                         'cait_24_224'] * 4)
    exp_optim = ['Adam', 'AdamW'] * 10
    exp_loss = []
    for i in range(1, 21, 4):
        if i and i + 1 in exps:
            exp_loss.extend(['CE'] * 2)
        if i + 2 and i + 3 in exps:
            exp_loss.extend(['wCE'] * 2)

    setup = pd.DataFrame({'Exp': exps, 'Net': exp_models, 'Loss': exp_loss, 'Optim': exp_optim})

    today = date.today()
    date2 = today.strftime('%B %d, %Y')

    train, val, test = split_dataset(data_table, label_col='image_label', seed=123)
    print('Train images:', len(train))
    print('Validation images:', len(val))
    print('Test images: ', len(test))

    exp = 16
    run = '01'

    # Extract the architecture, optimizer and loss from the setup table using index for exp 16
    idx = setup.Exp[setup.Exp == exp].index[0]
    model = setup.iloc[idx]['Net']
    optim = setup.Optim.iloc[idx]
    loss = setup.Loss.iloc[idx]

    # Creating weights to input onto the wCE
    class_count = {}
    for i, lb in enumerate(sorted(np.unique(train.image_label))):
        class_count[i] = train.image_label.value_counts()[lb]

    torch.set_printoptions(precision=5)
    weights = 1 / torch.tensor(list(class_count.values()))
    weights.to(device)

    # Output directory
    output_dir = os.path.join(model, 'exp_' + str(exp), run)
    try:
        os.makedirs(output_dir, exist_ok=False)
        print('Directory successfully created')
    except OSError as error:
        print('Directory already exist')

    if model != 'cait_24_224':
        train_dataset = CovidDataset(train, 'image_path', 'image_label',
                                     get_transform(train=True))
        val_dataset = CovidDataset(val, 'image_path', 'image_label',
                                   get_transform(train=False))
    else:
        train_dataset = CovidDataset(train, 'image_path', 'image_label',
                                     get_vt_transform(train=True))
        val_dataset = CovidDataset(val, 'image_path', 'image_label',
                                   get_vt_transform(train=False))

    params = {'exp': exp, 'model': model, 'run': run, 'num_classes': 3, 'device': device,
              'train_dataset': train_dataset, 'val_dataset': val_dataset, 'batch_size': 8,
              'workers': 2, 'loss': loss, 'weights': weights, 'optimizer': optim, 'lr': 2e-5,
              'momentum': 0.9, 'weight_decay': 1e-4, 'epochs': 8, 'output_dir': output_dir}

    print("Today's date:", date2)
    t = time.localtime()
    current_time = time.strftime('%H:%M:%S', t)
    print('Current_time', current_time)
    print('Model:', model, '|', 'run:', run, '|', 'loss:', loss, '|', 'optimizer:',
          optim)
    stats_log = main(**params)

    test_model, loader, y_true, y_predict, y_proba = test_evaluation(test, class_names, **params)

    # Show predictions- change cait to True if model is CaiT
    fig = plt.figure(figsize=(20, 20))

    if params['model'] == 'cait':
        cait = True
    else:
        cait = False
    show_predictions(test_model, loader, class_names, device=torch.device('cpu'), cait=cait,
                     outdir=None)

    # Show misses and save a csv with name of the incorrect file
    misses_data = misclassification(y_true, y_predict, y_proba, test)
    misses_data.to_csv(output_dir + '/misses.csv')
    show_misses(misses_data, class_names, 8)

    torch.cuda.empty_cache()
