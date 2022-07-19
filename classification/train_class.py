import os
import time
import torch
import torchvision.models as models
import torch.nn as nn
import timm
import math
from tqdm import tqdm
# from torchvision import transforms as T
# from PIL import Image
from collections import defaultdict
from utils import train_one_epoch, evaluate


def time_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_classification_model(model, num_classes, pretrained=True):
    """
    :param model: a string with the name of the architecture to load from PyTorch
    :param num_classes: number of classes in the dataset
    :param pretrained: a boolean indicating if the model is pretrained or not
    :return: a model with customised classifier for the num_classes
    """
    if model == 'resnet50':
        
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        # replace the pre-trained classifier with a new one
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)
        print(model.fc)

    elif model == 'resnet50r':
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        # original kernel size (7,7), replaced by size (5,5)
        # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.conv1 = torch.nn.Conv2d(3, 64, (5, 5), (2, 2), (3, 3), bias=False)
        # replace the pre-trained classifier with a new one
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)
        print(model.fc)

    elif model == 'densenet121':
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        # replace the pre-trained classifier with a new one
        model.classifier = nn.Linear(1024, out_features=num_classes)
        print(model.classifier)

    elif model == 'mobilenet_v3_l':
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
        # replace the pre-trained classifier with a new one
        model.classifier[3] = nn.Linear(1280, out_features=num_classes)
        print(model.classifier[3])

    elif model == 'cait_24_224':
        model = timm.create_model('cait_xxs24_224', pretrained)
        # replace the pre-trained head with a new one
        model.head = nn.Linear(in_features=192, out_features=num_classes)
        print(model.head)

    return model


def main(params_dict):
    """
    :param params_dict: a dictionary with the training hyperparameters
    :return: save weigths of the trained model, plots training and validation accuracies and loss
    """
    device = params_dict['device']

    print('...... Creating model', params_dict['model'], '......')

    # Create the correct model type
    model = get_classification_model(params_dict['model'], params_dict['num_classes'])

    # Move model to the correct device
    model = model.to(device)
    par = [p for p in model.parameters() if p.requires_grad]

    print("....... Creating train and validation dataloaders .......")
    # creates dataloaders

    train_loader = torch.utils.data.DataLoader(params_dict['train_dataset'],
                                               batch_size=params_dict['batch_size'], shuffle=True,
                                               num_workers=params_dict['workers'])
    val_loader = torch.utils.data.DataLoader(params_dict['val_dataset'],
                                             batch_size=round(params_dict['batch_size']),
                                             shuffle=True, num_workers=params_dict['workers'])

    if params_dict['loss'] == 'wCE':
        loss_fn = torch.nn.CrossEntropyLoss(params_dict['weights'].to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    if params_dict['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(par, lr=params_dict['lr'])

    elif params_dict['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(par, lr=params_dict['lr'],
                                      weight_decay=params_dict['weight_decay'])
    else:
        optimizer = torch.optim.SGD(par, lr=params_dict['lr'], momentum=params_dict['momentum'],
                                    weight_decay=params_dict['weight_decay'])

    # Start training
    best_acc = 0.0
    stats_log = defaultdict(list)
    print('..........Start training.........')

    epoch_start_time = time.time()

    for epoch in tqdm(range(params_dict['epochs'])):

        train_loss, train_acc = train_one_epoch(model, device, train_loader, loss_fn, optimizer)
        print('| End of epoch: {0:2d} | time: {1}'.format(epoch + 1,
                                                          time_minutes(
                                                              time.time() - epoch_start_time)))
        val_loss, val_acc = evaluate(model, device, val_loader)
        print('| Evaluating epoch: {0:2d} |'.format(epoch + 1))
        print('| train loss: {0:.4f} | train acc {1:.4f} | val loss: {2:.4f} | '
              'val_acc {3:.4f}'.format(train_loss, train_acc, val_loss, val_acc))

        checkpoint = {'epoch': epoch, 'model': model.state_dict(),
                      'optimizer_dict': optimizer.state_dict(), 'best_acc': best_acc,
                      'stats': stats_log,
                      'time': time.time() - epoch_start_time}

        stats_log['train_acc'].append(round(train_acc, 4))
        stats_log['train_loss'].append(round(train_loss, 4))
        stats_log['val_acc'].append(round(val_acc, 4))
        stats_log['val_loss'].append(round(val_loss, 4))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint,
                       os.path.join(params_dict['output_dir'], 'best_model.pth'))

        if epoch + 1 == params_dict['epochs']:
            total_time = time.time() - epoch_start_time
            print('Training time {}'.format(time_minutes(total_time)))
            # save final weights
            torch.save(checkpoint, os.path.join(params_dict['output_dir'],
                                                'model_final_{}.pth'.format(epoch + 1)))
    torch.cuda.empty_cache()
    return stats_log