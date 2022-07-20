import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def training_stats(stats, model, hideplot=False):
    """
    :param stats: a dctionary with accuracy and los collected during training and validation
    :param model: astring with the name of the models
    :param hideplot: boolean to display the plots
    :return: plot stats from the training data
    """

    # Plot the mean loss per epoch
    fig1 = plt.figure(figsize=(8, 4))
    plt.plot(stats['train_loss'], label='train')
    plt.plot(stats['val_loss'], label='val')
    plt.xticks(range(len(stats['train_loss'])))
    title = 'Loss trend ' + model
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if hideplot:
        plt.close(fig1)
    else:
        plt.rcParams.update({'font.size': 10})
        plt.show(fig1)

    # Plot mean accuracy per epoch
    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(stats['train_acc'], label='train')
    plt.plot(stats['val_acc'], label='val')
    plt.xticks(range(len(stats['train_acc'])))
    title2 = 'Accuracy trend ' + model
    plt.title(title2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if hideplot:
        plt.close(fig2)
    else:
        plt.rcParams.update({'font.size': 10})
        plt.show(fig2)

    # return fig1, fig2


def imshow_images(image, cait=False):
    """
    :param image: image tensor to display
    :param cait: a boolean indicating if the model is CaiT
    :return: displays the images
    """

    image = image.numpy().transpose((1, 2, 0))
    if not cait:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
    image = image * std + mean
    image = np.clip(image, 0., 1.)
    plt.imshow(image)


def show_predictions(model, test_loader, class_names, device, cait=False, outdir=None):
    """
    :param model: model to evaluate
    :param test_loader: image tensors
    :param class_names: the name of the classes
    :param device: device to process/store the tensor images
    :param cait: a boolean to apply transformation to cait or the other models
    :param outdir: directory to save the inference
    :return: show predictions and probability scores
    """
    model = model.to(device)
    model.eval()
    images, labels, path = next(iter(test_loader))
    images.to(device)
    labels.to(device)
    out = model(images)
    _, preds = torch.max(out, 1)
    preds = np.squeeze(preds.numpy())
    prob = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, out)]

    fig = plt.figure(figsize=(8, 4))
    for i in np.arange(len(images)):
        fig.add_subplot(len(images) // 4, 4, i + 1, xticks=[], yticks=[])
        imshow_images(images[i], cait)
        # class_names.sort()
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'
        plt.xlabel(f'{class_names[labels[i]]}', color='blue')
        plt.ylabel("{0}, {1:.1f}%".format(class_names[preds[i]], prob[i] * 100.0), color=col)
    plt.tight_layout()

    if outdir is not None:
        fname = 'predictions.eps'
        return plt.savefig(os.path.join(outdir, fname), bbox_inches='tight', format='eps', dpi=300)


def show_misses(df_misses, class_names, n_samples, outdir=None):
    """
    :param df_misses: a data frame with the missclassified images
    :param class_names:  a list the class names
    :param n_samples:  num samples to show (a factor of four)
    :param outdir: a string with the directory to save the misses
    :return: paths to the missclassified images
    random selected from the df_misses
    shows missclassifications
    """

    assert n_samples % 4 == 0, 'Sample needs to be an even number factor of four'

    preds = df_misses['predicted_label']
    labels = df_misses['true_label']
    probs = df_misses['probability']
    path = df_misses['image_path']
    samples = np.random.choice(len(preds), n_samples, replace=False)
    paths = []
    fig = plt.figure(figsize=(8, 4))
    for i in range(len(samples)):
        file = path[samples[i]]
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224), 2)
        paths.append(path[samples[i]])
        fig.add_subplot(len(samples) // 4, 4, i + 1, xticks=[], yticks=[])
        plt.imshow(img)
        col = 'red'
        plt.xlabel(f'{class_names[labels[samples[i]]]}', color='blue')
        plt.ylabel(
            "{0}, {1:.1f}%".format(class_names[preds[samples[i]]], probs[samples[i]] * 100.0),
            color=col)
    plt.tight_layout()

    if outdir is not None:
        fname = 'misses.eps'
        plt.savefig(os.path.join(outdir, fname), bbox_inches='tight',
                    format='eps', dpi=300)
    return paths
