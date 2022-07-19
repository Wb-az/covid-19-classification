
import torch


def train_one_epoch(model, device, train_loader, loss_fn, optimizer):
    """
    :param model: model to evealuate
    :param device: device to  store the tensor
    :param train_loader: loader with the tensor images
    :param loss_fn: objective funtion to minimise
    :param optimizer: optimizer to minimise the objective function
    :return: loss and accuracy on the train dataset
    """
    model = model.to(device)
    # Set the model to training mode
    model.train()

    train_loss, train_acc, = 0, 0

    # Process the images in batches
    for batch_idx, (images, labels, path) in enumerate(train_loader):
        
        # Use the CPU or GPU as appropriate
        images, labels = images.to(device), labels.to(device)

        # Set gradients to zero
        optimizer.zero_grad()

        # forward through the model layers
        output = model(images)

        # Compute the loss
        loss = loss_fn(output, labels)

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward()

        # Upgrade gradients
        optimizer.step()

        _, preds = torch.max(output, 1)
        train_acc += torch.sum(preds == labels).item()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    # return average loss  and accuracy per epoch
    return train_loss, train_acc


def evaluate(model, device, test_loader):
    """
    :param model: model to evaluate
    :param device: device for the tensor operations 'cuda' or 'cpu'
    :param test_loader: the loader with test set to evaluate
    :return: loss and accuracy on the validation/test dataset
    """

    # Set the model to evaluation mode
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
           
            # Get the predicted classes for this batch
            output = model(images)

            # Calculate the loss for this batch
            loss = criterion(output, labels)
            test_loss += loss.item()

            # Calculate the accuracy for this batch
            _, preds = torch.max(output, 1)
            test_acc += torch.sum(labels == preds).item()

    # Calculate the average loss and total accuracy for one epoch
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    # return average loses and accuracy for epoch
    return test_loss, test_acc