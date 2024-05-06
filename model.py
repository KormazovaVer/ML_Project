# загружаем библиотеки
import glob
import time
from collections import defaultdict

import IPython.display
import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils import data
from torchvision import transforms
from torchvision.models import ResNet18_Weights


# разделим картинки на train и val в отношении 70 на 30 для каждого класса
data_dir = "data"
data_image_paths = glob.glob(f"{data_dir}/*/*.jpg")
data_image_labels = [path.split('/')[-1][-2] for path in data_image_paths]
train_files_path, val_files_path = train_test_split(
    data_image_paths,
    test_size=0.3,
    stratify=data_image_labels
)

input_size = 224

train_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(
    data_dir,
    transform=train_transform,
    is_valid_file=lambda x: x in train_files_path
)

val_dataset = torchvision.datasets.ImageFolder(
    data_dir,
    transform=val_transform,
    is_valid_file=lambda x: x in val_files_path
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    model,
    criterion,
    optimizer,
    train_batch_gen,
    val_batch_gen,
    scheduler,
    num_epochs
):

    """

    Функция для обучения модели и вывода лосса и метрики во время обучения.

    """

    history = defaultdict(lambda: defaultdict(list))

    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        start_time = time.time()

        # устанавливаем поведение dropout / batch_norm  в обучение
        model.train(True)

        # на каждой "эпохе" делаем полный проход по данным
        for X_batch, y_batch in train_batch_gen:
            # обучаемся на текущем батче
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch.long().to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += np.sum(loss.detach().cpu().numpy())
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            train_acc += np.mean(y_batch.cpu().numpy() == y_pred)

        # подсчитываем лоссы и сохраням в "историю"
        train_loss /= len(train_batch_gen)
        train_acc /= len(train_batch_gen)
        history['loss']['train'].append(train_loss)
        history['acc']['train'].append(train_acc)
        scheduler.step()

        # устанавливаем поведение dropout / batch_norm в режим тестирования
        model.train(False)

        # полностью проходим по валидационному датасету
        for X_batch, y_batch in val_batch_gen:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch.long().to(device))
            val_loss += np.sum(loss.detach().cpu().numpy())
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            val_acc += np.mean(y_batch.cpu().numpy() == y_pred)

        # подсчитываем лоссы и сохраням в "историю"
        val_loss /= len(val_batch_gen)
        val_acc /= len(val_batch_gen)
        history['loss']['val'].append(val_loss)
        history['acc']['val'].append(val_acc)

        IPython.display.clear_output()

    return model, history


batch_size = 128

# не забудем перемешать train
train_batch_gen = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
# валидационный датасет мешать не нужно, а точнее бессмысленно
# сеть на нём не обучается
val_batch_gen = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)


fine_tuning_model_sched = nn.Sequential()
fine_tuning_model_sched.add_module('resnet', torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))
# добавим новые слои для классификации для нашей конкретной задачи
fine_tuning_model_sched.add_module('relu_1', nn.ReLU())
fine_tuning_model_sched.add_module('fc_1', nn.Linear(1000, 512))
fine_tuning_model_sched.add_module('relu_2', nn.ReLU())
fine_tuning_model_sched.add_module('fc_2', nn.Linear(512, 87))
fine_tuning_model_sched = fine_tuning_model_sched.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(fine_tuning_model_sched.parameters(), lr=0.1)
# добавим scheduler ExponentialLR для уменьшения скорости обучения каждой группы параметров на гамму в каждую эпоху
scheduler = ExponentialLR(optimizer,
                          gamma=0.9)

clf_model, history = train(
    fine_tuning_model_sched, criterion, optimizer,
    train_batch_gen, val_batch_gen,
    scheduler,
    num_epochs=20
)