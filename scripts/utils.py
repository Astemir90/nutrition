import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error
import random
import pandas as pd

# Модель
class FoodCaloriePredictor(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        # Извлечение признаков из изображения
        self.cnn = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.cnn.classifier = nn.Identity()  # Убираем последний слой
        self.img_features = 1280  # Выход EfficientNet-B0

        # Размер эмбеддинга из ингредиентов
        self.text_features = text_encoder.embedding_dim

        # Размер признака веса
        self.weight_features = 1

        # Финальный классификатор
        self.fc = nn.Sequential(
            nn.Linear(self.img_features + self.text_features + self.weight_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, images, ingredient_embedings, weights):
        # Признаки изображения
        img_features = self.cnn(images)

        # Признаки ингредиентов
        ingr_features = ingredient_embedings

        # Признак веса
        weight_features = weights.unsqueeze(1)

        # Объединяем
        combined = torch.cat([img_features, ingr_features, weight_features], dim=1)

        # Предсказание
        output = self.fc(combined)
        return output.squeeze(-1)
    
# Для обучения
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(config, model, text_endcoder, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.L1Loss()  # MAE

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for images, ingredients, weights, targets in train_loader:
            images, weights, targets = images.to(device), weights.to(device), targets.to(device)
            
            # Эмбеддинги ингредиентов
            with torch.no_grad():
                ingr_embeddings = text_endcoder(ingredients).to(device)
            
            optimizer.zero_grad()
            outputs = model(images, ingr_embeddings, weights)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Валидация
        val_mae = evaluate(model, text_endcoder, val_loader, device)

        print(f'Epoch {epoch+1}/{config["epochs"]}, Train Loss: {train_loss/len(train_loader):.4f}, Val MAE: {val_mae:.4f}')

        if val_mae < config['target_mae']:
            print('Цель по MAE достигнута, модель сохранена')
            torch.save(model.state_dict(), config['model_save_path'])
            break
        elif epoch + 1 == config['epochs']:
            print('Цель по MAE не достигнута, требуется доработка модели')

        scheduler.step(val_mae)

def evaluate(model, text_encoder, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, ingredients, weights, targets in val_loader:
            images, weights, targets = images.to(device), weights.to(device), targets.to(device)
            ingr_embeddings = text_encoder(ingredients).to(device)
            outputs = model(images, ingr_embeddings, weights)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    mae = mean_absolute_error(all_targets, all_preds)
    return mae