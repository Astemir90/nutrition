import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as T
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# Предобученная модель для текста
class PretrainedTextEncoder(nn.Module):
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2'):
        super().__init__()
        # Загружаем предобученную модель
        self.sentence_model = SentenceTransformer(model_name)
        self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
        
        # Замораживаем веса (не обучаем)
        for param in self.sentence_model.parameters():
            param.requires_grad = False

    def forward(self, ingredient_texts):
        embeddings = self.sentence_model.encode(ingredient_texts, convert_to_tensor=True)
        return embeddings

class FoodDataset(Dataset):
    def __init__(self, df, img_dir, text_encoder, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.text_encoder = text_encoder
        self.transform = transform

        # Нормализуем вес
        self.weight_mean = df['total_mass'].mean()
        self.weight_std = df['total_mass'].std()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row['dish_id']
        img_dir = os.path.join(self.img_dir, dish_id)
        
        # Находим фото в папке блюда
        img_files = [f for f in os.listdir(img_dir)]
        img_path = os.path.join(img_dir, img_files[0])
        
        # Загрузка изображения
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Ингредиенты
        ingredients = row['ingr_text']

        # Вес блюда (нормализуем)
        weight = (row['total_mass'] - self.weight_mean) / self.weight_std
        weight = torch.tensor(weight, dtype=torch.float32)
        
        # Целевая переменная
        calories = torch.tensor(row['total_calories'], dtype=torch.float32)
        
        return image, ingredients, weight, calories

def get_transforms():
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform