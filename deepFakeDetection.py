import os
import cv2
import json
import random
import shutil
import pathlib
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.models import VGG16_Weights

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from skopt import gp_minimize
from skopt.space import Real

from tqdm import tqdm  # Barre de progression
from tensorboard import program


def start_tensorboard(logdir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir])
    url = tb.launch()
    print(f"TensorBoard est en cours d'exécution à l'adresse {url}")


def __getitem__(self, idx):
    label = self.labels[idx]
    # Données d'image fictives sous forme de tableau NumPy
    image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)  # Simulation d'une image

    if self.transform:
        image = self.transform(Image.fromarray(image))  # Conversion en image PIL

    return image, label


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, frames_per_video=16):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frames_per_video = frames_per_video

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Chargement de plusieurs images de la vidéo
        frames = self.load_video_frames(video_path, self.frames_per_video)

        # Application des transformations aux images
        if self.transform:
            frames = [self.transform(Image.fromarray(frame)) for frame in frames]

        # Empilage des images pour former une séquence
        frames = torch.stack(frames)
        return frames, label

    def load_video_frames(self, video_path, num_frames):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Erreur lors de l'ouverture de la vidéo : {video_path}")
            return []  # Retourne une liste vide si la vidéo ne peut pas être ouverte

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=np.int32)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        cap.release()

        # S'il n'y a pas assez d'images, répétition de la dernière image
        if len(frames) < num_frames:
            frames += [frames[-1]] * (num_frames - len(frames))

        return frames


class CustomAdamOptimizer:
    def __init__(self, params, lr=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.params = list(params)
        self.m = [torch.zeros_like(param) for param in self.params]  # Initialisation du premier moment biaisé
        self.v = [torch.zeros_like(param) for param in self.params]  # Initialisation du second moment biaisé
        self.t = 0  # Pas temporel

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            # Ignore si le paramètre ne requiert pas de gradients
            if not param.requires_grad:
                continue

            grad = param.grad  # Obtention du gradient du paramètre

            # Mise à jour du premier moment biaisé
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Mise à jour du second moment biaisé
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            # Calcul du premier moment corrigé
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            # Calcul du second moment corrigé
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Mise à jour du paramètre avec la règle Adam
            param.data = param.data - self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


def balance_dataset(original_videos_dir, manipulated_videos_dir, output_dir, target_count=75):

    # Crée des répertoires de sortie
    balanced_original_dir = output_dir / "original"
    balanced_manipulated_dir = output_dir / "manipulated"
    balanced_original_dir.mkdir(parents=True, exist_ok=True)
    balanced_manipulated_dir.mkdir(parents=True, exist_ok=True)

    # Échantillonnage les vidéos
    original_videos = list(original_videos_dir.glob("*.mp4"))
    manipulated_videos = list(manipulated_videos_dir.glob("*.mp4"))

    sampled_original = random.sample(original_videos, target_count)
    sampled_manipulated = random.sample(manipulated_videos, target_count)

    # Copie les vidéos échantillonnées dans les répertoires de sortie
    for file in sampled_original:
        shutil.copy(file, balanced_original_dir / file.name)

    for file in sampled_manipulated:
        shutil.copy(file, balanced_manipulated_dir / file.name)

    print(f"Jeu de données équilibré créé avec {target_count} vidéos dans chaque classe.")
    return (sampled_original, [0] * len(sampled_original)), (sampled_manipulated, [1] * len(sampled_manipulated))


class VGG16LSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=256, lstm_num_layers=1, freeze_feature_extractor=True):
        super(VGG16LSTM, self).__init__()

        # Charger les caractéristiques du VGG16 pré-entraîné
        self.feature_extractor = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # VGG16 utilise par défaut un pool (7x7)
        self.fc_features = nn.Linear(512 * 7 * 7, 1024)

        # Optionnel : geler l'extracteur de caractéristiques
        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # LSTM pour la modélisation temporelle
        self.lstm = nn.LSTM(input_size=1024, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)

        # Couche entièrement connectée pour la classification
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        # Redimensionner l'entrée pour l'extracteur de caractéristiques
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)

        # Pooling et aplatissage
        features = self.avgpool(features)
        features = torch.flatten(features, start_dim=1)
        features = self.fc_features(features)

        # Redimensionnement pour l'entrée du LSTM
        features = features.view(batch_size, seq_len, -1)

        # Aplatissage des poids du LSTM (nécessaire pour CuDNN)
        self.lstm.flatten_parameters()

        # Passage avant à travers le LSTM
        lstm_out, _ = self.lstm(features)

        # Prend la sortie du dernier pas temporel
        final_output = lstm_out[:, -1, :]

        # Passage avant à travers le classifieur
        output = self.fc(final_output)
        return output


# Entraînement et validation
def train_and_validate(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()  # Passe le modèle en mode entraînement
        running_loss = 0.0

        # Boucle d'entraînement avec barre de progression
        print(f"Époque {epoch + 1}/{epochs}")
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc="Entraînement")

        for batch_idx, (videos, labels) in train_progress:
            videos, labels = videos.to(device), labels.to(device)

            # Passage avant
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)

            # Rétropropagation et mise à jour de l'optimiseur
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())

        print(f"Époque {epoch + 1} Perte d'entraînement : {running_loss / len(train_loader):.4f}")

        # Boucle de validation
        model.eval()  # Passe le modèle en mode évaluation
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_progress = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
            for batch_idx, (videos, labels) in val_progress:
                videos, labels = videos.to(device), labels.to(device)

                # Passage avant
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calcul de la précision
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Époque {epoch + 1} Perte de validation : {val_loss / len(val_loader):.4f}")
        print(f"Époque {epoch + 1} Précision de validation : {100 * correct / total:.2f}%\n")


def evaluate_model(model, val_loader):
    y_true = []
    y_pred = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Collecte des prédictions et des vraies étiquettes
    model.eval()
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    # Calcul des métriques

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    cm = np.array([[tp, fn], [fp, tn]])

    return cm, acc, precision, recall, f1


def plot_confusion_matrix(cm, labels=["Positif", "Négatif"]):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Positif", "Négatif"],
                yticklabels=["Positif", "Négatif"])
    plt.ylabel("Valeurs prédites")
    plt.xlabel("Valeurs réelles")
    plt.title("Matrice de confusion")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    dataset_dir = pathlib.Path('/kaggle/input/deep-fake-detection-dfd-entire-original-dataset')
    original_videos = dataset_dir / "DFD_original sequences"
    manipulated_videos = dataset_dir / "DFD_manipulated_sequences/DFD_manipulated_sequences"

    num_original_videos = len(list(original_videos.glob("*.mp4")))
    num_manipulated_videos = len(list(manipulated_videos.glob("*.mp4")))
    print(f"Vidéos originelles : {num_original_videos}")
    print(f"Vidéos DeedFake : {num_manipulated_videos}")

    writer = SummaryWriter(log_dir="runs/cross_validation")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_original_videos = len(list(original_videos.glob("*.mp4")))  # Modification de l'extension du fichier si nécessaire
    num_manipulated_videos = len(list(manipulated_videos.glob("*.mp4")))

    print(f"Vidéos originelles : {num_original_videos}")
    print(f"Vidéos DeedFake : {num_manipulated_videos}")

    # Chemins
    balanced_dir = pathlib.Path('./balanced_dataset')
    balanced_samples = balance_dataset(original_videos, manipulated_videos, balanced_dir)

    balanced_dir = pathlib.Path('./balanced_dataset')
    balanced_original = balanced_dir / "original"
    balanced_manipulated = balanced_dir / "manipulated"

    # S'assure que les répertoires de sortie existent
    balanced_original.mkdir(parents=True, exist_ok=True)
    balanced_manipulated.mkdir(parents=True, exist_ok=True)

    # Mise à jour des chemins du jeu de données pour pointer vers le nouveau sous-ensemble
    balanced_video_paths = []
    balanced_labels = []

    for video_path in balanced_original.glob("*.mp4"):
        balanced_video_paths.append(video_path)
        balanced_labels.append(0)  # Étiquette 0 pour les vidéos originelles

    for video_path in balanced_manipulated.glob("*.mp4"):
        balanced_video_paths.append(video_path)
        balanced_labels.append(1)  # Étiquette 1 pour les vidéos DeepFake

    print(f"Dataset équilibré créé avec 350 vidéos dans chaque classe.")
    print(f"Vidéos originelles enregistrées dans : {balanced_original}")
    print(f"Vidéos DeepFake enregistrées dans : {balanced_manipulated}")

    balanced_original_dir = pathlib.Path('./balanced_dataset/original')
    balanced_manipulated_dir = pathlib.Path('./balanced_dataset/manipulated')

    original_count = len(list(balanced_original_dir.glob("*.mp4")))
    manipulated_count = len(list(balanced_manipulated_dir.glob("*.mp4")))
    print(f"Nombre de vidéos originelles : {original_count}")
    print(f"Nombre de vidéos DeepFake : {manipulated_count}")

    balanced_dataset = VideoDataset(balanced_video_paths, balanced_labels, transform=transform)

    # Validation croisée en 3 plis
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_splits = [(train_idx, val_idx) for train_idx, val_idx in kf.split(balanced_video_paths)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialisation du modèle, de la fonction de perte et de l'optimiseur
    model = VGG16LSTM(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = CustomAdamOptimizer(model.parameters(), lr=0.0001)

    metrics = {
        "exactitude": [],
        "précision": [],
        "rappel": [],
        "score_f1": []
    }

    # Division du jeu de données
    train_idx = list(range(0, 120))  # Ajustement selon le jeu de données
    val_idx = list(range(120, 150))
    train_dataset = Subset(balanced_dataset, train_idx)
    val_dataset = Subset(balanced_dataset, val_idx)
    # Chargeurs de données (DataLoaders)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Vérifie si ces répertoires contiennent des fichiers vidéo
    print(len(list(original_videos.glob("*.mp4"))))
    print(len(list(manipulated_videos.glob("*.mp4"))))
    print(len(train_dataset))
    print(len(val_dataset))

    # Boucle de validation croisée
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\nPli {fold_idx + 1} :")

        # Préparation des DataLoaders pour le pli
        train_dataset = Subset(balanced_dataset, train_idx)
        val_dataset = Subset(balanced_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

        # Initialisation du modèle et de l'optimiseur pour chaque pli
        model = VGG16LSTM(num_classes=2).to(device)
        model.lstm.flatten_parameters()
        optimizer = CustomAdamOptimizer(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        # Entraînement et validation pour le pli
        train_and_validate(model, train_loader, val_loader, epochs=10)

        # Évaluation des métriques finales pour le pli
        cm, acc, precision, recall, f1 = evaluate_model(model, val_loader)
        plot_confusion_matrix(cm)

        print(f"Métriques finales pour le pli {fold_idx + 1} - Exactitude : {acc:.4f}, Précision: {precision:.4f}, Rappel : {recall:.4f}, Score F1: {f1:.4f}\n")
        # Évaluation après pli
        writer.add_scalar(f"Pli_{fold_idx+1}/Exactitude", acc, fold_idx + 1)
        writer.add_scalar(f"Pli_{fold_idx+1}/Précision", precision, fold_idx + 1)
        writer.add_scalar(f"Pli_{fold_idx+1}/Rappel", recall, fold_idx + 1)
        writer.add_scalar(f"Pli_{fold_idx+1}/Score_F1", f1, fold_idx + 1)
        writer.add_figure(f"Pli_{fold_idx+1}/Matrice_Confusion", plt.gcf(), fold_idx + 1)

    torch.save(model.state_dict(), '/kaggle/working/vgg16_best_model.pth')

    writer.close()
    start_tensorboard("runs/cross_validation")
