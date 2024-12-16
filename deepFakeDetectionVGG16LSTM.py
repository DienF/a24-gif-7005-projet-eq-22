import os
import cv2
import json
import random
import shutil
import pathlib
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
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
from typing import Callable, Iterable, List, Optional, Tuple
from tensorboard import program
from tensorboard.program import TensorBoard


class VideoDataset(Dataset):
    """Jeu de données pour charger des vidéos et leurs étiquettes associées

    Cette classe gère le chargement des images de vidéos, l'application
    d'éventuelles transformations, et la préparation des séquences pour
    des modèles d'apprentissage profond.

    Paramètre(s)
    ----------
    video_paths : List[Path]
        Liste des chemins des fichiers vidéo.
    labels : List[int]
        Liste des étiquettes correspondant à chaque vidéo.
    transform : Optional[Callable], optional
        Transformation à appliquer aux images, par défaut None.
    frames_per_video : int, optional
        Nombre d'images à charger par vidéo, par défaut 16.
    """

    def __init__(self, video_paths: List[Path], labels: List[int], transform: Optional[Callable] = None, frames_per_video: int = 16) -> None:
        self.video_paths: List[Path] = video_paths
        self.labels: List[int] = labels
        self.transform: Optional[Callable] = transform
        self.frames_per_video: int = frames_per_video

    def __len__(self) -> int:
        """Retourne la taille du jeu de données."""
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Retourne une séquence d'images et l'étiquette associée.

        Paramètre(s)
        ----------
        idx : int
            Indice de l'échantillon à récupérer.

        Retourne
        -------
        Tuple[torch.Tensor, int]
            Séquence d'images sous forme de tenseur PyTorch et étiquette associée.
        """
        video_path: Path = self.video_paths[idx]
        label: int = self.labels[idx]
        # Chargement de plusieurs images de la vidéo
        frames: List[np.ndarray] = self.load_video_frames(video_path, self.frames_per_video)
        # Application des transformations aux images
        if self.transform:
            frames = [self.transform(Image.fromarray(frame)) for frame in frames]
        # Empilage des images pour former une séquence
        frames_tensor: torch.Tensor = torch.stack(frames)
        return frames_tensor, label

    def load_video_frames(self, video_path: Path, num_frames: int) -> List[np.ndarray]:
        """Charge un nombre spécifique d'images d'une vidéo.

        Paramètre(s)
        ----------
        video_path : Path
            Chemin de la vidéo.
        num_frames : int
            Nombre de images à charger.

        Retourne
        -------
        List[np.ndarray]
            Liste des images sous forme de tableaux NumPy (en RGB).
        """
        cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Erreur lors de l'ouverture de la vidéo : {video_path}")
            return []  # Retourne une liste vide si la vidéo ne peut pas être ouverte
        frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices: np.ndarray = np.linspace(0, frame_count - 1, num_frames, dtype=np.int32)
        frames: List[np.ndarray] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                break
        cap.release()
        # S'il n'y a pas assez d'images, répétition de la dernière image
        if len(frames) < num_frames:
            frames += [frames[-1]] * (num_frames - len(frames))
        return frames


class CustomAdamOptimizer:
    """Implémentation personnalisée de l'optimiseur Adam

    Cet optimiseur applique la méthode d'Adam pour mettre à jour les paramètres
    d'un modèle avec un apprentissage adaptatif basé sur les moments biaisés.

    Paramètre(s)
    ----------
    params : Iterable[Parameter]
        Liste des paramètres à optimiser.
    lr : float, optional
        Taux d'apprentissage, par défaut 0.0001.
    beta1 : float, optional
        Coefficient pour le premier moment biaisé, par défaut 0.9.
    beta2 : float, optional
        Coefficient pour le second moment biaisé, par défaut 0.999.
    epsilon : float, optional
        Petite constante pour éviter la division par zéro, par défaut 1e-8.
    """

    def __init__(self, params: Iterable[Parameter], lr: float = 0.0001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.lr: float = lr
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.params: List[Parameter] = list(params)
        self.m: List[torch.Tensor] = [torch.zeros_like(param) for param in self.params]  # Initialisation du premier moment biaisé
        self.v: List[torch.Tensor] = [torch.zeros_like(param) for param in self.params]  # Initialisation du second moment biaisé
        self.t: int = 0  # Pas temporel

    def step(self) -> None:
        """Met à jour les paramètres en utilisant la méthode Adam."""
        self.t += 1
        for i, param in enumerate(self.params):
            # Ignore si le paramètre ne requiert pas de gradients
            if not param.requires_grad:
                continue
            grad: torch.Tensor = param.grad  # Obtention du gradient du paramètre
            # Mise à jour du premier moment biaisé
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Mise à jour du second moment biaisé
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            # Calcul du premier moment corrigé
            m_hat: torch.Tensor = self.m[i] / (1 - self.beta1**self.t)
            # Calcul du second moment corrigé
            v_hat: torch.Tensor = self.v[i] / (1 - self.beta2**self.t)
            # Mise à jour du paramètre avec la méthode d'Adam
            param.data = param.data - self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

    def zero_grad(self) -> None:
        """Réinitialise tous les gradients des paramètres suivis."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class VGG16LSTM(nn.Module):
    """Combinaison d'un extracteur de caractéristiques basé sur VGG16 et d'un LSTM pour la classification vidéo

    Cette architecture utilise VGG16 pour extraire les caractéristiques spatiales de chaque image,
    suivie d'un LSTM pour capturer les dépendances temporelles dans une séquence d'images.

    Parameters
    ----------
    num_classes : int, optional
        Nombre de classes pour la tâche de classification, par défaut 2.
    lstm_hidden_size : int, optional
        Taille de la couche cachée du LSTM, par défaut 256.
    lstm_num_layers : int, optional
        Nombre de couches dans le LSTM, par défaut 1.
    freeze_feature_extractor : bool, optional
        Si True, gèle les poids de l'extracteur de caractéristiques VGG16, par défaut True.

    Attributes
    ----------
    feature_extractor : nn.Module
        Extracteur de caractéristiques basé sur VGG16.
    avgpool : nn.AdaptiveAvgPool2d
        Pooling adaptatif pour réduire les dimensions des caractéristiques.
    fc_features : nn.Linear
        Couche linéaire pour réduire les dimensions des caractéristiques avant le LSTM.
    lstm : nn.LSTM
        Réseau LSTM pour capturer les dépendances temporelles.
    fc : nn.Linear
        Couche linéaire pour la classification finale.
    """

    def __init__(self, num_classes: int = 2, lstm_hidden_size: int = 256, lstm_num_layers: int = 1, freeze_feature_extractor: bool = True) -> None:
        super(VGG16LSTM, self).__init__()
        # Chargement des caractéristiques du VGG16 pré-entraîné
        self.feature_extractor: nn.Module = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((7, 7))  # VGG16 utilise par défaut un pool (7x7)
        self.fc_features: nn.Linear = nn.Linear(512 * 7 * 7, 1024)
        # Optionnel : geler l'extracteur de caractéristiques
        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        # LSTM pour la modélisation temporelle
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=1024,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        # Couche entièrement connectée pour la classification
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant du modèle

        Paramètre(s)
        ----------
        x : torch.Tensor
            Tenseur de forme (batch_size, seq_len, c, h, w), où :
            - batch_size : Taille du lot
            - seq_len : Longueur de la séquence temporelle
            - c : Nombre de canaux (typiquement 3 pour RGB)
            - h : Hauteur de l'image
            - w : Largeur de l'image

        Retourne
        -------
        torch.Tensor
            Prédictions de forme (batch_size, num_classes).
        """
        batch_size, seq_len, c, h, w = x.size()
        # Redimensionnement de l'entrée pour l'extracteur de caractéristiques
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
        final_output: torch.Tensor = lstm_out[:, -1, :]
        # Passage avant à travers le classifieur
        output: torch.Tensor = self.fc(final_output)
        return output


def start_tensorboard(logdir: str) -> None:
    """Démarrage de TensorBoard

    Configure et lance une instance de TensorBoard pour visualiser les logs.

    Paramètre(s)
    ----------
    logdir : str
        Chemin du répertoire contenant les journaux pour TensorBoard.

    Retourne
    -------
    None
    """
    tb: TensorBoard = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir])
    url: str = tb.launch()
    print(f"TensorBoard est en cours d'exécution à l'adresse {url}")


def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
    """Obtention d'un élément du jeu de données

    Récupère une image et son étiquette associée à l'indice spécifié.
    Les images sont simulées comme des tableaux NumPy. Une transformation peut être appliquée si définie.

    Paramètre(s)
    ----------
    idx : int
        Indice de l'élément à récupérer.

    Retourne
    -------
    Tuple[np.ndarray, int]
        Un tuple contenant l'image (après transformation si applicable) et l'étiquette associée.
    """
    label: int = self.labels[idx]
    # Données d'image fictives sous forme de tableau NumPy
    image: np.ndarray = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)  # Simulation d'une image
    if self.transform:
        image = self.transform(Image.fromarray(image))  # Conversion en image PIL
    return image, label


def balance_dataset(original_videos_dir: Path, manipulated_videos_dir: Path, output_dir: Path, target_count: int = 75) -> Tuple[Tuple[List[Path], List[int]], Tuple[List[Path], List[int]]]:
    """Équilibrage du jeu de données

    Équilibre un jeu de données en sélectionnant un nombre égal de vidéos originales
    et DeepFake, puis les copie dans un répertoire de sortie.

    Paramètre(s)
    ----------
    original_videos_dir : Path
        Répertoire contenant les vidéos originales.
    manipulated_videos_dir : Path
        Répertoire contenant les vidéos DeepFake.
    output_dir : Path
        Répertoire où les vidéos équilibrées seront sauvegardées.
    target_count : int, optional
        Nombre de vidéos à sélectionner pour chaque classe, par défaut 75.

    Retourne
    -------
    Tuple[Tuple[List[Path], List[int]], Tuple[List[Path], List[int]]]
        Une paire de tuples contenant les listes de chemins des vidéos sélectionnées
        et leurs étiquettes correspondantes (0 pour originales, 1 pour DeepFake).
    """
    # Création des répertoires de sortie
    balanced_original_dir: Path = output_dir / "original"
    balanced_manipulated_dir: Path = output_dir / "manipulated"
    balanced_original_dir.mkdir(parents=True, exist_ok=True)
    balanced_manipulated_dir.mkdir(parents=True, exist_ok=True)
    # Échantillonnage des vidéos
    original_videos: List[Path] = list(original_videos_dir.glob("*.mp4"))
    manipulated_videos: List[Path] = list(manipulated_videos_dir.glob("*.mp4"))
    sampled_original: List[Path] = random.sample(original_videos, target_count)
    sampled_manipulated: List[Path] = random.sample(manipulated_videos, target_count)
    # Copie des vidéos échantillonnées dans les répertoires de sortie
    for file in sampled_original:
        shutil.copy(file, balanced_original_dir / file.name)
    for file in sampled_manipulated:
        shutil.copy(file, balanced_manipulated_dir / file.name)
    print(f"Jeu de données équilibré créé avec {target_count} vidéos dans chaque classe.")
    return (sampled_original, [0] * len(sampled_original)), (sampled_manipulated, [1] * len(sampled_manipulated))


def train_and_validate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
    """Entraînement et validation d'un modèle PyTorch sur plusieurs époques

    Cette fonction alterne entre les étapes d'entraînement et de validation à chaque époque,
    en affichant les pertes et la précision à la fin de chaque phase.

    Paramètre(s)
    ----------
    model : nn.Module
        Le modèle PyTorch à entraîner.
    train_loader : DataLoader
        DataLoader pour l'ensemble d'entraînement.
    val_loader : DataLoader
        DataLoader pour l'ensemble de validation.
    epochs : int
        Nombre d'époques pour l'entraînement.
    """

    for epoch in range(epochs):
        model.train()  # Passe le modèle en mode entraînement
        running_loss: float = 0.0
        # Boucle d'entraînement avec barre de progression
        print(f"Époque {epoch + 1}/{epochs}")
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc="Entraînement")
        for batch_idx, (videos, labels) in train_progress:
            videos, labels = videos.to(device), labels.to(device)
            # Passage avant
            optimizer.zero_grad()
            outputs: torch.Tensor = model(videos)
            loss: torch.Tensor = criterion(outputs, labels)
            # Rétropropagation et mise à jour de l'optimiseur
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
        avg_train_loss: float = running_loss / len(train_loader)
        print(f"Époque {epoch + 1} Perte d'entraînement : {avg_train_loss:.4f}")
        # Boucle de validation
        model.eval()  # Passe le modèle en mode évaluation
        val_loss: float = 0.0
        correct: int = 0
        total: int = 0
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
        avg_val_loss: float = val_loss / len(val_loader)
        val_accuracy: float = 100 * correct / total
        print(f"Époque {epoch + 1} Perte de validation : {avg_val_loss:.4f}")
        print(f"Époque {epoch + 1} Précision de validation : {val_accuracy:.2f}%\n")


def evaluate_model(model: nn.Module, val_loader: DataLoader) -> Tuple[np.ndarray, float, float, float, float]:
    """Évaluation d'un modèle PyTorch sur un ensemble de validation

    Cette fonction calcule les métriques de classification (matrice de confusion, précision,
    rappel, score F1) en collectant les prédictions et les étiquettes réelles.

    Paramètre(s)
    ----------
    model : nn.Module
        Le modèle PyTorch à évaluer.
    val_loader : DataLoader
        DataLoader pour l'ensemble de validation.

    Retourne
    -------
    Tuple[np.ndarray, float, float, float, float]
        - Matrice de confusion (2x2) sous forme de tableau NumPy.
        - Précision globale (accuracy).
        - Précision pondérée (precision).
        - Rappel pondéré (recall).
        - Score F1 pondéré (f1_score).
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    tp: int = 0  # Vrais positifs
    tn: int = 0  # Vrais négatifs
    fp: int = 0  # Faux positifs
    fn: int = 0  # Faux négatifs
    # Collecte des prédictions et des vraies étiquettes
    model.eval()
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs: torch.Tensor = model(videos)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    # Calcul des métriques
    acc: float = accuracy_score(y_true, y_pred)
    precision: float = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall: float = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1: float = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    cm: np.ndarray = np.array([[tp, fn], [fp, tn]])
    return cm, acc, precision, recall, f1


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = ["Positif", "Négatif"]) -> None:
    """Affichage d'une matrice de confusion sous forme de carte thermique (heatmap)

    Cette fonction prend une matrice de confusion sous forme de tableau NumPy
    et la visualise en utilisant `seaborn` et `matplotlib`.

    Paramètre(s)
    ----------
    cm : np.ndarray
        Matrice de confusion (2x2) sous forme de tableau NumPy.
    labels : List[str], optional
        Liste des étiquettes à afficher pour les axes x et y. Par défaut, ["Positif", "Négatif"].
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.ylabel("Valeurs prédites")
    plt.xlabel("Valeurs réelles")
    plt.title("Matrice de confusion")
    plt.show()


if __name__ == "__main__":
    # Initialisation des générateurs de nombres aléatoires pour garantir la reproductibilité
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    # Répertoire des vidéos originales et manipulées
    dataset_dir: Path = pathlib.Path('/kaggle/input/deep-fake-detection-dfd-entire-original-dataset')
    original_videos: Path = dataset_dir / "DFD_original_sequences"
    manipulated_videos: Path = dataset_dir / "DFD_manipulated_sequences"
    # Comptage des vidéos dans chaque répertoire
    num_original_videos: int = len(list(original_videos.glob("*.mp4")))
    num_manipulated_videos: int = len(list(manipulated_videos.glob("*.mp4")))
    print(f"Vidéos originelles : {num_original_videos}")
    print(f"Vidéos DeedFake : {num_manipulated_videos}")
    # Initialisation du SummaryWriter pour TensorBoard
    writer: SummaryWriter = SummaryWriter(log_dir="runs/cross_validation")
    # Transformations pour l'image (redimensionnement, normalisation, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Nouvelle vérification du nombre de vidéos dans les répertoires
    num_original_videos = len(list(original_videos.glob("*.mp4")))  # Modification de l'extension du fichier si nécessaire
    num_manipulated_videos = len(list(manipulated_videos.glob("*.mp4")))
    # Affichage du nombre de vidéos par catégorie
    print(f"Vidéos originelles : {num_original_videos}")
    print(f"Vidéos DeedFake : {num_manipulated_videos}")
    # Création des répertoires pour les vidéos équilibrées
    balanced_dir: Path = pathlib.Path('./balanced_dataset')
    balanced_samples: Tuple[Tuple[List[Path], List[int]], Tuple[List[Path], List[int]]] = balance_dataset(original_videos, manipulated_videos, balanced_dir)
    # Redéfinition des chemins pour les vidéos équilibrées
    balanced_dir = pathlib.Path('./balanced_dataset')
    balanced_original: Path = balanced_dir / "original"
    balanced_manipulated: Path = balanced_dir / "manipulated"
    # S'assure que les répertoires de sortie existent
    balanced_original.mkdir(parents=True, exist_ok=True)
    balanced_manipulated.mkdir(parents=True, exist_ok=True)
    # Mise à jour des chemins du jeu de données pour pointer vers le nouveau sous-ensemble
    balanced_video_paths: List[Path] = []
    balanced_labels: List[int] = []
    for video_path in balanced_original.glob("*.mp4"):
        balanced_video_paths.append(video_path)
        balanced_labels.append(0)  # Étiquette 0 pour les vidéos originelles
    for video_path in balanced_manipulated.glob("*.mp4"):
        balanced_video_paths.append(video_path)
        balanced_labels.append(1)  # Étiquette 1 pour les vidéos DeepFake
    # Affichage de l'état du jeu de données équilibré
    print(f"Dataset équilibré créé avec 350 vidéos dans chaque classe.")
    print(f"Vidéos originelles enregistrées dans : {balanced_original}")
    print(f"Vidéos DeepFake enregistrées dans : {balanced_manipulated}")
    # Vérification des vidéos après équilibrage
    balanced_original_dir: Path = pathlib.Path('./balanced_dataset/original')
    balanced_manipulated_dir: Path = pathlib.Path('./balanced_dataset/manipulated')
    original_count = len(list(balanced_original_dir.glob("*.mp4")))
    manipulated_count = len(list(balanced_manipulated_dir.glob("*.mp4")))
    print(f"Nombre de vidéos originelles : {original_count}")
    print(f"Nombre de vidéos DeepFake : {manipulated_count}")
    # Initialisation du jeu de données avec les vidéos et leurs étiquettes
    balanced_dataset: VideoDataset = VideoDataset(balanced_video_paths, balanced_labels, transform=transform)
    # Validation croisée en 3 plis
    kf: KFold = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_splits: List[Tuple[np.ndarray, np.ndarray]] = [(train_idx, val_idx) for train_idx, val_idx in kf.split(balanced_video_paths)]
    # Détection du périphérique pour l'entraînement (GPU si disponible)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialisation du modèle, de la fonction de perte et de l'optimiseur
    model: VGG16LSTM = VGG16LSTM(num_classes=2).to(device)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: CustomAdamOptimizer = CustomAdamOptimizer(model.parameters(), lr=0.0001)
    # Dictionnaire pour stocker les métriques des différents plis
    metrics: dict[str, List[float]] = {
        "exactitude": [],
        "précision": [],
        "rappel": [],
        "score_f1": []
    }
    # Sélection des indices pour l'entraînement et la validation
    train_idx: List[int] = list(range(0, 120))  # Ajustement selon le jeu de données
    val_idx: List[int] = list(range(120, 150))
    train_dataset: Subset = Subset(balanced_dataset, train_idx)
    val_dataset: Subset = Subset(balanced_dataset, val_idx)
    # Initialisation des DataLoaders
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
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
        # Affichage des métriques du pli
        print(f"Métriques finales pour le pli {fold_idx + 1} - Exactitude : {acc:.4f}, Précision: {precision:.4f}, Rappel : {recall:.4f}, Score F1: {f1:.4f}\n")
        # Enregistrement des métriques dans TensorBoard
        writer.add_scalar(f"Pli_{fold_idx+1}/Exactitude", acc, fold_idx + 1)
        writer.add_scalar(f"Pli_{fold_idx+1}/Précision", precision, fold_idx + 1)
        writer.add_scalar(f"Pli_{fold_idx+1}/Rappel", recall, fold_idx + 1)
        writer.add_scalar(f"Pli_{fold_idx+1}/Score_F1", f1, fold_idx + 1)
        writer.add_figure(f"Pli_{fold_idx+1}/Matrice_Confusion", plt.gcf(), fold_idx + 1)
    # Sauvegarde du meilleur modèle
    torch.save(model.state_dict(), '/kaggle/working/vgg16_best_model.pth')
    # Fermeture de TensorBoard
    writer.close()
    # Lancement de TensorBoard
    start_tensorboard("runs/cross_validation")
