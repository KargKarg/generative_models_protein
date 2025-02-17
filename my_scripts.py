import os
from Bio import SeqIO
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


AA_to_indice = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    '-': 20, 'X': 21
}

def generation_dataset(limite: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[int: str]]:
    """
    Génére une matrice dont chaque ligne est une séquence.
    Les colonnes sont encodées par des blocs de taille 22 représentant un AA encodé en one hot.
    """
    np.random.seed(seed)
    random.seed(seed)

    X, Y, labels = [], [], {}

    for classe, fic in enumerate(os.listdir("data")):
        
        ss_domaine = fic.split("_has_")[1][:-4]
        labels[classe] = ss_domaine
        SEQS = list(SeqIO.parse(f"data/{fic}", "fasta"))
        SEQS = random.sample(SEQS, min(limite, len(SEQS)))
        numerique = [list(map(lambda x: AA_to_indice[x], ss.seq)) for ss in SEQS]
        
        for seq in numerique:
            Y.append(classe)
            X.append([1 if i == aa else 0 for aa in seq for i in range(len(AA_to_indice))])

    return np.array(X), np.array(Y), labels


def plot_acp(X_pca: np.ndarray, Y: np.ndarray, fenetre: int, labels: dict) -> None:
    """
    Crée un plot qui affiche par deux les axes de l'ACP de la dimension 1 à fenetre+1 et la sauvegarde.
    """
    palette = sns.color_palette("tab10", len(labels))
    color_dict = {label: palette[i] for i, label in enumerate(labels)}
    colors = [color_dict[int(label)] for label in Y]

    fig, axes = plt.subplots(fenetre, fenetre, figsize=(fenetre*5, fenetre*5))

    for i in range(fenetre):
        for j in range(fenetre):
            if i != j:
                axes[i, j].scatter(X_pca[:, i], X_pca[:, j], c=colors, alpha=0.5, edgecolor="black")
                axes[i, j].set_xlabel(f'PC {i+1}')
                axes[i, j].set_ylabel(f'PC {j+1}')
            else:
                axes[i, j].axis('off')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=color_dict[label]) for label in labels]
    fig.legend(handles, [f"Classe {labels[label]}" for label in labels], title="Sous-domaines", loc="upper right")

    plt.tight_layout()
    plt.savefig(f"visualization/pca_{fenetre}_components.svg")
    plt.close()
    return


def composition_clusters(labels: dict[int: str], affectations: np.ndarray, Y: np.ndarray) -> dict[int: list[int]]:
    """
    Renvoie un dictionnaire affichant les labels par cluster.
    """
    composition = dict()
    for classe in np.unique(affectations):
        composition[classe] = [labels[Y[i]] for i in range(len(affectations)) if affectations[i] == classe]
    return composition


def plot_composition(algo: str, labels: dict[int: str], composition: dict[int: list[int]]) -> None:
    """
    Affiche la composition de chaque cluster.
    """
    _, axes = plt.subplots(1, len(composition), figsize=(len(composition)*10, 5))

    for i, key in enumerate(composition):
        axes[i].hist(composition[key], color="grey", alpha=0.5, edgecolor="black", width=0.5)
        axes[i].set_xlabel('Classe')
        axes[i].set_xticks(list(labels.values()))
        axes[i].set_ylabel('Fréquence')
        axes[i].set_title(f'Cluster({i})')

    plt.tight_layout()
    plt.savefig(f"visualization/{algo}_composition.svg")
    plt.close()
    return