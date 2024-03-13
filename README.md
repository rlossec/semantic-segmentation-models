# Projet de Segmentation d'Images de DashCam

## Aperçu

Ce dépôt contient le code source et les ressources nécessaires pour entraîner un modèle de deep learning capable de segmenter des images provenant de DashCams. 

Pour expérimenter quelques modèles un autre dépot github est disponible avec une application est fournie pour utiliser le modèle entraîné sur des images de votre choix.


## Prérequis

Python 3.10.

Dépendances nécessaires :

## Jeu de données

On s’appuie sur le jeu de données « Cityscapes Dataset » disponible en ligne .
Le jeu de données Cityscape contient un total de 5000 images de villes européennes, divisées en 2975 images pour l'entraînement, 500 images pour la validation et 1525 images pour les tests. Les images ont une résolution de 2048x1024 pixels, en couleur (RGB) et sont fournies au format png.
Les annotations de segmentation sémantique pour chaque image sont fournies sous forme de cartes de couleur, où chaque couleur correspond à une catégorie d'objet, telle que la route, les piétons, les bâtiments, les panneaux de signalisation, les arbres, etc.

## Préparation des données

### Pré-traitement
Les données en entrée sont les images et leur masques. Une partie des pré-traitements sont communs :
- Réduction de la dimension des images/masques
- Transformation en matrice
Ensuite pour les images les données sont divisées par 255 pour obtenir des nombres décimaux entre 0 et 1
Ensuite pour les masques de CitysScape, ils sont plus détaillés que ce que nous souhaitons avec 32 catégories au lieu des 8 classes que l’on souhaite. Il s’agit donc de convertir dans la segmentation à 8 classes. Ensuite on procède à du One-hot-Encoding.

### Générateur
Les images étant une donnée volumineuse et le nombre d’images que de tel réseau doivent traiter étant important, il convient d’utiliser un générateur d’images qui piochera à la volée un certain nombres d’images. Cela permettant d’économiser des ressources mémoire, la RAM.
Ce générateur construit des lots de 20 images et s’occupe également des tâches de pré-traitement avant que les données ne soient soumises au réseau de neurones.

### Augmentation des données
On parle d’augmentation des données dans le domaine de la Computer vision, lorsqu’on créé de nouvelles images à partir des images existantes en y appliquant des modifications telles que des rotations, différents bruits ou encore des changement de coloration. Elles permettent donc d’obtenir plus d’images mais aussi de limiter le surapprentissage sur nos données.
Dans notre projet, l’augmentation des données est appliquée dans le générateur de données qui applique aléatoirement une modifications de l’image.
Les images que notre modèle devra exploiter a certaines particularités, je n’ai donc pas appliqué un grand nombre de modifications. On peut imaginer que de la neige ou de la pluie peuvent être simulées par un bruit gaussien ou un flou gaussien. On peut imaginer que la dashcam soit légèrement décalé de sa position, et on le simulera par de légers recadrages ou de légères rotations.
Enfin le renversement horizontal permet d’avoir de nouvelles images totalement possibles avec une inversion des lieux que l’on rencontre.
Voici finalement les modifications parmi lesquels le générateur va piocher :
- Rotation aléatoire entre -25 et 25 degrés
- Flou gaussien aléatoire avec un sigma entre 0 et 1
- Bruit gaussien avec une variance aléatoire
- Renversement horizontal avec une probabilité de 50%
- Recadrage aléatoire de 0 à 20% de la taille de l'image


## Entraînement du Modèle

Dans un premier temps, il est important de prendre les métriques et fonction de coût adaptées pour la segmentation sémantique des images.
### Métriques

On prendra en compte trois différentes métriques :
- **Indice de Jaccard ou IoU**
- Le F1-score
- L’exactitude : accuracy

L’exactitude correspond tout simplement au taux de pixels correctement prédits.

L’indice de Jaccard ou IoU pour Intersection over Union correspond pour deux ensembles \(A\) et \(B\) à :

```math
J(A,B) = \frac{\vert A \cap B \vert}{\vert A \cup B \vert}
```

Cet indice est compris entre 0 et 1. Une valeur de 0 indique que les ensembles sont disjoints, tandis qu'une valeur de 1 indique que les ensembles sont identiques.

Voici quelques situations illustrant l'indice de Jaccard :
```math 
J(\{1,2,3\}, \{2,3,4\}) = \frac{2}{4} = 0.5
```

Le F1-score correspond à un équilibre entre précision et recall donné par cette formule :
```math
 F_1 \text{ score} = \frac{2 \times (\text{precision} \times \text{recall})}{\text{precision} + \text{recall}}
```

### Fonctions de coût

Les deux fonctions de coût que l’on testera sont :
- Le Dice loss qui correspond à $`1 - \text{F1-score}`$
- Le Jaccard Loss qui correspond à $`1 - \text{IoU}`$

En termes de formules, on a donc :
```math 
\text{Dice Loss} = 1 - \frac{2 \times (\text{precision} \times \text{recall})}{\text{precision} + \text{recall}}
```
```math 
 \text{Jaccard Loss} = 1 - \text{IoU} 
```


### Optimiseur

Pour l’optimisation on optera pour Adam en affinant le Learning rate au fur et à mesure de l’apprentissage.


## Résultats

In progress