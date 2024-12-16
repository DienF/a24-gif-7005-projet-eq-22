# Projet : Détection de deepfake

## Contexte et problématique

Le *deepfake* est "une technique de synthèse multimédia reposant sur l'intelligence artificielle" qui superpose un ou plusieurs fichiers image, vidéo ou audio sur d’autres du même type, pour par exemple remplacer le visage ou la voix d’une personne par une autre. Cette technique a une évolution constante et accélérative aujourd’hui où l’intelligence artificielle générative est en plein essor, et bien qu’elle puisse innocemment paraître comme un outil créatif permettant de ramener à la vie d’anciennes célébrités ou de créer des montages humoristiques, la première pensée que l’on pourrait avoir est la crainte de la création d’infox et de canulars malveillants.
Malheureusement, cette crainte est maintenant réalité : il devient nécessaire de mettre en place des systèmes de reconnaissance de fichiers *deepfake*, afin de distinguer la réalité de la fiction. C’est ce que nous chercherons à faire, plus précisément sur le *deepfake* image et vidéo.

## Méthodes envisagés

Notre objectif va être d’obtenir un modèle de détection de visage "*deepfaké*", c’est-à-dire qu’il va pouvoir nous dire binairement si une vidéo qui lui a été donné présente du *deepfake* ou non. Pour y parvenir, il va falloir entraîner le modèle sur plusieurs vidéos, où il va analyser chaque image de la vidéo et y détecter les traits de visage de la personne. Il saura alors différencier les patterns présents sur un visage réelle et sur un visage *deepfake*.
Le modèle approprié pour l’analyse d’images est celui des réseaux de neurones, où l’on peut utiliser par exemple des techniques comme le réseau à convolution, la composition de filtres ou encore le modèle *Generative Adversarial Networks*, afin de classifier plus facilement les pixels d’une image et détecter les différents patterns respectivement. Nous utiliserons les techniques les plus récentes de la classification de vidéo que l’on trouvera dans des revus scientifique.

## Jeu de données utilisé

Le jeu de données utilisé est disponible sur [Kaggle](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset?select=DFD_manipulated_sequences). Il s’agit de deux dossiers : l’un contenant des vidéos concentrés sur le visage d’une personne sans aucun montage ni trucage, et l’autre contenant plusieurs versions de ces mêmes vidéos avec un visage auquel on a appliqué le *deepfake* d’un autre visage.
Le jeu de données possède au total 3432 fichiers, dont 364 vidéos non-modifiés et 3068 variantes modifiées. Ce sera à nous de juger si le modèle nécessitera et aura des performances satisfaisantes en utilisant toutes les vidéos à notre disposition ou non.

## Approches

- VGG16LSTM
- AlexNetLSTM