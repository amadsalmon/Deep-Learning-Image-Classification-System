# Projet - Système de classification d'images par apprendissage profond

Nous allons utiliser l'outil PyTorch.
Torch est un ensemble de logiciels pour faire du deep learning.
Pytorch est principalement une interface python pour torch.

## Installations (si vous souhaitez le faire sur vos machines, Linux)  

Installation de l'installateur [Miniconda](https://conda.io/miniconda.html). Choisir python 3.7 et 64 bits.

Installation de [PyTorch](http://pytorch.org/) :
```bash
conda install pytorch torchvision -c pytorch
```
  
Version installée sur les machines de l'UFR sur le compte `quenotg`, inclure la ligne suivante dans votre .bashrc (ou 
l'entrer dans votre terminal) :
```bash
export PATH="/home/Public/quenotg/miniconda3/bin:$PATH"
```

Pour un effet immédiat, exécuter la commande :
```
bashsource ~/.bashrc
```


Si nécessaire, installation de [[https://matplotlib.org/|matplotlib]] :
```bash
$ python -mpip install -U pip
$ python -mpip install -U matplotlib
```

## Tutoriels  

[[https://docs.python.org/3.5/tutorial/|Tutoriel Python]] \\
[[http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html|Tutoriel débutants PyTorch]]

Pour ne pas télécharger inutilement les données CIFAR-10 (180 Mo) sur les machines de l'UFR, faire le lien symbolique suivant :
```bash
ln -s /home/Public/quenotg/data .
```

## Pour commencer  

Le TP2 en pytorch :   
En python : [tgabor.py](http://mrim.imag.fr/georges.quenot/tp2/tgabor.py)   
En jupyter notebook : [tgabor.ipynb](http://mrim.imag.fr/georges.quenot/tp2/tgabor.ipynb)  
En jupyter notebook exécuté HTML : [tgabor.html](http://mrim.imag.fr/georges.quenot/tp2/tgabor.html)   
En jupyter notebook exécuté PDF : [tgabor.pdf](http://mrim.imag.fr/georges.quenot/tp2/tgabor.pdf)


## Mise en route  

Faire le tutoriel **manuellement étape par étape** avec le réseau proposé jusqu'à l'entraînement pendant deux itérations et l'évaluation sur les données de test. \\
**Bien comprendre toutes les étapes et le rôle de chaque module ou fonction.** \\
Il ya beaucoup de choses "cachées" dans les différentes fonctions, il faut comprendre tout ce qui se passe.

Lancer :
```bash
ipython --matplotlib
```

Puis :
```bash
%autoindent
```

Copier-coller à partir des fichiers ".py" fournis à la fin de chaque étape.


## Évaluation "en continu" du système  

Dans la partie entraînement du réseau CNN, lister les différentes couches et sous-couches. \\
Donner la taille des différents tenseurs de données Xn et de poids Wn le long du calcul. \\
Modifier le programme pour faire l'évaluation après chaque époque et aussi avant la première (faire une fonction spécialisée). Supprimer les autres affichages intermédiaires. \\
Modifier les fonctions pour calculer à chaque étape le nombre d'opérations flottantes effectuées, séparément pour les additions, les multiplications, les maximums et le total. \\
Afficher en fin de passe, en plus du taux d'erreur global, le nombre d'opérations effectuées, le temps d'exécution, et le nombre d'opérations par seconde.

## Modification du réseau  

Modifier le réseau de manière cohérente : commencer par modifier la taille d'une ou plusieurs couche, plus de cartes (filtres ou plans) dans une couche de convolution ou plus de neurones dans une couche complètement connectée.

Essayer ensuite d'insérer une couche supplémentaire soit de convolution, soit complètement connectée, soit les deux.

Dans tous les cas, rester sur deux itérations seulement pour limiter le temps d'exécution et comparer les performances des différentes variantes. Dans un ou deux cas, laisser tourner l'entraînement jusqu'à ce que la performance jusqu'à ce que la fonction de coût (running loss) ne décroisse plus ou plus significativement. Comparer la performance finale du réseau du tutoriel et d'une de vos variantes.

Essayer ensuite des variantes de la fonction de coût (loss), de l'optimiseur et/ou de la fonction d'activation.


** Prendre systématiquement des notes pour le compte rendu final. **


## Objectif final  

Optimisation d'un réseau pour la tâche de classification CIFAR-10 sous les deux contraintes :
  - Le nombre total de paramètres (c'est le nombre total de paramètres qui définissent le réseau et qui sont appris par la descente de gradient ; c'est la somme des paramètres pour chacune des couches ou sous-couches impliquées (pour celles qui ont des paramètres) ; on ne compte chaque paramètre qu'une seule fois, qu'il soit utilisé plusieurs fois, comme dans les couches de convolution, ou une seule fois, comme dans les couches complètement connectées) ne doit pas dépasser 256K (1 Moctet pour le modèle) ;
  - Le temps d'entraînement sur un GPU de Grid'5000 ne doit pas dépasser une heure (à lancer en dehors des heures de TP).
Vous pouvez partir du réseau défini dans le tutoriel et faire toutes sortes de modification comme : nombre de couches, nombre de filtres dans les couches, essayer des branches parallèles (comme les modules inception), insérer éventuellement un module de "dropout" (https://pytorch.org/docs/stable/nn.html#dropout-layers, pour les couches complètement connectées), organiser une décroissance du taux d'apprentissage), etc. Il faut préparer une manip pendant le cours en mode interactif et la laisser tourner une heure en mode batch plus tard. La note ne portera évidement pas exclusivement sur la performance finale.

<!--un module de "batch normalization" (https://pytorch.org/docs/stable/nn.html#batch-norm), -->
## Suggestions  

Utilisre des tailles de batch plus grandes que 4. L'effet est significatif sur le temps de calcul. Il faut utiliser des sous-multiples de 10000 pour ne pas avoir de problèmes liés à une taille différente pour le dernir batch. 16 est une bonne valeur. **Attention : ** la vitesse de calcul augmente en effet sensiblement avec la taille du batch mais la convergece est en contrepartie plus lente pour le même nombre d'époques en raison de l'affaiblissement de l'effet stochastique dans la descente de gradient. Il faut donc trouver le bon compromis entre le temps gagné sur le traitement d'une époque et le nombre d'époques requis pour atteindre la même performance finale.

Ajuster les tailles des différentes couches pour rester en dessous du budget paramètres fixé (256K).

Augmenter les données. Utiliser (décommenter) les transformations aléatoires additionelles sur les images d'entrée (apprentissage seulement) :
```bash
transform_train = transforms.Compose(
    [# transforms.RandomHorizontalFlip(),
     # transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```
Il n'y a pas toujours un gain et il peut falloir plus d'itérations pour l'avoir.


Mesurer le temps de calcul d'une passe (époque) et estimer le nombre de passes faisables en une heure. Faire une programmation du taux d'apprentissage avec trois palier, respectivement à 0.01, 0.001 et 0.0001 et pour 50%, 25% et 25% du nombre total de passes disponibles.

## Accès aux GPUs sur Grid'5000  

Voir : [[Accès Grid'5000]]

Pour accéder à l'installation via conda (dans le terminal ou dans le .bashrc) :
```bash
export PATH="/srv/storage/irim@storage1.lille.grid5000.fr/anaconda3/bin:$PATH"
```

Pour ne pas recharger inutilement les données images, dans votre répertoire de travail :
```bash
ln -s /srv/storage/irim@storage1.lille.grid5000.fr/ens/data .
``` 

En dehors de ces créneaux, vous pouvez toujours lancer des jobs directement en mode interactif ou en mode batch, voir [[Accès Grid'5000]]. \\
Pas de job de plus d'une heure et pas plus d'un à la fois.


## Evaluation du projet (note de CC)  

Vous ferez un compte-rendu sur le projet uniquement, à envoyer sous forme électronique à Georges.Quenot@imag.fr pour le 24 avril au plus tard. Celui-ci doit comprendre un document expliquant ce que vous avez fait et le code PyTorch de votre meilleur système. Dans le compte-rendu, vous rappellerez succinctement l'objectif du projet et les différentes variantes que vous avez étudiées, en mentionnant leurs performances. Vous pouvez discuter des avantages et inconvénients des différents choix, faire des analyses par catégories pour le meilleur système, et d'une manière générale, faire des commentaires indiquant ce que vous avez compris et retenu en faisant ce projet.

Ce qui est attendu :
  * Fonction spécifique pour le calcul de l'erreur (ou de la précision) globale sur l'ensemble de test
  * Appel de cette fonction avant la première itération (époque) et après chaque itération
  * Affichage de l'évolution de l'erreur (ou de la précision) globale en fonction des itérations
  * Essais de modification de l'architecture du réseau (taille, nombre et/ou types des couches) ou de conditions d'apprentissage (augmentation de données)
  * Description complète de votre réseau final avec : nombre de couches et sous-couches, identification de toutes les variables intermédiaires (les "Xn") de l'entrée à la sortie et en donnant leurs dimensions (tailles de tenseurs) ; identification de toutes les fonctions successives (les "Fn") avec leurs types ; et identification de tous les paramètres de ces fonctions (les "Wn"), en donnant leurs dimensions (tailles de tenseurs) et en précisant éventuellement s'ils sont nuls (s'il n'y a pas de paramètres pour la fonction correspondante, e.g., ReLU).
  * Et, finalement, la meilleure performance obtenue avec les contraintes de nombre de paramètres et de temps d'entraînement.
La performance finale n'est pas un critère essentiel. Vous pouvez rapporter même les essais qui n'ont pas conduit à des améliorations de performance. Il n'est pas indispensable de tout faire.


## Conseils plus détaillés  

Je reproduis ici les informations complémentaires données sur discord et que je donne normalement à l'oral pendant les séances de TP. Pas forcément très structuré.

### Séance 5  

ATTENTION, je rappelle qu'il faut faire deux réservations pour ne pas franchir la frontière nuit / jour de 9h00 : une qui termine à 8h59 et une qui commence à 9h01.
Normalement, vous avez tous dû finir le tutoriel jusqu'à la partie 4 incluse. Ce tutoriel sert de base au projet. On va améliorer et compléter les mesures, analyser les différents éléments, et améliorer la performance globale en modifiant le réseau et/ou les conditions d'entraînement.

Première étape, un peu fastidieuse mais qui aide bien à la compréhension, est de lister les différents éléments avec leurs tailles. On décompose et on numérote au niveau de la sous-couche. Je l'ai déjà fait en cours et revu pendant les TPs sur le tutoriel, c'est à mettre dans le rapport : listes de toutes les Fn avec leure caractéristiques (hyper-paramètres), de tous les Xn (données) avec leurs tailles, et de tous les Wn (paramètres).

Deuxième étape : améliorer les mesures. Ajouter une mesure du temps d'exécution d'une époque (voir les fonctions de récupération du temps dans tgabor.py). Mettre dans une fonction la partie qui fait les statistiques sur la précision, globale et par catégorie. Appeler cette fonction avant la première époque (performance du hasard) et à la fin de chaque époque. Afficher au fur et à mesure la performance globale et seulement à la fin, la performance par catégorie. Vous pouvez mettre en commentaire la partie qui affiche la "loss" au fur et à mesure. Le plus simple est de faire une fonction qui calcule tout à chaque fois mais de n'afficher que ce qu'on veut quand on veut (global ou détaillé, selon). Pousser jusqu'à une dizaine d'itérations pour voir à quel moment ça commence à stagner. Le passage en stagnation est progressif et la stagnation elle-même est un peu instable. Le but est de voir "en gros" à partir de quand ça ne vaut plus la peine de continuer.

### Séance 6  

Une première chose à essayer est de changer la taille du batch. Pour ça il faut le définir en entête et le mettre en paramètre partout. Il est bien de définir une taille différente pour le "train" et le "test". Si le batch est plus grand, le temps d'une époque est plus rapide. Par contre l'apprentissage est plus lent car l'effet stochastique est réduit. Un bon compromis pour la taille de batch est de 20 pour le "train". Pour le test, 400 est une bonne taille. Vous devez constater qu'avec une taille de 20, le temps d'un époque est très réduit et que la précision monte moins vite à chaque époque mais qu'elle monte plus vite pour un temps de calcul donné. Pour le test, il n'y a pas de problème à prendre une taille de batch plus importante et 400 va bien. Au delà, il n'y a plus trop de gain en parallélisme.

Pour la suite, on va modifier le réseau pour améliorer la performance. La modification la plus simple est de changes la taille des couches, c'est à dire le nombre de plans pour les couches de convolution et le nombre d'unités dans les couches complètement connectées. Le seul point délicat est de s'assurer que si on modifie la taille de sortie d'une couche, on modifie de manière cohérente la taille de l'entrée de la couche suivante. Évidement, on ne peut modifier ni la taille de l'entrée de la première couche ni la taille de la sortie de la dernière. On peut typiquement doubler la taille de chaque couche intermédiaire, voire plus pour la première. Une bonne idée est de mettre en paramètres globaux les tailles de ces couches (les 6, 16, 120 et 84). Ce sont des hyper-paramètres en fait. Ils ne sont pas appris et ils définissent la taille (ou la quantité) de ceux qui le sont.

Le deuxième niveau de modification possible consiste à changer le nombre de couches. Le plus simple ici est de remplacer une couche de convolution 5x5 par deux couches 3x3. Ceci a l'avantage de ne pas changer le rognage global. Il faut doubler les sous-couches conv2d et relu mais pas la sous-couche maxpool. De la même façon, il faut assurer la cohérence entre la taille des entrées et celle des sorties.

### Séance 7  

Une dernière chose à essayer est l'augmentation de données comme suggéré ci-dessus. De préférence sur votre réseau final, essayer les quatre combinaisons : avec ou sans symétrie x avec ou sans décalage (extension / rognage) et mesurer les gains associés. Notez que ces transformations se font au niveau des data loader avec un coût négligeable en temps de calcul.

En ce qui concerne la performance finale, jusqu'à un certain point, elle dépend de la taille du réseau (nombre de paramètres) et du nombre d'itérations. Pour que ça reste raisonnable et que les comparaisons entre vous restent valables (même si ce n'est pas le critère principal d'évaluation), j'ai limité la taille à 250000 paramètres (1 Moctet) et le temps d'entraînement à une heure.



