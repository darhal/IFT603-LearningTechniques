# Notes
## Commandes d'exécution dans un terminal UNIX
Veuillez exécuter le script dans le dossier racine tp1_prog, en utilisant python ou python3 regression.py sk modele_gen nb_train nb_test bruit M lambda
avec les paramètres suivants :
```
sk=0: using_sklearn=False, sk=1: using_sklearn=True

modele_gen=lineaire, sin ou tanh

nb_train: nombre de donnees d'entrainement

nb_test: nombre de donnees de test

bruit: amplitude du bruit appliqué aux données

M: degré du polynome de la fonction de base (recherche d'hyperparametre lorsque M<0)

lambda: lambda utilisé par le modele de Ridge
```

Par exemple: python3 regression.py 1 sin 20 20 0.3 10 0.00

## Requirements
Voir le README.md dans le répertoire parent Lab1 pour avoir les librairies python nécessaires à l'exécution du code.
