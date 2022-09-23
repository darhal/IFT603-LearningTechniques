# -*- coding: utf-8 -*-

#####
# VosNoms (Matricule) .~= À MODIFIER =~.
###

import numpy as np
import random
from sklearn import linear_model


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        # AJOUTER CODE ICI
        repeat_mat = x if (np.isscalar(x)) else np.reshape(np.repeat(x, self.M+1, axis=0), [len(x), self.M+1])
        phi_x = repeat_mat ** np.arange(self.M+1)
        return phi_x


    def recherche_hyperparametre(self, X, t):
        """
        Trouver la meilleure valeur pour l'hyper-parametre self.M (pour un lambda fixe donné en entrée).

        Option 1
        Validation croisée de type "k-fold" avec k=10. La méthode array_split de numpy peut être utlisée 
        pour diviser les données en "k" parties. Si le nombre de données en entrée N est plus petit que "k", 
        k devient égal à N. Il est important de mélanger les données ("shuffle") avant de les sous-diviser
        en "k" parties.

        Option 2
        Sous-échantillonage aléatoire avec ratio 80:20 pour Dtrain et Dvalid, avec un nombre de répétition k=10.

        Note: 

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        # AJOUTER CODE ICI
        K = len(X) if len(X) < 10 else 10
        Min, Max = 1, 20
        # Shuffling
        zippedXt = list(zip(X, t))
        random.shuffle(zippedXt)
        nX, nT = zip(*zippedXt)
        # Splitting
        Xparts = np.array(np.array_split(nX, K))
        Tparts = np.array(np.array_split(nT, K))
        # Init
        goodM = self.M
        minErr = np.Inf

        for m in range(Min, Max):
            self.M = m
            err_list = []
            
            for f in range(K):
                # Merge X and t folds togther
                Xtrain = np.concatenate(Xparts[np.arange(K)!=f], axis=None)
                Ttrain = np.concatenate(Tparts[np.arange(K)!=f], axis=None)
                self.entrainement(Xtrain, Ttrain, False)
                # Predict and calculate error on validation fold
                Xvalid = Xparts[f]
                Tvalid = Tparts[f]
                predict = self.prediction(Xvalid)
                err_list.append(self.erreur(Tvalid, predict).mean())
            
            # Update the value of M
            err_mean = np.mean(err_list)
            if err_mean < minErr:
                minErr = err_mean
                goodM = self.M
        
        self.M = goodM


    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.
        
        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de 
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)
        
        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        #AJOUTER CODE ICI
        if (using_sklearn):
            reg = linear_model.Ridge(alpha=self.lamb)
            reg.fit(X.reshape(-1, 1), t)
            self.w = reg.coef_
            return

        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)
        phi_x_trans = phi_x.transpose()
        phi_square = phi_x_trans @ phi_x
        I = np.eye(phi_square.shape[0], phi_square.shape[1])
        first_term = np.linalg.solve(self.lamb * I + phi_square, I)
        second_term = phi_x_trans @ t
        self.w = first_term @ second_term


    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        # AJOUTER CODE ICI
        return self.fonction_base_polynomiale(x) @ self.w


    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        # AJOUTER CODE ICI
        return (t - prediction) ** 2
