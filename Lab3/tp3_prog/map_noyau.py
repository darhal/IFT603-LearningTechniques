# -*- coding: utf-8 -*-

#####
# Vos Noms (Vos Matricules) .~= À MODIFIER =~.
###

import random
import numpy as np
import matplotlib.pyplot as plt


class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, polynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None

        

    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        #AJOUTER CODE ICI
        rbf = lambda x1, x2: np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * self.sigma_square))
        lin = lambda x1, x2: (x1.T @ x2 + self.c)
        poly = lambda x1, x2: (x1.T @ x2 + self.c) ** self.M
        sig = lambda x1, x2: np.tanh(self.b * x1.T @ x2 + self.d)
        kernels = { "rbf": rbf, "lineaire": lin, "polynomial": poly, "sigmoidal": sig }
        N = x_train.shape[0]
        self.x_train = x_train
        K = np.zeros((N, N))
        for i in range(0, N):
            for j in range(0, N):
                K[i,j] = kernels[self.noyau](x_train[i,:], x_train[j,:])
        self.a = np.linalg.inv(K + self.lamb * np.identity(N)) @ t_train
        
    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        #AJOUTER CODE ICI
        rbf = lambda x1, x2: np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * self.sigma_square))
        lin = lambda x1, x2: (x1.T @ x2 + self.c)
        poly = lambda x1, x2: (x1.T @ x2 + self.c) ** self.M
        sig = lambda x1, x2: np.tanh(self.b * x1.T @ x2 + self.d)
        kernels = { "rbf": rbf, "lineaire": lin, "polynomial": poly, "sigmoidal": sig }
        N = self.x_train.shape[0]
        K = np.zeros(N)
        for i in range(0, N):
            K[i] = kernels[self.noyau](self.x_train[i,:], x)
        y = self.a.T @ K
        return 1 if y > 0.5 else 0

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        # AJOUTER CODE ICI
        return (t - prediction) ** 2

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=10 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        # AJOUTER CODE ICI
        K = 10
        # Shuffling
        zippedXt = list(zip(x_tab, t_tab))
        random.shuffle(zippedXt)
        nX, nT = zip(*zippedXt)
        # Splitting
        Xparts = np.array(np.array_split(nX, K))
        Tparts = np.array(np.array_split(nT, K))
        minErr = np.Inf

        def cross_validation_impl():
            err_list = []
            for f in range(K):
                # Merge X and t folds togther
                Xtrain = np.concatenate(Xparts[np.arange(K)!=f], axis=0)
                Ttrain = np.concatenate(Tparts[np.arange(K)!=f], axis=0)
                self.entrainement(Xtrain, Ttrain)
                # Predict and calculate error on validation fold
                Xvalid = Xparts[f]
                Tvalid = Tparts[f]
                predict = self.prediction(Xvalid)
                err_list.append(self.erreur(Tvalid, predict).mean())
            return np.mean(err_list)

        goodL = self.lamb
        l = 0.000000001
        while (l <= 2):
            self.lamb = l
            if (self.noyau == "rbf"):
                goodSs = self.sigma_square
                ss = 0.000000001
                while (ss <= 2):
                    self.sigma_square = ss
                    err_mean = cross_validation_impl()
                    if err_mean < minErr:
                        minErr = err_mean
                        goodSs = self.sigma_square
                        goodL = self.lamb
                    ss += 0.000000001
                self.sigma_square = goodSs
            elif (self.noyau == "polynomial" or self.noyau == "lineaire"):
                goodM = self.M
                goodC = self.c
                start, end = 2, 7
                if (self.noyau == "lineaire"):
                    start, end = 1, 2
                c = 0
                while (c <= 5):
                    for m in range(start, end, 1):
                        self.c = c
                        self.M = m
                        err_mean = cross_validation_impl()
                        if err_mean < minErr:
                            minErr = err_mean
                            goodM = self.M
                            goodC = self.c
                    c += 0.1
                self.M = goodM
                self.c = goodC
            elif (self.noyau == "sigmoidal"): 
                goodB = self.b
                goodD = self.d
                start, end = 2, 6
                b = 0.00001
                while (b <= 0.01):
                    d = 0.00001
                    while (d <= 0.01):
                        self.b = b
                        self.d = d
                        err_mean = cross_validation_impl()
                        if err_mean < minErr:
                            minErr = err_mean
                            goodB = self.b
                            goodD = self.d
                        d += 0.00001
                    b += 0.00001
                self.b = goodB
                self.d = goodD
            l += 0.000000001
        self.lamb = goodL
        self.entrainement(x_tab, t_tab)

    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()
