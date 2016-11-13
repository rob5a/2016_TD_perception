from __future__ import print_function
import urllib
import bz2
import os
import numpy as np
from scipy.sparse import lil_matrix
import pylab as pl
import time
from scipy.optimize import least_squares

from mpl_toolkits.mplot3d import Axes3D


#%%

dataset = '/home/viki/data_td/perception/TD2/bundle/'

#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
#FILE_NAME = "problem-49-7776-pre.txt.bz2"


#BASE_URL = 'http://grail.cs.washington.edu/projects/bal/data/venice/'
#FILE_NAME = 'problem-52-64053-pre.txt.bz2'

#BASE_URL = 'http://grail.cs.washington.edu/projects/bal/data/trafalgar/'
#FILE_NAME = 'problem-21-11315-pre.txt.bz2'

#BASE_URL = 'http://grail.cs.washington.edu/projects/bal/data/final/'
#FILE_NAME = 'problem-93-61203-pre.txt.bz2'

BASE_URL = 'http://grail.cs.washington.edu/projects/bal/data/dubrovnik/'
FILE_NAME= 'problem-16-22106-pre.txt.bz2'
#FILE_NAME = 'problem-135-90642-pre.txt.bz2'
#FILE_NAME = 'problem-356-226730-pre.txt.bz2'

URL = BASE_URL + FILE_NAME
FILE_NAME = dataset+FILE_NAME

if not os.path.isfile(FILE_NAME):
    urllib.urlretrieve(URL, FILE_NAME)

#%%
#
# Fonction de lecture des donnees
#


def read_bal_data(file_name):
    with bz2.BZ2File(file_name, "r") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())
        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))
        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]
        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))
        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))
    return camera_params, points_3d, camera_indices, point_indices, points_2d



#%%
# Lecture des donnees :

camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

#%%

n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: %d"%(n_cameras))
print("n_points: %d"%(n_points))
print("nombre total des parametres : %d"%(n))
print("nombre total des mesures : %d"%(m))

#%%
###############################################################################
# Question :
#             Quel est la taille du système a résoudre lorsqu'on fait un ajustement
#             de faisceau sur ces données? quel est la quantité de RAM nécéssaire
#             pour stocker l'opérateur dense (en suposant qu'on utilise des double)
#             de 64bit pour stoquer ses éléments?
###############################################################################
# Réponse :


###############################################################################



#%%

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
#%%

def project(points, camera_params):
    '''
        fonction de projection des points 3D en coordonnées normalisés
        les parametres de la cameras i sont
        r = camera_params[i,:3] (notation de Rodrigues)
        t = camera_params[i;3:6]
        f = camera_params[i,6]
        k1 = camera_params[i,7]
        k2 = camera_params[i,8]

        les équation projectant
        X' = R X + T
        x' = -(X'[0]/X'[2], X'[1]/X'[2])
        x = f(1+k1 * ||x'||^2 + k2* ||x'||^4) * x
    '''
###############################################################################
# Question :
#       Ercivez la fonction de projectant les points (points 3D)
#       sur les mesures (points 2D normalisé) points_proj
###############################################################################
# Reponse :



###############################################################################
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
###############################################################################
# Question :
#       Ercivez la fonction de calculant l'erreure de reprojection des points
#       3D et mettez le dans un vecteur de résidus à 1 dimension avec ravel()
###############################################################################
# Reponse :


###############################################################################
    return residus
#%%

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    '''
        Construction de la matrice sparse de visibilité :
        cameras_indices = vecteur de taille nb-mesure reliant les mesures aux
                          cameras
        points_indices = vecteur de la taille du nb de mesure reliant les mesures
                         aux points 3D qu'on cherche a reconstruire

    '''
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)

###############################################################################
# Question :
#   Remplire la matrice A lorsque le coefficient associé est utilisé par
#   l'opérateur de bundle. la matrice A est de taille nb-mesure nb-inconnus
#   ou nb-inconnus = 9*nb-cameras + 3*nb-pts3D.
#   si la ieme mesure est mesuré par la cameras k et correspond au points j
#   on a alors :
#   for tt in range(9):
#       A[2*i,  9*k+tt] = 1
#       A[2*i+1,9*k+tt] = 1
#   et
#   for tt in range(3):
#       A[2*i,    n_cameras + 3* j +tt] = 1
#       A[2*i+1,  n_cameras + 3* j +tt] = 1
###############################################################################
# Reponse :


###############################################################################
    return A

#%%
# Calcul du point initial :
x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

# résidus initial
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

#%%
# Lancement de l'optimisation

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
t1 = time.time()

#%%
#Affichage du résidus

pl.figure(1)
pl.plot(f0)
pl.plot(res['fun'])
pl.legend(['before', 'after'])
pl.title('erreur residuel (reprojection) avant et apres optimisation')

###############################################################################
# Question: que remarquez vous au niveau du residus avant et apres optimisation?
###############################################################################
# Reponse :


###############################################################################

#%%
#
#  Récupération des points 3D apres optimisation
#
cam_size = camera_params.ravel().shape[0]

pts_after = res['x'][cam_size:].reshape([n_points,3])

#%%
# affichage des points avant/apres

fig = pl.figure(2)
ax = Axes3D(fig)
ax.scatter(points_3d[::4,0], points_3d[::4,1], points_3d[::4,2], zdir = 'z', c = [[1.,0,0]])
ax.scatter(pts_after[::4,0], pts_after[::4,1], pts_after[::4,2], zdir = 'z', c = [0,0,1.])
ax.set_xlim3d(-20,20)
ax.set_ylim3d(-20,20)
ax.set_zlim3d(-30,0)

#%%

