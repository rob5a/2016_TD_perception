# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:26:49 2015

@author: Aurélien Plyer
"""

#
# Partie implementation :
# - carte 2D d'élèvation
# - carte 3D d'occupation (10x 2D)
# - lancé de rayon 2D
#


print 'importation des modules'
import cv2                # OpenCV
import numpy as np        # numpy pour les calculs sur matrices
from pylab import figure, plot
import glob
import time
import pcl



print 'Chargement des données '

#%%

half_res = True

sequence = ''
data_loc = '/home/viki/data_td/perception/TD3/mapping/clairiere/'
dataset = data_loc+sequence+'/%05d_%02d.png'
geometry = np.load(data_loc+sequence+'/geometry.npz')
trajectorie = np.load(data_loc+sequence+'/trajectoire.npz')
T01 = geometry['T01']
K0 = geometry['K0']
K1 = geometry['K1']
baseline = -T01[0]

if half_res:
    K0  = 0.5 * K0
    K1  = 0.5 * K1

#%%
#
#  Etant donné la taille des sequence on les liera au fure et a mesure
#  Pour ce faire on utilise un "Générateur", c'est a dire une fonction
#  possèdant un point de sortie dans une boucle (for ici) et utilisant
#  l'instruction yield pour renvoyer des données à l'appelant
#
def getImages():
    cv2.namedWindow('video',cv2.WINDOW_NORMAL)
    for idx in range(len(glob.glob(data_loc+sequence+'/*_00.png'))):
        I0 = cv2.imread(dataset%(idx+1,0),cv2.IMREAD_GRAYSCALE)
        I1 = cv2.imread(dataset%(idx+1,1),cv2.IMREAD_GRAYSCALE)
        T = trajectorie['T'][idx]
        R = trajectorie['R'][idx]
        if half_res:
            I1 = cv2.pyrDown(I1)
            I0 = cv2.pyrDown(I0)
        cv2.imshow('video', np.concatenate([I0,I1],axis = 1))
        if cv2.waitKey(50) == 27:
            cv2.destroyWindow('video')
            cv2.waitKey(10)
            break
        yield I0, I1, (T,R)


#
# Par exemple une fonction play est alors facille a implémenté :
#
def play():
    '''
       juste pour visualiser la séquence
    '''
    cv2.namedWindow('video',cv2.WINDOW_NORMAL)
    for I0, I1, _ in getImages():
        pass
#%%
play()


############################################################
# Question : La trajectoire vous parait-elle satisfaisante ?
############################################################
# Reponse :
############################################################

#%%
#
# La problématique que l'on se pose maintenant est de combiner les fonctionnalités
# que nous avons vue jusqu'a présent :
#   - SLAM
#   - carte dense de profondeur
# afin de construire une carte d'occupation pour permettre au robot de naviguer
# dans son environnement en évitant la colision avec la 3D de celui-ci.
#
# stoquer les nuages de points 3D brut est pas une solution efficace à manipuler
# on va donc chercher à construire une carte simple en agrégeant ces nuages de
# points 3D.
#
# Pour commencer on va construire une nouvelle classe CartoElevation2D stoquant
# les information de hauteur dans une carte 2D.
#
# La première difficulter valide aussi dans le cas volumique est de choisir une
# géométrie de travail, c'est a dire un plan de référence pour la mesure de
# distance permettant ainsi de passer les nuages de points 3D exprimés dans le
# repère de la caméra à une information plus global
#
############################################################
# Question : quel plan de référence peut-on choisir ?
############################################################
# Réponse :
#
############################################################

class CartoElevation2D:
    def __init__(self, trajectoire,
                 depth_max = 10., height_max = 2.,
                 resolution = (1000,1000), cell_size = 0.1, origin = (512,512),
                 downsample = True, dense_decim = 4 , visu_scale = 5):
        '''
            Constructeur de la carte d'élévation :
            depth_max  : profondeur maximum utilisé dans la carte
            height_max : hauteur maximum des imformation ajouté à la carte
            resolution : taille (x,y) en pixels de la carte
            cell_size  : taille en mètres de la cellule (pixel), la taille réele
                         cartographié est alors cell_size*resolution
            origin     : position de l'origine dans la carte (position en pixel)
            downsample : défini si la disparite est calculé sur une image a plus
                         basse résolution
            dense_decim : decimation temporel des carte stereo inséré
        '''
        self.height_max = height_max
        self.height_min = 0.2
        self.traj = trajectoire
        self.depth_max = depth_max
        self.origin = origin
        self.carte = np.zeros(resolution)
        self.votes = np.zeros(resolution)
        self.cell_size = cell_size
        self.current_time = -1
        self.ref_plane = None
        self.downsample = downsample
        self.decimation_dense = dense_decim
        if self.downsample:
            disp_lvl = 32
        else:
            disp_lvl = 64
        self.stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,disp_lvl, 9)
        #self.stereo = cv2.StereoSGBM(0,disp_lvl, 7, P1 = 10, P2 = 500, speckleWindowSize = 51, speckleRange= 1 )
        self.visu_scale = visu_scale
    def setCalibration(self, baseline, K):
        self.K = K
        self.baseline = baseline
    def run(self,I0,I1):
        if self.current_time == -1:
            #
            # si c'est la première paire d'image traité on cherche
            # a identifier le plan du sols
            #
            self.ref_plane, mask = self._planeExtraction(I0,I1)
            self.ref_image = I0 * mask
            ref_transform = self._computeWorldTransform(self.ref_plane)
            self.initRef = ref_transform
        self.current_time += 1
        #
        # Calcul d'odométrie de la nouvelle donnée :
        #
        self._markPosition()
        if np.mod(self.current_time, self.decimation_dense) == 0:
            self._stereoIntegration(I0,I1)
        self._makeVisu()

    def _planeExtraction(self,I0,I1):
        disp = self._computeStereo(I0,I1)
        pts3D, valid = self._disparityTo3D(disp)

        pc3d = pcl.PointCloud(pts3D.astype(np.float32))
        seg = pc3d.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_distance_threshold(0.1)
        seg.set_method_type(pcl.SAC_PROSAC)
        indices, model = seg.segment()
        #
        # vérifions que c'est bien le plan du sol!
        # pour y arriver il faut remonter la trace des index des points
        # 3D utilisé par pcl
        #
        idx_valid = np.where(valid)
        mask = np.zeros(valid.shape)
        for i in indices:
            mask[idx_valid[0][i],idx_valid[1][i]] = 1
        return np.array(model), mask

    def _computeWorldTransform(self, plane):
        '''
            Calcul la transformation de la première caméras vers le repère
            monde porté par le plan

        '''
        #
        # le vecteur plane = [a, b, c, d] correspond aux équations du plan
        # c'est a dire que tout point X =[x, y, z] appartient au plan ssi
        #    a*x + b*y + c*z + d = 0
        #
        # la normale du plan est obtennue par normalisation des parametres
        # du plan :
        plane_n = plane / np.sqrt(np.sum(plane[:3]**2))
        normal = plane_n[:3]
        d = plane_n[-1]
        #
        # La distance du point X au plan est définit par
        # |np.dot(normal, X) + d|
        # vue que l'origine correspond au point X = [0,0,0], on a alors facilement
        # que d correspond à la distance de notre caméra au plan!
        #
        # Si on garde une convention de repère en vision par ordinateur, la
        # translation du repère caméras vers notre repère monde est alors :
        # T_cam_to_world = [0, d, 0]
        # (qui transforme bien notre centre de projection en un point [0, d,0])
        #
        # On cherche la matrice de rotation envoyant le vecteur u [0, -1, 0]
        # sur la normal du plan. Cela s'obtient facillement par la solution
        # suivante qu'on peut obtennir à partir de la formule de Rodrigues :
        #
        # c = | u . n |
        # v = u x n
        # s = || v ||
        # R = I + [v] + [v]^2 * (1-c)/s^2
        #
        #
        ###############################################
        # Question : implémenté le calcul de R
        ###############################################
        # Réponse :
        ###############################################


        ###############################################
        #
        # On peut vérifier que les vecteurs ox et oz sont bien allingé
        # dans le plan en vérifiant les valeurs de :
        #
        #   np.dot(np.dot(R,np.array([[1.],[0],[0]])).T,normal)
        #   np.dot(np.dot(R,np.array([[0.],[0],[1]])).T,normal)
        #
        # Qui doivent etre trés petites.
        #
        T = np.array([[0],[d], [0]])
        #
        # La transformation des point du repere camera 0 vers
        # le repère monde est alors simplement l'inverse
        # de la transformation qu'on a estimé
        #
        return {'R':cv2.Rodrigues(R.T)[0], 'T':-np.dot(R.T,T)}


    def _disparityTo3D(self, disp):
        # récuperation des info de calibration
        baseline = self.baseline
        K = self.K

        nrows, ncols = disp.shape
        U, V = np.meshgrid(range(ncols), range(nrows))
        Z = baseline * K[0,0] / disp
        X = (U - K[0,2]) * Z / K[0,0]
        Y = (V - K[1,2]) * Z / K[1,1]
        valid = np.ones(Z.shape)
        valid[disp == 0] = 0
        valid[Z < 1.] = 0
        valid[Z > self.depth_max] = 0
        Z = Z[valid == 1]
        X = X[valid == 1]
        Y = Y[valid == 1]
        N = Z.shape[0]
        X = X.reshape([N,1])
        Y = Y.reshape([N,1])
        Z = Z.reshape([N,1])
        pts3D = np.concatenate([X,Y,Z], axis = 1)
        return pts3D, valid

    def _computeStereo(self, I0, I1):
        ######################################################
        # Question : le calcul de disparité est une opération
        #            couteuse  en temps de calcul, on peut se contenté d'un
        #            calcul a plus basse résolution, modifier cette methode
        #            afin d'effectuer le calcul de disparite sur une resolution
        #            plus faible.
        #######################################################
        # Réponse :


        #########################################################
        return disp
    def _stereoIntegration(self,I0, I1):
        disp = carto._computeStereo(I0,I1)
        pts3d, mask = carto._disparityTo3D(disp)
        T = self.traj['T'][self.current_time]
        R = self.traj['R'][self.current_time]
        # passage du nuage de point dans le repere monde
        N = pts3d.shape[0]
        pts3d_w = (np.dot(R,pts3d.T) + np.dot(T,np.ones([1,N]))).T
        self._heightAcumulation(pts3d_w, T)
    def _markPosition(self):
        orig = self.traj['T'][self.current_time]
        max_v, max_u = self.carte.shape
        ox = int(0.5 + orig[0] / self.cell_size + self.origin[0])
        oy = max_v-int(0.5 + orig[2] / self.cell_size + self.origin[1])
        self.carte[oy,ox] = 10000

    def _heightAcumulation(self, pts, orig):
        max_v, max_u = self.carte.shape
        # passage dans un repere robotique
        x = pts[:,0]
        y = pts[:,2]
        z = -pts[:,1]
        # conversion vers les coordonnees
        u = (0.5 + x / self.cell_size + self.origin[0]).astype(np.int)
        v = max_v - (0.5 + y / self.cell_size + self.origin[1]).astype(np.int) -1
        valid = np.ones(u.shape)
        valid[u  <  0   ] = 0
        valid[u >= max_u] = 0
        valid[v  <  0   ] = 0
        valid[v >= max_v] = 0

        t1 = time.time()
        carte, vote = self.__voteCarte(self.carte, self.votes, u, v, z, valid)
        t2 = time.time()
        self.voteTime = t2-t1
        print 'insertion de nuage de points en %f secondes'%self.voteTime
    def _makeVisu(self):
        visu = self.carte.copy()
        visu[self.votes > 0] = visu[self.votes > 0]/self.votes[self.votes > 0]
        visu = (visu/self.visu_scale)*255
        visu[visu>255] = 255
        visu[visu < 0] = 0
        visu = cv2.applyColorMap(visu.astype(np.uint8),cv2.COLORMAP_JET)
        self.visu = visu
        cv2.imshow('carte',visu)
    @staticmethod
    def __voteCarte(carte, votes, u, v, z, valid):
        carte[v[valid==1],u[valid==1]] += z[valid==1]
        votes[v[valid==1],u[valid==1]] += 1
        return carte, votes



#%%
#%%
# Jeux de parametres suivant la sequence traité
# salon :
param = dict(depth_max = 12, resolution = (600,600), origin = (400,300), visu_scale = 3)

#%%
# clairiere 
param = dict(depth_max = 12, resolution = (400,400), origin = (200,100), visu_scale = 3)
#%%

# Création de la classe de cartographie 

carto = CartoElevation2D(trajectorie, **param)
carto.setCalibration(baseline,K0)

#%%

cv2.namedWindow('carte',cv2.WINDOW_NORMAL)
acc = 0
start_image = 0
fourcc = cv2.cv.FOURCC(*'XVID')

vid = None
for I0, I1, _ in getImages():
    acc+=1
    if acc > start_image:
        carto.run(I0,I1)
        I0c = cv2.cvtColor(I0, cv2.COLOR_GRAY2BGR)
        visu = np.zeros(I0c.shape)
        for i in range(3):
            visu[:,:,i] = cv2.resize(carto.visu[:,:,i],(I0.shape[1], I0.shape[0]))
        visu = np.concatenate([I0c,visu], axis = 1).astype(np.uint8)
        if vid is None:
            row, col = visu.shape[:2]
            vid = cv2.VideoWriter('/home/viki/data_td/perception//TP3_'+sequence+'.avi',fourcc, 10.0, (col, row), True)
            vid.write(visu)
        else:
            vid.write(visu)
vid.release()
cv2.destroyWindow('carte')


#%%
###############################################################################
# Question :
#       Faire une classe CartoOccupation3D de carte d'occupation libre/occupe
#       intégrant l'information d'occupation le long des rayons d'observation
#       vous repartierez de la classe CartoElevation2D.
#
#       La carte CartoOccupation3D fonctionnera non plus en 2D mais en 3D
#       L'intégration dans CartoOccupation3D se fera par lancé de rayon en
#       utilisant l'algorithme de Breseham (Cf. Wikipedia).
#
###############################################################################


#%%
# Optimisation : utilisation de cython
#
# à copier directement dans le shell :
#
#%load_ext cythonmagic
#%%cython
#def voteCython(carte, votes, dx, dy, z, valid):
#    for i, v in enumerate(valid):
#        if v == 1:
#            carte[dy[i],dx[i]] += z[i]
#            votes[dy[i],dx[i]] += 1
#
#
# Relancer le code en utilisant la fonction cython pour le vote :
#
# carto.__voteCarte = voteCython

###############################################################################
# Ouverture : profiler le code de mapping et le SLAM et trouvez quels sont
#             les fonctions les plus propices pour optimiser le temps de
#             calcul
###############################################################################



