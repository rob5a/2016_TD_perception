# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:37:47 2015

@author: Aurélien Plyer
"""
print 'importation des modules'
import cv2                # OpenCV
import pylab as pl        # pylab pour l'affichage
import numpy as np        # numpy pour les calculs sur matrices
from scipy import ndimage # scipy pour les convolution
from mpl_toolkits.mplot3d import Axes3D
from pylab import plot
import glob
#%%


print 'ouverture de la séquence'

sequence = 'drone_stereo'
data_loc = '/home/viki//data_td/perception/TD2/calcul_pose/'
dataset = data_loc+sequence+'/%05d_%02d.png'
geometry = np.load(data_loc+sequence+'/geometry.npz')
VT = np.load(data_loc+sequence+'/traj.npz')['Pos'].T
VR = np.load(data_loc+sequence+'/traj.npz')['Angle'].T
VR = VR[:,(1,2,0)]
T01 = geometry['T01']
K0 = geometry['K0']
K1 = geometry['K1']
baseline = -T01[0]

frames_number = len(glob.glob(data_loc+sequence+'/*_00.png'))


V0=[cv2.imread(dataset%(i+1,0), cv2.IMREAD_GRAYSCALE) for i in range(frames_number)]
V1=[cv2.imread(dataset%(i+1,1), cv2.IMREAD_GRAYSCALE) for i in range(frames_number)]


#%%
#
#   Objectif :
#      On a maintenant tout les outils pour réaliser une solution de SLAM
#      stéréo, pour ce faire on va reprendre les différents composants
#      vue au cours des TP (matching, calcul de pose) pour les assembler
#      dans une classe de SLAM
#

class SimpleSLAM:
    def __init__(self, ratio_matching = 0.5, ratio_pose = 0.5, detecteur = 'ORB', descripteur = 'ORB'):
        self.ratio_matching = ratio_matching
        self.ratio_pose = ratio_pose
        self.detecteur = detecteur.upper()
        self.descripteur = descripteur.upper()
        self.detecteurs_type = ['FAST', 'STAR', 'SIFT', 'SURF', 'ORB', 'BRISK', 'MSER', 'GFTT',
              'HARRIS', 'Dense', 'SimpleBlob']
        self.descripteurs_type = ['SIFT','SURF', 'BRIEF', 'BRISK','ORB','FREAK']
        self.detecteurs = {}
        self.descripteurs = {}
        for key in self.detecteurs_type:
            detect = cv2.FeatureDetector_create(key)
            self.detecteurs[key.upper()] = detect
        for key in self.descripteurs_type:
            descripteur = cv2.DescriptorExtractor_create(key)
            self.descripteurs[key.upper()] = descripteur
        self.detector =   self.detecteurs[self.detecteur]
        self.descriptor = self.descripteurs[self.descripteur]
        self.matcheur = cv2.BFMatcher(normType = cv2.NORM_L2, crossCheck = True )
        self.KF = []
        self.current_kf = None
        self.traj = []
        self.current_time = 0
        self.prev_pts = None
        print 'construction du SLAM'
    def setCalibration(self, K0, K1, baseline, init_R = np.eye(3), init_T = np.zeros([3,1])):
        self.K0 = K0
        self.K1 = K1
        self.baseline = baseline
        self.traj.append({'R':init_R, 'T':init_T})
        self.distortion = np.zeros([5,1])
        print 'initialisation de la géométrie'

    def run(self, I0, I1):
        '''
            Fonction principale ajoutant au SLAM une nouvelle paire d'image
        '''
        tps = self.current_time
        self.current_time += 1

        if self.current_kf is None:
            # La paire d'image est la première : on génère une premiere KF
            kf = self._computeKeyFrame(tps,I0,I1)
            self._addNewKeyFrame(kf)
        else:
            #
            #  On commence par calculer la pose par rapport à la KF actuelle
            #
            is_track, is_pose, new_pose = self._suivitTemporel(I0)
            # on ajoute la pose à la trajectoire
            self._addPose(new_pose)
            if not is_track:
                # Si jamais le suivit est plus satisfaisant, on ajoute une
                # nouvelle keyframe :
                kf = self._computeKeyFrame(tps,I0,I1)
                self._addNewKeyFrame(kf)

    def _computeKeyFrame(self, tps, I0, I1):
        #======================================================================
        # Question : détection des amers et calcul des descripteurs
        #======================================================================
        # Reponse :


        #======================================================================
        # Question : Calcul de la mise en correspondance entre les descripteurs
        #======================================================================
        # Reponse :

        #======================================================================
        pts0 = np.array([kp0[m.queryIdx].pt for m in match])
        kp0 = [kp0[m.queryIdx] for m in match]
        desc0 = [desc0[m.queryIdx] for m in match]
        pts1 = np.array([kp1[m.trainIdx].pt for m in match])
        #======================================================================
        # Question : Verification hypothèse stereo
        #     error = |pts1-pts2|
        #======================================================================
        # Reponse :

        #======================================================================
        mask = np.ones([pts1.shape[0],1])
        mask[error > 1] = 0
        matching_inlier_ratio = float(np.sum(mask == 1)) / mask.shape[0]
        # Recuperation points valide
        pts0  = pts0[mask.ravel() == 1]
        pts1  = pts1[mask.ravel() == 1]
        kp0   = [k for i, k in enumerate(kp0) if mask[i] == 1]
        desc0 = np.array([d for i, d in enumerate(desc0) if mask[i] == 1])
        # Triangulation
        pts3D = self._triangulateStereo(pts0,pts1)
        # Transformation du nuage de point dans un repère Global
        pts3D = self._applyCameraToWorld(self.traj[tps], pts3D)

        print '===================================='
        print ' calcul d\'une nouvelle image clef'
        print ' nombre de point trouvés : %d '%pts3D.shape[0]
        print ' ratio d\'inlier stéréo : %f'%matching_inlier_ratio
        print '===================================='
        kf = {'desc':desc0,
              'kp':kp0,
              'pts':pts0,
              'pts3D':pts3D,
              'timestamps':tps,
              'inlier_matching':matching_inlier_ratio}
        return kf

    def _applyCameraToWorld(self, transform, pts3D):
        N = pts3D.shape[0]
        R = transform['R']
        T = transform['T']
        #======================================================================
        # Question : Changement de repere :
        #    res = R*pts3D + T
        #======================================================================
        # Réponse :


        #======================================================================
        return res

    def _suivitTemporel(self, I0):
        pts3D = self.current_kf['pts3D']
        desc0 = self.current_kf['desc']
        kp0   = self.current_kf['kp']
        kp1 = self.detector.detect(I0)
        kp1, desc1 = self.descriptor.compute(I0, kp1)
        match = self.matcheur.match(desc0, desc1)
        pts0 = np.array([kp0[m.queryIdx].pt for m in match])
        pts1 = np.array([kp1[m.trainIdx].pt for m in match])
        pts3D = np.array([pts3D[m.queryIdx,:] for m in match])

        F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.RANSAC, 5.0)
        matching_inlier_ratio = float(np.sum(mask == 1)) / mask.shape[0]

        pts0 = pts0[mask.ravel() == 1]
        pts1 = pts1[mask.ravel() == 1]

        pts3D_valid = pts3D[mask.ravel() == 1,:]

        N = pts0.shape[0]
        object_pts = pts3D_valid.reshape([N,1,3]).astype(np.float32)
        image_pts  = pts1.reshape([N,1,2]).astype(np.float32)

        r, t, inliers_idx = cv2.solvePnPRansac(object_pts, image_pts, self.K0, self.distortion)
        pose_inlier_ratio = float(inliers_idx.shape[0])/N
        print '==============================='
        print ' suivit temporel temps : %03d '%self.current_time
        print ' inlier matching ratio  : %f' %  matching_inlier_ratio
        print ' inlier pose ratio      : %f' %  pose_inlier_ratio
        print ' T :  (%02.02f,%02.02f,%02.02f) '%(t[0],t[1],t[2])
        print ' R :  (%02.02f,%02.02f,%02.02f) '%(r[0],r[1],r[2])
        print '==============================='
        if matching_inlier_ratio > self.ratio_matching:
            is_track = True
        else:
            is_track = False
        if pose_inlier_ratio > self.ratio_pose:
            is_pose = True
        else:
            is_pose = False
        R, _ = cv2.Rodrigues(r)
        R = R.T
        t = - np.dot(R, t)
        return is_track, is_pose, {'R':R,'T':t}

    def _triangulateStereo(self, pts0, pts1):
        disparity = np.expand_dims(pts1[:,0] - pts0[:,0],axis = 1)
        u = np.expand_dims(pts0[:,0], axis = 1)
        v = np.expand_dims(pts0[:,1], axis = 1)
        #======================================================================
        # Question : calculez le nuage de points 3D à partir des disparités
        #            rappel : z = baseline * f  / disparity
        #                     x = (u - cx) * Z / f
        #                     y = (v - cy) * Z / f
        #======================================================================
        # Reponse :


        #======================================================================
        pts3D = np.concatenate([X,Y,Z], axis = 1)
        return pts3D
    def _addPose(self, pose):
        self.traj.append(pose)
    def _addNewKeyFrame(self, kf):
        self.KF.append(kf)
        self.current_kf = kf
        print '+++++++++++++++++++++++++++++++++'
        print '================================='
        print ' ajout d\'une nouvelle keyframe'
        print ' nombre de key frames : %d'%len(self.KF)
        print ' nombre de poses      : %d'%len(self.traj)
        print ' ratio de keyframe    : %f'% (float(len(self.KF))/len(self.traj))
        print '================================='

##
# Pour évaluer les résultats
##
def evalResultat(T,R,VT,VR):
    '''
        Fonction de plot de trajectoire pour comparaison
    '''
    figure()
    plot(T[:,0],'r-')
    plot(T[:,1],'g-')
    plot(T[:,2],'b-')
    plot(VT[:,0],'r--')
    plot(VT[:,1], 'g--')
    plot(VT[:,2], 'b--')
    title('Comparaison de la pose')
    figure()
    plot(R[:,0],'r-')
    plot(R[:,1],'g-')
    plot(R[:,2],'b-')
    plot(VR[:,0], 'r--')
    plot(VR[:,1], 'g--')
    plot(VR[:,2], 'b--')
    title('Comparaison des angles')

###############################################################################
#%%
# Testons la classe
#
###############################################################################

slam = SimpleSLAM()
slam.setCalibration(K0,K1,baseline)

#%%
###############################################################################
# Lancons le calcul :
###############################################################################
for i in range(frames_number):
    slam.run(V0[i],V1[i])
T = np.array([traj['T'] for traj in slam.traj])
R = np.array([cv2.Rodrigues(traj['R'])[0] for traj in slam.traj])

#%%
###############################################################################
# evaluation du resultat
###############################################################################

evalResultat(T,R,VT,VR)

###############################################################################
# Question : interprétation des résultats, que remarquez vous sur ces courbes?
#            quand-est-ce qu'on prend le plus de dérive?
###############################################################################
# Réponse :


###############################################################################
#%%
def runAndEval(slam, V0, V1, fname):
    '''
        Fonction permettant de benchmarquer et sauvegarder les résulats
    '''
    for I0, I1 in zip(V0,V1):
        slam.run(I0,I1)
    T = np.array([traj['T'] for traj in slam.traj])
    R = np.array([cv2.Rodrigues(traj['R'])[0] for traj in slam.traj])
    figure()
    plot(T[:,0],'r-')
    plot(T[:,1],'g-')
    plot(T[:,2],'b-')
    plot(VT[:,0],'r--')
    plot(VT[:,1], 'g--')
    plot(VT[:,2], 'b--')
    title('Comparaison de la pose')
    savefig(fname+'_trans.png',bbox_inches='tight')
    figure()
    plot(R[:,0],'r-')
    plot(R[:,1],'g-')
    plot(R[:,2],'b-')
    plot(VR[:,0], 'r--')
    plot(VR[:,1], 'g--')
    plot(VR[:,2], 'b--')
    title('Comparaison des angles')
    savefig(fname+'_rot.png',bbox_inches='tight')


###############################################################################
# Question : Testez différents paramétrage sur le seuil de ratio_matching et
#            testez différents descripteurs
###############################################################################
# Réponse :
#

#%%
