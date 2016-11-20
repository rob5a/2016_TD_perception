# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:27:14 2016

@author: Aurelien Plyer
"""

# importation des modules qu'on va utiliser : 
import cv2                # OpenCV
import pylab as pl        # pylab pour l'affichage
import numpy as np        # numpy pour les calculs sur matrices
from scipy import ndimage # scipy pour les convolution
from pylab import plot


#%% 
#
# Cette partie du TD est essentiellement une application, il est recomander
# de lire les différentes fonctions en jeux dans cet exemple pour faire le
# lien avec les parties précédente du TP
#



#%%
# Chargement des données 

data_dir = '/home/viki/data_td/perception/TD1/panorama/'
images_name = data_dir+'/stitching_%02d.jpg'
images = [cv2.resize(cv2.imread(images_name%i),(640,480)) for i in range(1,10)]

Iref = images[0]


#%%

def homography(image_a, image_b):
    """
        Fonction de calcul de l'homographie allignant image_b sur image_a
    """
    # Détection des SIFT
    sift = cv2.SIFT(edgeThreshold=10, sigma=1.5, contrastThreshold=0.08)
    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)
    
    # association des SIFTS
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, trainDescriptors=des_b, k=2)

    # Lowes Ratio
    good_matches = []
    for m, n in matches:
        if m.distance < .75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches])\
        .reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches])\
        .reshape(-1, 1, 2)

    # Calcul de l'homographie 
    if len(src_pts) > 4:
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
    else:
        M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    return M

#%%


def warp_image(image, homography):
    """Warps 'image' by 'homography'

    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.

    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    h, w, z = image.shape
    # calcul des offset
    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(homography, p)

    yrow = p_prime[1] / p_prime[2]
    xrow = p_prime[0] / p_prime[2]
    ymin , xmin, ymax, xmax = min(yrow), min(xrow),  max(yrow), max(xrow) 

    # prise en compte des offset
    new_mat = np.array([[1, 0, -1 * xmin], [0, 1, -1 * ymin], [0, 0, 1]])
    homography = np.dot(new_mat, homography)

    # height and width of new image frame
    height = int(round(ymax - ymin))
    width = int(round(xmax - xmin))
    size = (width, height)
    # Do the warp
    warped = cv2.warpPerspective(src=image, M=homography, dsize=size)

    return warped, (int(xmin), int(ymin))

#%%

def create_mosaic(images, origins):
    """Combine multiple images into a mosaic.
    Arguments:
    images: a list of 4-channel images to combine in the mosaic.
    origins: a list of the locations upper-left corner of each image in
    a common frame, e.g. the frame of a central image.
    Returns: a new 4-channel mosaic combining all of the input images. pixels
    in the mosaic not covered by any input image should have their
    alpha channel set to zero.
    """
    # find central image
    for i in range(0, len(origins)):
        if origins[i] == (0, 0):
            central_index = i
            break
    central_image = images[central_index]
    central_origin = origins[central_index]
    # zip origins and images together
    zipped = zip(origins, images)

    # sort by distance from origin (highest to lowest)
    func = lambda x: math.sqrt(x[0][0] ** 2 + x[0][1] ** 2)
    dist_sorted = sorted(zipped, key=func, reverse=True)
    # sort by x value
    x_sorted = sorted(zipped, key=lambda x: x[0][0])
    # sort by y value
    y_sorted = sorted(zipped, key=lambda x: x[0][1])

    # determine the coordinates in the new frame of the central image
    if x_sorted[0][0][0] > 0:
        cent_x = 0  # leftmost image is central image
    else:
        cent_x = abs(x_sorted[0][0][0])

    if y_sorted[0][0][1] > 0:
        cent_y = 0  # topmost image is central image
    else:
        cent_y = abs(y_sorted[0][0][1])

    # make a new list of the starting points in new frame of each image
    spots = []
    for origin in origins:
        spots.append((origin[0]+cent_x, origin[1] + cent_y))

    zipped = zip(spots, images)

    # get height and width of new frame
    total_height = 0
    total_width = 0

    for spot, image in zipped:
        total_width = max(total_width, spot[0]+image.shape[1])
        total_height = max(total_height, spot[1]+image.shape[0])

    print( "height %d"%total_height)
    print( "width %d"%total_width)

    # new frame of panorama
    stitch = np.zeros((total_height, total_width, 4), np.uint8)

    # stitch images into frame by order of distance
    for image in dist_sorted:
        offset_y = image[0][1] + cent_y
        offset_x = image[0][0] + cent_x
        for i in range(0, image[1].shape[0]):
            for j in range(0, image[1].shape[1]):
                # print i, j
                if image[1][i][j][3] != 0 :
                    stitch[i+offset_y][j+offset_x][:4] = image[1][i][j]
    return stitch



#%%
ref = images[5]

###############################################################################
# Question : utlisier la fonction homography pour calculer une liste d'homographie
#           recalant les images du vecteurs images sur l'image de reference ref
################################################################################
# Reponse :

H = [homography(ref, im) for im in images]
################################################################################


#%%
###############################################################################
# Question : calculer le vecteur des images warper et des offset avec la fonction
#            warp_image et les homographies précédement calculées :
###############################################################################
# Réponse :

result = [warp_image(im,h) for im,h in zip(images,H) if norm(h) > 0]

###############################################################################

#%%
###############################################################################
# Question : fusioner les différentes images avec la fonction create_mosaic
###############################################################################
# Réponse :

pt = [p[1] for p in result]
im = [p[0] for p in result]

test = create_mosaic(im[::-1],pt[::-1])
#%%
pl.figure()
pl.imshow(test)