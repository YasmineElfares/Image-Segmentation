import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import cv2
import math
from decimal import Decimal
from sklearn.cluster import KMeans

# Read in the image
#image = cv2.imread('/content/black_kitten.jpg')
#image = image[:,:,0]
#plt.imshow(image)
#print(image.shape)
#width = image.shape[0]
#height = image.shape[1]
#features = np.reshape(image, (width, height))
#print(features)


### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    #print(idxs)
    centers = features[idxs]
    #print(centers[0])
    assignments = np.zeros(N, dtype=np.uint32)

      #2. Assign each point to the closest center
      #3. Compute new center of each cluster
      #4. Stop if cluster assignments did not change
      #5. Go to step 2

    old = np.zeros(N, dtype=np.uint32)
    l = len(assignments)
    for element in range(l):
      old[element] = assignments[element]
    for n in range(num_iters):
      #print(len(features))
        for i in range(len(features)):
          f = features[i]
          dist = Decimal(999.9)
          allDistances = []
          #assignments[i] = Decimal(dist)
          #print(f)
          for x in centers:
            #print(f)
            #print(x) 
            currentDist = abs(math.dist(f, x))
            allDistances.append(currentDist)
            #print(allDistances) 
            index = np.argmin(allDistances) 
            #print(index)  
          assignments[i] = index
        #print(assignments)
      ############################################
        for j in range(k):
          cluster = [features[x] for x in range(k) if assignments[x]== j]
          #if len(cluster) != 0:
          sum = 0
          avg = 0
          for s in range(len(cluster)):
            sum += cluster[s]
          if len(cluster) != 0:
            avg = sum/len(cluster)
            centers[j] = avg
      ############################################
        if np.array_equal(old, assignments):
            break
        l = len(assignments)
        for k in range(l):
          #print(k)
          old[k] = assignments[k]
        #old = assignments.copy()

    return assignments

#kmeans(features, 2, num_iters=100)

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)
    #old = np.zeros(N, dtype=np.uint32)
    oldt = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        for i, f in enumerate(features):
            assignments[i] = np.argmin([np.linalg.norm(f-x) for x in centers]) 
        for j in range(k):
            #print(features)
            cluster = [features[x] for x in range(len(features)) if assignments[x]== j]
            if len(cluster) != 0:
                centers[j] = np.mean(cluster, axis=0)
        if np.array_equal(oldt, assignments):
            #print("BREAK Temp!", j)
            break  
        oldt = assignments.copy()
    return assignments




### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
 
    # train   H:  399 (number of rows)   W:  624 (number of cols)    C: 3
    # img[0].size              = 624*3
    # features.size            = 399*624*3 = 746928
    # features[0][1].size      = 1
    # features[0].size         = 3
    # features[0]              = [0,0,0]
    # N, D = features.shape  
    #            N = 399*624   = 248976
    #            D             = 3 

    k = 0
    for i in range(H) :
      for j in range(W):
        color = img[i][j]
        features[k] = color
        k+=1 
 
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    
    #x 64 , 32
    k = 0
    for i in range(H) :
      for j in range(W):
        colorFeature = color[i][j]
        coords = np.array([i,j])
        feature = np.concatenate((colorFeature, coords), axis=None)

        #meanNorm = feature - float(np.mean(feature, axis=0)))
        #unitNorm =  np.std(feature, axis=0))        
        #if( unitNorm != 0):
        #  unitNorm = meanNorm/ unitNorm
 
        #features[k] = unitNorm
        features[k] = feature
        k+=1

    #print(features[0])
    #print(features[92])
    #print("------------------")
    features = features - np.mean(features, axis=0)
    features = features / np.std(features , axis=0)

    #print(features[0])
    #print(features[92])
    
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = 0
    ### YOUR CODE HERE
    TP = 0;
    TN = 0;
    P = 0;
    N = 0;
    H, W = mask.shape
    for i in range(H) :
      for j in range(W):
        if( mask_gt[i][j] == 1 ):
          P +=1
          if( mask[i][j] == 1):
            TP +=1
      else:
        N +=1
        if( mask[i][j] == 0):
          TN +=1

    accuracy = (TP+TN)/(P+N)
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy





