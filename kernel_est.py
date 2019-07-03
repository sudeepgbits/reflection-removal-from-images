import numpy as np
import scipy.ndimage as sp_ndimage
import cv2
import scipy as sp

import scipy.ndimage.filters as nd_filters


def local_filter(x, order):
    x.sort()
    return x[order]

def ordfilt2(A, order, mask_size):
    return nd_filters.generic_filter(A, lambda x, ord=order: local_filter(x, ord), size=(mask_size, mask_size))

def get_patch(I, x, y, hw):
    #print('x value in patch = ', x )
    if ((x>hw) and (x < I.shape[1]-hw) and (y>hw) and (y< I.shape[0] - hw)):
        p = I[y-hw:y+hw, x-hw:x+hw]
    else :
        p = None
    
    return p

def est_attenuation(I_in, dx, dy):
    num_features = 200
    
    I = (I_in * 255).astype(np.uint8)
    feat_detector = cv2.ORB(nfeatures=num_features)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(I, None)
    
    cns = np.zeros((num_features,2), dtype=np.uint64)
    count = 0
    for i,kp in enumerate(image_1_kp):
        cns[i,:] = kp.pt
        count = count + 1
     
    hw = 18
    score = np.zeros((len(cns),1))
    atten = score * 0
    w = score * 0
    
    for i in range(0,len(cns)):
        
        p1 = get_patch(I, int(cns[i,1]), int(cns[i,0]), hw)
        p2 = get_patch(I, int(cns[i,1] + dx), int(cns[i,0] + dy), hw)
        
        if p1 is not None and p2 is not None:
            p1 = p1 - np.mean(p1)
            p2 = p2 - np.mean(p2)
            score[i] = np.sum(p1 * p2)/np.sum(p1 ** 2)**0.5/np.sum(p2 ** 2)**0.5
            atten[i] = (np.max(p2)-np.min(p2))/(np.max(p1)-np.min(p1))
            if (atten[i] < 1) and (atten[i] > 0):
                w[i] = np.exp(-1* score[i]/(2*(0.2**2)))
    
    c = sum(w * atten)/sum(w)
    
    #print c
    return c

def kernel_est(I_in):
    
    I_in=I_in.astype(np.float32)
    if len(I_in.shape) == 3:
        I_in = cv2.cvtColor(I_in, cv2.COLOR_RGB2GRAY)

    Laplacian = np.array([[0., -1., 0], [-1., 4., -1.],[0, -1., 0]])
    
    resp = cv2.filter2D(I_in, -1, Laplacian, borderType=cv2.BORDER_DEFAULT)
    auto_corr = sp.signal.fftconvolve(resp, resp[::-1, ::-1])
    max_1 = ordfilt2(auto_corr, 24, 5)
    max_2 = ordfilt2(auto_corr, 23, 5)
    
    (x,y) = auto_corr.shape
    auto_corr[(x/2) - 4 : (x/2) + 4, (y/2) - 4 : (y/2)+4]=0
    
    
    c = np.ones(max_1.shape)
    c[(max_1 - max_2)<=70] = 0
    c[auto_corr != max_1] = 0
    candidates = np.where(c == 1)
    if candidates[0].size > 2:
        idx =  np.argmax(auto_corr[candidates])
        dy = candidates[0][idx] - x//2
        dx = candidates[1][idx] - y//2
    else:
        dx = 2
        dy = 0
    c = est_attenuation(I_in, dx, dy)
    
    return[dx, dy, c]
    
    
def two_pulses(dx,dy,c):
    kernel=np.zeros((2*abs(dy)+1, 2*abs(dx)+1),dtype=np.float)
    kernel[abs(dy), abs(dx)] = 1.0
    kernel[abs(dy)+dy, abs(dx)+dx]=c;
    
    return kernel[::-1,::-1]
    
    
def generate(w,h,dx,dy,c):
    I1 = np.zeros((w,h))

    I2 = np.zeros((w,h))
    cv2.circle(I1,(39,39),23,0.4,-1)
    
    I2[9:50, 4:25]=0.3;
    
    If = sp_ndimage.correlate(I2, two_pulses(dx, dy, c), mode='constant')
    I_in = I1 + If
    return I_in