# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 01:24:57 2017

@author: SUDEEP
"""

import numpy as np
import cv2
from numpy.linalg import inv
import scipy.ndimage as sp_ndimage

import scipy.ndimage.filters as nd_filters
import scipy.signal as spsig
import scipy as sp
from math import exp
from scipy import sparse
from scipy.sparse import spdiags
from scipy.fftpack import dst, idst
from scipy.sparse import csr_matrix
import scipy.io as spio

from math import cos
import time
import reflection_removal


def get_k_old(h, w, dx, dy, c):
    all_ids = np.zeros((1,h*w), dtype=np.uint64)
    for i in range(0,h*w):
        all_ids[0,i] = i
    all_ids = np.reshape(all_ids, [h, w])
    self_ids=all_ids
    
    #circle shift
    negh_ids2 = np.roll(all_ids, [dx, 0], axis=0)
    negh_ids2 = np.roll(negh_ids2, [0, dy], axis=1)
    
    #ncircle shift
    negh_ids = np.roll(all_ids, [0, dy], axis=1)
    negh_ids = np.roll(negh_ids, [dx, 0], axis=0)
    negh_ids[:,0:dy] =0
    negh_ids[0:dx,:] =0

    ind = np.ones((h,w), dtype=np.uint64)
    indc = ind * c
    indc[negh_ids==0]=0
    
    #S_plus = sparse.csr_matrix(ind)
    
    S_plus = sparse.csr_matrix((np.asarray(ind.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(self_ids.T).reshape(-1))),shape=(np.size(ind),np.size(ind)),dtype=np.float64)

    #S_minus = sparse.csr_matrix(indc)
    S_minus = sparse.csr_matrix((np.asarray(indc.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(negh_ids2.T).reshape(-1))),shape=(np.size(indc),np.size(indc)),dtype=np.float64)

    A = S_plus + S_minus
    
    return[A, self_ids, negh_ids2, ind, indc]

def get_k(h, w, dx, dy, c):
    idx = np.arange((h*w)).reshape(w,h).transpose()
    idx_shift = np.roll(idx,(dy,dx),axis=(0,1))
    data = np.ones((h,w),dtype=np.float)
    data_c = c * np.ones((h,w),dtype=np.float)
    data_c[0:dy,:] = 0
    data_c[:,0:dx] = 0
    return sp.sparse.coo_matrix((data.ravel(),(idx.ravel(),idx.ravel())),shape=(h*w,h*w)) + sp.sparse.coo_matrix((data_c.ravel(),(idx.ravel(),idx_shift.ravel())),shape=(h*w,h*w))
      
def get_fx(h,w):    
    idx = np.arange((h*w)).reshape(w,h).transpose()
    idx_shift = np.roll(idx,(0,-1),axis=(0,1))
    data = np.ones((h,w),dtype=np.float)
    data_c = data.copy()
    data_c[:,-1] = 0
    return sp.sparse.coo_matrix((data.ravel(),(idx.ravel(),idx.ravel())),shape=(h*w,h*w)) - sp.sparse.coo_matrix((data_c.ravel(),(idx.ravel(),idx_shift.ravel())),shape=(h*w,h*w))
      

def get_fy(h,w):
    idx = np.arange((h*w)).reshape(w,h).transpose()
    idx_shift = np.roll(idx,(-1,0),axis=(0,1))
    data = np.ones((h,w),dtype=np.float)
    data_c = data.copy()
    data_c[-1,:] = 0
    return sp.sparse.coo_matrix((data.ravel(),(idx.ravel(),idx.ravel())),shape=(h*w,h*w)) - sp.sparse.coo_matrix((data_c.ravel(),(idx.ravel(),idx_shift.ravel())),shape=(h*w,h*w))
      

def get_fu(h,w):
    idx = np.arange((h*w)).reshape(w,h).transpose()
    idx_shift = np.roll(idx,(1,-1),axis=(0,1))
    data = np.ones((h,w),dtype=np.float)
    data_c = data.copy()
    data_c[:,-1] = 0
    data_c[0,:] = 0
    return sp.sparse.coo_matrix((data.ravel(),(idx.ravel(),idx.ravel())),shape=(h*w,h*w)) - sp.sparse.coo_matrix((data_c.ravel(),(idx.ravel(),idx_shift.ravel())),shape=(h*w,h*w))
      

def get_fv(h,w):
    idx = np.arange((h*w)).reshape(w,h).transpose()
    idx_shift = np.roll(idx,(-1,-1),axis=(0,1))
    data = np.ones((h,w),dtype=np.float)
    data_c = data.copy()
    data_c[-1,:] = 0
    data_c[:,-1] = 0
    return sp.sparse.coo_matrix((data.ravel(),(idx.ravel(),idx.ravel())),shape=(h*w,h*w)) - sp.sparse.coo_matrix((data_c.ravel(),(idx.ravel(),idx_shift.ravel())),shape=(h*w,h*w))
        

def get_lap(h,w):
    idx = np.arange((h*w)).reshape(w,h).transpose()
    data = np.ones((h,w),dtype=np.float)
    S_plus = sp.sparse.coo_matrix((data.ravel(),(idx.ravel(),idx.ravel())),shape=(h*w,h*w))
    
    
    idx_shift = np.roll(idx,(0,-1),axis=(0,1))
    data_c = data.copy()
    data_c[:,-1] = 0
    S_minus_1 = sp.sparse.coo_matrix((data_c.ravel(),(idx.ravel(),idx_shift.ravel())),shape=(h*w,h*w))
    
    idx_shift2 = np.roll(idx,(0,1),axis=(0,1))
    data_c1 = data.copy()
    data_c1[:,0] = 0
    S_minus_2 = sp.sparse.coo_matrix((data_c1.ravel(),(idx.ravel(),idx_shift2.ravel())),shape=(h*w,h*w))

    idx_shift3 = np.roll(idx,(-1,0),axis=(0,1))
    data_c2 = data.copy()
    data_c2[-1,:] = 0
    S_minus_3 = sp.sparse.coo_matrix((data_c2.ravel(),(idx.ravel(),idx_shift3.ravel())),shape=(h*w,h*w))
    
    
    idx_shift4 = np.roll(idx,(1,0),axis=(0,1))
    data_c3 = data.copy()
    data_c3[0,-1] = 0
    S_minus_4 = sp.sparse.coo_matrix((data_c3.ravel(),(idx.ravel(),idx_shift4.ravel())),shape=(h*w,h*w))
    
    A = 4*S_plus - S_minus_1 -S_minus_2 - S_minus_3 - S_minus_4
    return  A


def irls_grad(I_x, tx, out_xi, mh, configs, mx, my,  mu, mv, mlap):
    p = configs.p
    num_px=configs.num_px
    out_x=out_xi
    if configs.use_cross:
        mcross = mx.dot(my)
    if configs.use_lap2:
        mx2 = mx.dot(mx)
        my2 = my.dot(my)
        
    for i in range(0,configs.niter):
        if (configs.delta == 'exp_fall'):
            delta = 0.01*exp(-(i-5)*0.4)
        else:
            delta=configs.delta
        
        out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
        
        w1 = (abs(out_x1) **2 + delta)**(p/2-1)
        I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
        w2 = (abs((mh.dot((I_x1 - out_x1))))**2 + delta) ** (p/2-1)
        
        data2= np.reshape(w1, (1,np.size(w1)), order='F')
        A1 = spdiags(data2, 0, num_px, num_px) 
        data2= np.reshape(w2, (1,np.size(w2)), order='F')
        
        A2_temp = (spdiags(data2, 0, num_px, num_px)).dot(mh)
        A2 = (np.transpose(mh)).dot(A2_temp)
        
        Atot = A1 + A2
        Ab = A2
        
        if configs.use_lap :
            out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
            I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
            w3 = (abs((mx.dot(out_x1)))**2 + delta)**(p/2-1)
            w4 = (abs((my.dot(out_x1)))**2 + delta)**(p/2-1)
            val_w5 = mx.dot(mh)
            val_w5 = val_w5.dot(I_x1 - out_x1)
            w5 = (abs(val_w5)**2 + delta)**(p/2-1)
            val_w6 = my.dot(mh)
            val_w6 = val_w6.dot(I_x1 - out_x1)
            w6 = (abs(val_w6)**2 + delta)**(p/2-1)
    
            A3 = (np.transpose(mx)).dot(spdiags_cal(w3,0,num_px,num_px))
            A3 = A3.dot(mx)
            
            A4 = (np.transpose(my)).dot(spdiags_cal(w4,0,num_px,num_px))
            A4 = A4.dot(my)
            
            A5 = (np.transpose(mx)).dot(spdiags_cal(w5,0,num_px,num_px))
            A5 = A5.dot(mx)
            
            A6 = (np.transpose(my)).dot(spdiags_cal(w6,0,num_px,num_px))
            A6 = A6.dot(my)
            
            A7 = (np.transpose(mh)).dot((A5+A6))
            A7 = A7.dot(mh)
            
            Atot = Atot+A3+A4+A7
            Ab = Ab + A7
            
            
        if configs.use_diagnoal:
            out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
            I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
            
            w8 = (abs(mu.dot(out_x1))**2 + delta)**(p/2-1)
            w9 = (abs(mv.dot(out_x1))**2 + delta)**(p/2-1)
            
            w10_val = mu.dot(mh)
            w10_val = w10_val.dot((I_x1-out_x1))
            w10 = (abs(w10_val)**2 + delta)**(p/2-1)
            
            w11_val = mv.dot(mh)
            w11_val = w11_val.dot((I_x1-out_x1))
            w11 = (abs(w11_val)**2 + delta)**(p/2-1)
            
            A8 = (np.transpose(mu)).dot(spdiags_cal(w8,0,num_px,num_px))
            A8 = A8.dot(mu)
            
            A9 = (np.transpose(mv)).dot(spdiags_cal(w9,0,num_px,num_px))
            A9 = A9.dot(mv)
            
            A10 = (np.transpose(mu)).dot(spdiags_cal(w10,0,num_px,num_px))
            A10 = A10.dot(mu)
            
            A11 = (np.transpose(mv)).dot(spdiags_cal(w11,0,num_px,num_px))
            A11 = A11.dot(mv)
            
            A12 = (np.transpose(mh)).dot((A10+A11))
            A12 = A12.dot(mh)
            
            Atot = Atot+A8+A9+A12
            Ab = Ab+A12
            
        if configs.use_lap2:
            out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
            I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
            
            w17 = (abs(mx2.dot(out_x1))**2 + delta)**(p/2-1)
            w18 = (abs(my2.dot(out_x1))**2 + delta)**(p/2-1)
            
            w19_val = mx2.dot(mh)
            w19_val = w19_val.dot((I_x1-out_x1))
            w19 = (abs(w19_val)**2 + delta)**(p/2-1)
            
            w20_val = my2.dot(mh)
            w20_val = w20_val.dot((I_x1-out_x1))
            w20 = (abs(w20_val)**2 + delta)**(p/2-1)
            
            A17 = (np.transpose(mx2)).dot(spdiags_cal(w17,0,num_px,num_px))
            A17 = A17.dot(mx2)
            
            A18 = (np.transpose(my2)).dot(spdiags_cal(w18,0,num_px,num_px))
            A18 = A18.dot(my2)
            
            A19 = (np.transpose(mx2)).dot(spdiags_cal(w19,0,num_px,num_px))
            A19 = A19.dot(mx2)
            
            A20 = (np.transpose(my2)).dot(spdiags_cal(w20,0,num_px,num_px))
            A20 = A20.dot(my2)
            
            A21 = (np.transpose(mh)).dot((A19+A20))
            A21 = A21.dot(mh)
            
            Atot=Atot+A17+A18+A21
            Ab=Ab+A21
            
        if configs.use_cross :
            out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
            I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
            w15 = (abs(mcross.dot(out_x1))**2 + delta)**(p/2-1)
            
            w16_val = mcross.dot(mh)
            w16_val = w16_val.dot((I_x1-out_x1))
            w16 = (abs(w16_val)**2 + delta)**(p/2-1)
            
            A15 = (np.transpose(mcross)).dot(spdiags_cal(w15,0,num_px,num_px))
            A15 = np.dot(A15,mcross)
            
            A16 = (np.transpose(mh)).dot(np.transpose(mcross))
            A16 = A16.dot(spdiags_cal(w16,0,num_px,num_px))
            A16 = A16.dot(mcross)
            A16 = A16.dot(mh)
            Atot=Atot+A15+A16
            
            Ab=Ab+A16
        
        
        
        I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
        
        out_x = sp.sparse.linalg.spsolve(Atot,(Ab.dot(I_x1)))
       
        out_x = np.reshape(out_x, (np.size(out_x),1))
        res = I_x1 - out_x
        
            
    out_x = np.reshape(out_x, configs.dims, order = 'F')
            
            
    return out_x
    
  
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
def grad_irls(I_in, configs):
    
    dx = configs.dx
    dy = configs.dy
    c = configs.c
    configs.dims=[np.shape(I_in)[0], np.shape(I_in)[1]]
    dims = configs.dims
    
    configs.delta= 1 * (10**(-4))
    configs.p=0.2
    configs.use_lap=1
    configs.use_diagnoal=1
    configs.use_lap2=1
    configs.use_cross=0
    configs.niter=20

    A = reflection_removal.get_k(configs.h, configs.w, dx, dy, c)


    mk = A
    
    mh = sp.sparse.linalg.inv(mk.tocsc())

    
    Ax = get_fx(configs.h, configs.w)
    mx = Ax
    Ay = get_fy(configs.h, configs.w)
    my = Ay
    Au = get_fu(configs.h, configs.w)
    mu = Au
    Av = get_fv(configs.h, configs.w)
    mv = Av

    Alap = get_lap(configs.h, configs.w)
    mlap = Alap
    
    k = configs.ch
    
    kernel = np.array([0,-1, 1]).reshape(1,3)
        
    I_x = sp_ndimage.correlate(I_in,kernel,mode='constant')
    
    kernel = np.array([[0],[-1], [1]])
    
    I_y = sp_ndimage.correlate(I_in,kernel,mode='constant')
    out_xi=I_x/2
    out_yi=I_y/2

    out_x=irls_grad(I_x, [], out_xi, mh, configs, mx, my,  mu, mv, mlap)
    
    out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
    I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
    outr_x = np.reshape(mh.dot((I_x1-out_x1)), (dims),order = 'F')
    
    out_y=irls_grad(I_y, [], out_yi, mh, configs, mx, my, mu, mv, mlap)
    out_y1 = np.reshape(out_y, (np.size(out_y),1), order='F')
    I_y1 = np.reshape(I_y, (np.size(I_y),1), order='F')
    outr_y = np.reshape(mh.dot((I_y1-out_y1)), (dims),order='F')
    
    I_t = poisson_solver_function(out_x, out_y, I_in)
    I_r = poisson_solver_function(outr_x, outr_y, I_in)
    
    return [I_t, I_r, configs]


def poisson_solver_function(gx,gy,in_img):
    boundary_image = in_img.copy()
    [H,W] = np.shape(boundary_image)
    gxx = np.zeros((H,W))
    gyy = np.zeros((H,W))
    
    
    for m in range(0,H-1):
        for n in range(0,W-1):           
            gyy[m+1,n] = (gy[m+1,n] - gy[m,n])
            gxx[m,n+1] = gx[m,n+1] - gx[m,n]
            

    f = gxx + gyy

    
    boundary_image[1:-1, 1:-1] = 0
    

    f_bp = np.zeros((H,W))
    
    for j in range (1,H-1):
        for k in range(1,W-1):
            f_bp[j,k] = -4*boundary_image[j,k] + boundary_image[j,k+1] + boundary_image[j,k-1] + boundary_image[j-1,k] + boundary_image[j+1,k]
    
    f1 = f - f_bp
    
    f2 = f1[1:-1, 1:-1]
    
    tt = dst(f2.T,1).T
    f2sin = dst(tt,1)

        
    nx, ny = (W-2, H-2)
    x = np.linspace(1, W-2, nx)
    y = np.linspace(1, H-2, ny)
    xv, yv = np.meshgrid(x, y)

    denom = (2*np.cos(np.pi*xv/(W-1))-2) + (2*np.cos(np.pi*yv/(H-1)) - 2)
    f3 = f2sin / denom
    tt = (idst(f3.T,1).T)/(f3.shape[0]*2 + 2)
    
    img_tt = idst(tt,1)/(tt.shape[1]*2 + 2)
    img_direct = boundary_image.copy()
    img_direct[1:-1,1:-1] = 0        
    img_direct[1:-1,1:-1] = img_tt
       
    return img_direct

def spdiags_cal(x, t, num_px, num_px2):
    
    data2= np.reshape(x, (1,np.size(x)), order='F')
    spdiageI = spdiags(data2, t, num_px, num_px2)
    
    return spdiageI       
    
    