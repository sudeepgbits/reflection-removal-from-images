import numpy as np
import scipy as sp
import scipy.io as spio

import kernel_est
import cv2

import grad_irls
import sys

class configs:
    dx = 15
    dy = 7
    c = 0.5
    padding = 0
    match_input = 0
    linear = 0
    h = 64
    w = 64
    num_px = 4096
    ch = 1
    non_negative = 1
    beta_factor = 2
    beta_i = 200
    dims = [64,64]
    delta = 1.00e-04
    p = 0.2
    use_lap = 1
    use_diagonal = 1
    use_lap2 = 1
    use_cross = 0
    niter = 20
    
def get_k(h, w, dx, dy, c):
    idx = np.arange((h*w)).reshape(w,h).transpose()
    idx_shift = np.roll(idx,(dy,dx),axis=(0,1))
    data = np.ones((h,w),dtype=np.float)
    data_c = c * np.ones((h,w),dtype=np.float)
    data_c[0:dy,:] = 0
    data_c[:,0:dx] = 0
    return sp.sparse.coo_matrix((data.ravel(),(idx.ravel(),idx.ravel())),shape=(h*w,h*w)) + sp.sparse.coo_matrix((data_c.ravel(),(idx.ravel(),idx_shift.ravel())),shape=(h*w,h*w))
    
def merge_patches(patch,h,w,psize):
    patches = patch.transpose().reshape((h-psize+1, w-psize+1, psize**2),order='F')
    out = np.zeros((h, w))
    k=0;
    for j in range(psize):
      for i in range(psize):
        out[i:i+h-psize+1, j:j+w-psize+1] = out[i:i+h-psize+1, j:j+w-psize+1] + patches[:,:,k]; 
        k=k+1;
    return out
def merge_two_patches(est_t, est_r, h, w, psize):
    t_merge = merge_patches(est_t, h, w, psize).ravel(order='F')
    r_merge = merge_patches(est_r, h, w, psize).ravel(order='F')
    return np.hstack([t_merge.ravel(order='F'),r_merge.ravel(order='F')]).reshape(-1,1)
    
def im2patches(img, psize):
    m,n = img.shape
    s0, s1 = img.strides    
    nrows = m-psize+1
    ncols = n-psize+1
    shp = psize,psize,nrows,ncols
    strd = s0,s1,s0,s1

    return np.lib.stride_tricks.as_strided(img, shape=shp, strides=strd).reshape(psize**2,-1,order='F')

def loggausspdf2(x,sigma):
    d = x.shape[0];
    R = np.linalg.cholesky(sigma).T
    B = np.linalg.solve(R,x)
    q = np.sum(B**2,axis=0) # quadratic term (M distance)
    c = d*np.log(2*np.pi)+2*np.sum(np.log(np.diag(R)));   # normalization constant
    y = -(c+q)/2;
    return y
def aprxMAPGMM(x,psize,noiseSD,h,w,GS):
    SigmaNoise = (noiseSD**2)*np.identity(psize);
    SigmaNoise1 = (noiseSD**2)*np.identity(psize**2);
    mean_x = np.mean(x,axis=0).reshape((1,-1))
    x = x - mean_x
    
    dim = GS['dim']+0
    nmodels = GS['nmodels'] + 0
    means = GS['means'] + 0
    covs = GS['covs'] + 0
    invcovs = GS['invcovs'] + 0
    mixweights = GS['mixweights'] + 0
    
    biased_covs = covs + np.tile(SigmaNoise.ravel(order='F').reshape(-1,1),[1,1,nmodels])
    PYZ = np.zeros((nmodels,x.shape[1]))
    for i in range(nmodels):
        PYZ[i,:] = np.log(mixweights[i]) + loggausspdf2(x,biased_covs[:,:,i])
    
    # find the most likely component for each patch
    ks = np.argmax(PYZ,axis=0)

    #and now perform weiner filtering
    Xhat = np.zeros(x.shape).reshape(dim,-1);
    for i in range(nmodels):
        inds = np.where(ks==i)
        Xhat[:,inds[0]] = np.linalg.solve((covs[:,:,i]+SigmaNoise1),(np.dot(covs[:,:,i],x[:,inds].reshape(dim,-1)) + np.dot(SigmaNoise1,np.tile(means[:,i].ravel().reshape(-1,1),[1,inds[0].size]))))

    return Xhat + mean_x
    
def patch_gmm(img,dx,dy,c):
    (h,w) = img.shape
    id_mat = sp.sparse.identity((h*w))
    k_mat = get_k(h, w, dx, dy, c)
    A = sp.sparse.hstack([id_mat,k_mat])
    
    lbda = 1e6;
    psize = 8;
    num_patches = (h-psize+1)*(w-psize+1)
    mask=merge_two_patches(np.ones((psize**2, num_patches)),np.ones((psize**2, num_patches)), h, w, psize)
    
    beta = 200
    beta_factor = 2
    mat = spio.loadmat('GSModel_8x8_200_2M_noDC_zeromean.mat', squeeze_me=True)
    GS = mat['GS']
    configs1 = configs()
    configs1.dx = dx
    configs1.dy = dy
    configs1.c = c
    configs1.h = h
    configs1.w = w
    configs1.num_px = h*w
    (I_t_i,I_r_i,configs1) = grad_irls.grad_irls(img,configs1)
    
    est_t = im2patches(I_t_i, psize)
    est_r = im2patches(I_r_i, psize)
    
        
    for i in range(25):
        print 'Optimizine %d iter...\n'%i
        x0 = np.hstack([I_t_i.ravel(order='F'),I_r_i.ravel(order='F')]).reshape(-1,1)
        sum_piT_zi = merge_two_patches(est_t, est_r, h, w, psize);
        sum_zi_2 = np.linalg.norm(est_t.ravel(order='F'))**2 + np.linalg.norm(est_r.ravel(order='F'))**2;
        z = (lbda * A.T.tocsr().dot(img.ravel(order='F'))).reshape(-1,1) +  (beta * sum_piT_zi); 
        def calc_func_value_and_gradient(x,*args):
            f = lbda * np.linalg.norm(A.dot(x) - img.ravel(order='F'))**2 + beta*(sum(x.reshape(-1,1)*mask*x.reshape(-1,1) - 2 * x.reshape(-1,1)* sum_piT_zi.ravel(order='F').reshape(-1,1)) + sum_zi_2)
            g = 2*(lbda * (A.T.tocsr().dot(A.dot(x))).reshape(-1,1) + beta*(mask*x.reshape(-1,1)) - z)
            return f,g
        (out,f,d) = sp.optimize.fmin_l_bfgs_b(calc_func_value_and_gradient, x0,args=(A,mask),bounds=[(0,1) for i in range(h*w*2)],m=50,factr=1e4,pgtol=1e-8)
        
        out = out.reshape(h,w,2,order='F')
        I_t_i = out[:,:,0]
        I_r_i = out[:,:,1] 

        #Restore patches using the prior
        est_t = im2patches(I_t_i, psize);
        est_r = im2patches(I_r_i, psize);
        noiseSD=(1/beta)**0.5;
        
        est_t = aprxMAPGMM(est_t,psize,noiseSD,h,w, GS)
        est_r = aprxMAPGMM(est_r,psize,noiseSD,h,w, GS)

        beta = beta*beta_factor;
    return I_t_i,I_r_i

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'insufficient options'
        print 'python reflection_removal.py option'
        sys.exit(0)
    option = int(sys.argv[1])
    # to use apples image from paper
    if option == 1:
        print 'Running image from paper.. It will 20hrs to run'
        mat = spio.loadmat('apples.mat', squeeze_me=True)
        I_in = mat['I_in'] # array
        (dx,dy,c)= kernel_est.kernel_est(I_in)
        result = 'apples'
    #to use  synthatic image
    if option == 2:
        print 'Generating synthatic image.. It will take 5 mins'
        dx = 15
        dy = 7
        c = 0.5
        I_in = kernel_est.generate(128,128,dx,dy,c)
        result = 'synthatic'
    if option == 3:
        print 'running test image..  '
        if len(sys.argv) < 3:
            print 'please provide input image filename as second arguement'
            sys.exit(0)
        result = str(sys.argv[2])
        I = cv2.imread(result)
        I_in = I.copy()
        I_in = I_in.astype(np.float)
        I_in = I_in /255.
        (dx,dy,c)= kernel_est.kernel_est(I_in)
        
        
    if len(I_in.shape) == 2:
        (h,w) = I_in.shape
        ch = 1
    else:
        (h,w,ch) = I_in.shape
    I_in = I_in.reshape((h,w,ch))
    I_out_t = np.zeros((h,w,ch))
    I_out_r = np.zeros((h,w,ch))
    for i in range(ch):
        I_t,I_r = patch_gmm(I_in[:,:,i],dx,dy,c)
        I_out_t[:,:,i] = I_t
        I_out_r[:,:,i] = I_r
    if len(I_in.shape) == 2:
        cv2.imwrite('output_t_'+result+'.png',I_out_t.reshape(h,w)*255)
        cv2.imwrite('output_r_'+result+'.png',I_out_r.reshape(h,w)*255)
    else:
        cv2.imwrite('output_t_'+result+'.png',I_out_t.reshape(h,w,ch)*255)
        cv2.imwrite('output_r_'+result+'.png',I_out_r.reshape(h,w,ch)*255)
    