import numpy as np
import cv2
from utils import update_xy_w_z,  rectify_rot
from pysixd import misc
from scipy import spatial

def forward_real275(_query, sess, enc_x, pose_retr, dec_y):
    if _query.dtype == 'uint8':
        _query = _query / 255.
    if _query.ndim == 3:
        _query = np.expand_dims(_query, 0)
        
    info_pose, recon_y = sess.run([pose_retr, dec_y], feed_dict={enc_x: _query})
    return info_pose, recon_y
    

def preprocess_query(query_bgr, query_depth, query_mask, det_bbox2d, pad_factor=1.2, enc_x_size=128):
    
    img_h, img_w, img_c=query_bgr.shape
    query_depth=np.where(query_mask,0,query_depth)

    for cc in range(0,img_c):
        query_bgr[:,:,cc]=np.where(query_mask,255,query_bgr[:,:,cc])
    
    sx,sy,w,h=det_bbox2d  
    size = int(np.maximum(h,w) * pad_factor)
    left = int(np.max([sx + w // 2 - size // 2, 0]))
    right = int(np.min([sx + w // 2 + size // 2, img_w]))
    top = int(np.max([sy + h // 2 - size // 2, 0]))
    bottom = int(np.min([sy + h // 2 + size // 2, img_h]))

    query_crop = query_bgr[top:bottom, left:right].copy()
    query_crop =cv2.resize(query_crop,(enc_x_size,enc_x_size),interpolation=cv2.INTER_CUBIC)
    
    img_yuv = cv2.cvtColor(query_crop, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    query_crop = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return query_crop, query_depth





def traz_scale_estimation_real275(test_depth,test_est_bb,test_K, train_depth,train_K,train_render_z,train_scale,\
                test_bg_thres=32, train_bg_thres=0.2, max_iterations=2,inliers_ratio=0.99):
    
    #process test mask
    test_mask_bb = np.zeros(test_depth.shape, dtype=np.int32)
    bbox_est=[0,0,0,0]
    bbox_est[0] = max(0, int(test_est_bb[0]))
    bbox_est[1] = max(0, int(test_est_bb[1]))
    bbox_est[2] = min(test_depth.shape[1] - 1, int(test_est_bb[0] + test_est_bb[2]))
    bbox_est[3] = min(test_depth.shape[0] - 1, int(test_est_bb[1] + test_est_bb[3]))
    test_mask_bb[bbox_est[1]:bbox_est[3], bbox_est[0]:bbox_est[2]] = 1


    #test depth->pt
    test_depth = np.where(test_mask_bb, test_depth, test_bg_thres)
    test_depth = np.where(test_depth<test_bg_thres, test_depth,0)
    test_ys, _ = np.where(test_depth>0)
    if len(test_ys)==0:
        return np.array([0,0,1]), 1, np.array([[0,0,0]])
    test_pts = misc.rgbd_to_point_cloud(test_K, test_depth)[0]

    train_depth=np.where(train_depth<1-train_bg_thres,(train_depth-0.5)*train_scale+train_render_z,0)#(0,0,render_z) as center

    if train_depth.ndim==3:
        train_depth=train_depth.squeeze()
    train_ys, _ =np.where(train_depth>0)
    if len(train_ys)==0:
        test_mean_z = np.sum(test_depth) / len(test_ys)
        est_tra=update_xy_w_z(test_mean_z,test_K,test_est_bb).flatten()
        est_scale= 1
        return est_tra, est_scale, test_pts
        
    #RGB-based scale and depth_y
    train_pts=misc.rgbd_to_point_cloud(train_K,train_depth)[0]
    train_pts[:,2]-=train_render_z#(0,0,0 as center)
    train_pts=train_pts/train_scale
    

    totest_kdtree=spatial.cKDTree(test_pts)
    totest_dist_pair, _ =totest_kdtree.query(test_pts,k=min(100,test_pts.shape[0]//10),p=2,n_jobs=-1)
    totest_dist_pair=totest_dist_pair[:,-1].flatten()
    totest_idx_inliers=np.where(totest_dist_pair<0.05)[0]#np.mean(totest_dist_pair)+2*np.std(totest_dist_pair))[0]

    test_pts = test_pts[totest_idx_inliers].copy()
    
    for iter_id in range(0,max_iterations):
        # estimate scale
        
        test_pts_bbox_max,test_pts_bbox_min=np.max(test_pts,axis=0),np.min(test_pts,axis=0)
        train_pts_bbox_max,train_pts_bbox_min=np.max(train_pts,axis=0),np.min(train_pts,axis=0)

        train_3dbbx=max(np.fabs(train_pts_bbox_max[0]),np.fabs(train_pts_bbox_min[0]))
        train_3dbby=max(np.fabs(train_pts_bbox_max[1]),np.fabs(train_pts_bbox_min[1]))

        test_size=(test_pts_bbox_max[0]-test_pts_bbox_min[0])**2+(test_pts_bbox_max[1]-test_pts_bbox_min[1])**2#
        train_size=(2*train_3dbbx)**2+(2*train_3dbby)**2#
        est_scale= (test_size / train_size) ** 0.5
        
        # first calculate mean depth and tra z
        test_mean_z = np.mean(test_pts[:,2])
        train_mean_z = est_scale*np.mean(train_pts[:,2])
        est_z=test_mean_z-train_mean_z

        # update tra x/y
        est_tra=update_xy_w_z(est_z,test_K,test_est_bb)

        transformed_train_pts=est_scale*train_pts+est_tra.reshape((1,3))
       
        # check outliers
        pref_num_train_pts=transformed_train_pts.shape[0]
        totest_kdtree=spatial.cKDTree(test_pts)
        totest_dist_pair, _ =totest_kdtree.query(transformed_train_pts,k=1,p=2,n_jobs=-1)
        totest_idx_inliers=np.where(totest_dist_pair<max(0.05,np.mean(totest_dist_pair)+np.std(totest_dist_pair)))[0]
        
        postf_num_train_pts=totest_idx_inliers.shape[0]
        train_pts = train_pts[totest_idx_inliers]
        
        pref_num_test_pts=test_pts.shape[0]
        totrain_kdtree=spatial.cKDTree(transformed_train_pts)
        totrain_dist_pair, _ =totrain_kdtree.query(test_pts,k=1,p=2,n_jobs=-1)
        totrain_idx_inliers=np.where(totrain_dist_pair<max(0.05,np.mean(totrain_dist_pair)+np.std(totrain_dist_pair)))[0]
       
        postf_num_test_pts=totrain_idx_inliers.shape[0]
        test_pts = test_pts[totrain_idx_inliers]


        if postf_num_train_pts>=inliers_ratio*pref_num_train_pts and postf_num_test_pts>=inliers_ratio*pref_num_test_pts:
            break

    return est_tra.flatten(), est_scale, train_pts


def pose_estimation_real275(sess, enc_x, pose_retr, dec_y, query_bgr, query_depth, det_info, query_mask, K_test, embedding_rotmat):
    query_crop,query_masked_depth=preprocess_query(query_bgr,query_depth,query_mask,det_info['obj_bb'])
    info_pose,recon_y=forward_real275(query_crop, sess, enc_x, pose_retr, dec_y)

    recon_depth = (recon_y['x_depth'][0]).astype(np.float32)
    pose_idx = info_pose['encoding_indices'][0]

    R_cb= embedding_rotmat[pose_idx]
    K_train=np.array([[140., 0, 64.], [0., 140., 64], [0., 0., 1.]]).reshape((3, 3))

    Radius_render_train = 1.0
    train_scale= 1.0
    K_test=np.array(K_test).reshape((3, 3))


    t,s, dec_pts = traz_scale_estimation_real275(test_depth=query_masked_depth,
                                                        test_est_bb=det_info['obj_bb'],
                                                        test_K=K_test.copy(),
                                                        train_depth=recon_depth.copy(),
                                                        train_K=K_train.copy(),
                                                        train_render_z=Radius_render_train,
                                                        train_scale=train_scale)
                
    mesh_pts=misc.transform_pts_Rt(dec_pts,np.linalg.inv(R_cb),np.array([0,0,0]))
    sx,sy,sz=np.min(mesh_pts,axis=0)
    ex,ey,ez=np.max(mesh_pts,axis=0)
    lx=2*max(np.fabs(sx),np.fabs(ex))
    ly=2*max(np.fabs(sy),np.fabs(ey))
    lz=2*max(np.fabs(sz),np.fabs(ez))
    pred_scales=[lx,ly,lz]

    R = rectify_rot(R_cb,t)

    print('Estimation result')
    print('2D object bounding box',det_info['obj_bb'])
    print('R_m2c',R)
    print('t_m2c(unit: m)',t.squeeze())
    print('s_m2c', s)
    
    pred_RTs = np.eye(4)
    pred_RTs[:3, :3] = s*R
    pred_RTs[:3, 3] = t.reshape((3,))
    print('pred_RTs',pred_RTs)
    print('pred_scales',pred_scales)
    print('score',det_info['score'])
