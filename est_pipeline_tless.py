import numpy as np
import cv2
from utils import est_tra_w_tz, rectify_rot

def forward_tless(_query, sess, enc_x, pose_retr):
    if _query.dtype == 'uint8':
        _query = _query / 255.
    if _query.ndim == 3:
        _query = np.expand_dims(_query, 0)

    info_lookup = sess.run([pose_retr], feed_dict={enc_x: _query})
    return info_lookup


def generate_query_crop(query_bgr, det_bbox2d, pad_factor=1.2, enc_x_size=128):
    
    img_h, img_w, _=query_bgr.shape
    
    sx,sy,w,h=det_bbox2d
    size = int(np.maximum(h, w) * pad_factor)
    left = int(np.max([sx + w // 2 - size // 2, 0]))
    right = int(np.min([sx + w // 2 + size // 2, img_w]))
    top = int(np.max([sy + h // 2 - size // 2, 0]))
    bottom = int(np.min([sy + h // 2 + size // 2, img_h]))

    crop = query_bgr[top:bottom, left:right].copy()
    # print 'Original Crop Size: ', crop.shape
    resized_crop = cv2.resize(crop, (enc_x_size,enc_x_size))

    return resized_crop



################################################################################################################
def pose_estimation_tless(sess, enc_x, pose_retr, query_bgr, det_info, K_test, embedding_rotmat, embedding_bbox2d):
    
    query_crop=generate_query_crop(query_bgr,det_info['obj_bb'])
    

    info_lookup = forward_tless(query_crop, sess, enc_x, pose_retr)
    #est rot by pose retrieval
    idx=info_lookup[0]['encoding_indices'][0]
    R_cb= embedding_rotmat[idx]

    #est translation with pin-hole camera model
    K_train =np.array([1075.65, 0, 720 / 2, 0, 1073.90, 540 / 2, 0, 0, 1]).reshape(3, 3)
    Radius_render_train = 700 #unit: mm

    K_test = np.array(K_test).reshape((3, 3))
    K00_ratio = K_test[0, 0] / K_train[0, 0]
    K11_ratio = K_test[1, 1] / K_train[1, 1]
    mean_K_ratio = np.mean([K00_ratio, K11_ratio])

    render_bb = embedding_bbox2d[idx].squeeze()
    est_bb = det_info['obj_bb']
    diag_bb_ratio = np.linalg.norm(np.float32(render_bb[2:])) / np.linalg.norm(np.float32(est_bb[2:]))

    mm_tz = diag_bb_ratio * mean_K_ratio * Radius_render_train

    # object center in image plane (bb center =/= object center)
    center_obj_x_train = render_bb[0] + render_bb[2] / 2. - K_train[0, 2]
    center_obj_y_train = render_bb[1] + render_bb[3] / 2. - K_train[1, 2]

    center_obj_x_test = est_bb[0] + est_bb[2] // 2 - K_test[0, 2]
    center_obj_y_test = est_bb[1] + est_bb[3] // 2 - K_test[1, 2]

    t = est_tra_w_tz(mm_tz, Radius_render_train, K_test, center_obj_x_test,
                                                    center_obj_y_test,
                                                    K_train, center_obj_x_train, center_obj_y_train)

    R = rectify_rot(R_cb,t)

    print('Estimation result')
    print('2D object bounding box',det_info['obj_bb'])
    print('R_m2c',R)
    print('t_m2c(unit: mm)',t.squeeze())
    print('score',det_info['score'])
    
    