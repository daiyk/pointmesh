# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:55:31 2018

@author: nanyang wang
"""

import os,sys
import numpy as np
# import cv2
import trimesh
import sklearn.preprocessing

TRAINING_FILES = True # change for preparing training or testing

def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3]*np.sin(phi)
    temp = param[3]*np.cos(phi)
    camX = temp * np.cos(theta)    
    camZ = temp * np.sin(theta)        
    cam_pos = np.array([camX, camY, camZ])        

    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
    return cam_mat, cam_pos

if __name__ == '__main__':
    
    # 1 sampling
    obj_paths = 'model/'
    obj_prefix = 'data/model/'
    # get all .obj file in the directory
    obj_files = []
    file_list = os.listdir(obj_paths)
    for i in file_list:
        if os.path.splitext(i)[1] == '.obj':
            obj_files.append(i)
    
    # record all train files
    if(TRAINING_FILES):
        file_list = open('train_list.txt','w+')
    else:
        file_list = open('test_list.txt','w+')

    # transform all obj file
    for obj_path in obj_files:
        mesh_list = trimesh.load_mesh(obj_paths+obj_path)
    if not isinstance(mesh_list, list):
        mesh_list = [mesh_list]
    area_sum = 0
    for mesh in mesh_list:
        area_sum += np.sum(mesh.area_faces)

    sample = np.zeros((0,3), dtype=np.float32)
    normal = np.zeros((0,3), dtype=np.float32)


    for mesh in mesh_list:
        number = int(round(16384*np.sum(mesh.area_faces)/area_sum))
        if number < 1:
            continue
        points, index = trimesh.sample.sample_surface_even(mesh, number)
        sample = np.append(sample, points, axis=0)

        triangles = mesh.triangles[index]
        pt1 = triangles[:,0,:]
        pt2 = triangles[:,1,:]
        pt3 = triangles[:,2,:]
        norm = np.cross(pt3-pt1, pt2-pt1)
        norm = sklearn.preprocessing.normalize(norm, axis=1)
        normal = np.append(normal, norm, axis=0)

    # 2 tranform to camera view
    position = sample * 0.57

    view_path = '1a0bc9ab92c915167ae33d942430658c/rendering/rendering_metadata.txt'
    cam_params = np.loadtxt(view_path)
    for index, param in enumerate(cam_params):
        # camera tranform
        cam_mat, cam_pos = camera_info(param)

        pt_trans = np.dot(position-cam_pos, cam_mat.transpose())
        nom_trans = np.dot(normal, cam_mat.transpose())
        train_data = np.hstack((pt_trans, nom_trans))
        
        pts_path = os.path.join(os.path.split(obj_path)[0], os.path.splitext(obj_path)[0]+'%02d.dat'%index)
        np.savetxt(obj_paths+pts_path, train_data)

        file_list.write(obj_prefix+pts_path+'\n')
        # save for training

        
        #### project for sure
        # img = cv2.imread(img_path)
        # img = cv2.resize(img, (224,224))
        
        # X,Y,Z = pt_trans.T
        # F = 250
        # h = (-Y)/(-Z)*F + 224/2.0
        # w = X/(-Z)*F + 224/2.0
        # h = np.minimum(np.maximum(h, 0), 223)
        # w = np.minimum(np.maximum(w, 0), 223)
        # img[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
        # img[np.round(h).astype(int), np.round(w).astype(int), 1] = 255
        # cv2.imwrite(img_path.replace('.png','_prj.png'), img)
    file_list.close() 
