import argparse

import torch
import numpy as np
from losses import multiview_iou_loss
import imageio
import soft_renderer as sr
import soft_renderer.functional as srf
import datasets
import models
import time
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils import img_cvt

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')

RANDOM_SEED = 42

BATCH_SIZE = 24
LEARNING_RATE = 1e-4

SIGMA_VAL = 1e-4
LAMBDA_LAPLACIAN = 5e-3
LAMBDA_FLATTEN = 5e-4

IMAGE_SIZE = 64
NUM_VIEWS = 2

MODEL_NAME = "Baseline"
RESUME_MODEL_NAME = MODEL_NAME
MODEL_PATH = "./models_saved"
DATA_PATH = "./data/datasets"

DEMO_PATH = "./mesh_demo/"
DEMO_EVERY = 25

LAST_EPOCH = 0
NUM_EPOCHS = 100

parser = argparse.ArgumentParser()
parser.add_argument('-mp', '--model-path', type=str, default=MODEL_PATH)
parser.add_argument('-mn', '--model-name', type=str, default=MODEL_NAME)
parser.add_argument('-rmn', '--resume_model-name', type=str, default=RESUME_MODEL_NAME)
parser.add_argument('-dp', '--data-path', type=str, default=DATA_PATH)
parser.add_argument('-cls', '--class-ids', type=str, default=CLASS_IDS_ALL)

parser.add_argument('-nv', '--num-views', type=int, default=NUM_VIEWS)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)

parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('-lr', '--learning-rate', type=float, default=LEARNING_RATE)

parser.add_argument('-ne', '--num-epochs', type=int, default=NUM_EPOCHS)
parser.add_argument('-le', '--last-epoch', type=int, default=LAST_EPOCH)

parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)
parser.add_argument('-ll', '--lambda-laplacian', type=float, default=LAMBDA_LAPLACIAN)
parser.add_argument('-lf', '--lambda-flatten', type=float, default=LAMBDA_FLATTEN)

parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)
parser.add_argument('-dmp', '--demo_path', type=str, default=DEMO_PATH)
parser.add_argument('-de', '--demo_every', type=int, default=DEMO_EVERY)
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

if args.num_views <= 1:
    raise Exception

def voxel2mesh(voxels, surface_view):
    # taken from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0
    
    voxels = voxels.cpu().numpy()
    
    positions = np.where(voxels > 0.3)
    voxels[positions] = 1 
    for i, j, k in zip(*positions):
        # identifies if current voxel has an exposed face 
        if not surface_view or np.sum(voxels[i-1:i+2, j-1:j+2, k-1:k+2]) < 27:
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)  
              
    return np.array(verts), np.array(faces)
    
    
def write_obj(filename, verts, faces):
    # taken from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py:
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))

    
def voxel2obj(filename, pred, surface_view = True):
    # taken from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py
    verts, faces = voxel2mesh(pred, surface_view)
    write_obj(filename, verts, faces)

test_dataset = datasets.ShapeNet(args.num_views, DATA_PATH, args.class_ids.split(','), 'test')
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model_dir = args.model_path+"/"+args.model_name
resume_model_dir = args.model_path+"/"+args.resume_model_name

model = models.Model('./data/obj/sphere/sphere_642.obj', args=args)

if args.last_epoch != 0:
  checkpoint = torch.load(resume_model_dir+"_"+str(args.last_epoch)+".pth")
  model.load_state_dict(checkpoint["model"])
  #optimizer.load_state_dict(checkpoint['optimizer'])

if torch.cuda.is_available():
    model = model.cuda()

model.eval()
val_losses = []
val_ious = []

with torch.no_grad():
    for batch, data in enumerate(test_dataloader):
        images = torch.autograd.Variable(data["img"][:,0,:,:,:]).cuda()
        viewpoints = data["view"][:,0,:]
        voxels = data["vox"].numpy()
        
        loss = 0
        for v in range(1,args.num_views):
            images_ = torch.autograd.Variable(data["img"][:,v,:,:,:]).cuda()
            viewpoints_ = data["view"][:,v,:]
            
            render_images, laplacian_loss, flatten_loss, batch_iou = model([images, images_], [viewpoints, viewpoints_], voxels, task='train')
            laplacian_loss = laplacian_loss.mean()
            flatten_loss = flatten_loss.mean()
            
            if v==1:
                render_images_demo = render_images

            loss += (multiview_iou_loss(render_images, images, images_) + \
            args.lambda_laplacian * laplacian_loss + \
            args.lambda_flatten * flatten_loss) / (args.num_views-1)
            
        val_losses.append(loss.data.item())
        val_ious.append(batch_iou.mean())

        if batch%args.demo_every==0:
            demo_image = images[0:1]
            obj_path = args.demo_path + str(args.last_epoch) +"-"+ str(batch) + "_mesh.obj"
            demo_v, demo_f = model.reconstruct(demo_image)
            srf.save_obj(obj_path, demo_v[0], demo_f[0])
            
            obj_path_ = args.demo_path + str(args.last_epoch) +"-"+ str(batch) + "_truth.obj"
            voxel2obj(obj_path_, data["vox"][0,:,:,:])
        
            img1_path = args.demo_path + str(args.last_epoch) +"-"+ str(batch) + "_render.png"
            img2_path = args.demo_path + str(args.last_epoch) +"-"+ str(batch) + "_input.png"
            
            imageio.imsave(img1_path, img_cvt(render_images_demo[0][0]))
            imageio.imsave(img2_path, img_cvt(images[0]))

mean_val_loss = np.mean(val_losses)
mean_val_iou = np.mean(val_ious)
print('Mean Testing Loss: ' + str(mean_val_loss))
print('Mean Testing IOU: ' + str(mean_val_iou))