import argparse

import torch
import numpy as np
from losses import multiview_iou_loss
import soft_renderer as sr
import soft_renderer.functional as srf
import datasets
import models
import time
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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

MODEL_NAME = "AddTransformer_v3"
MODEL_PATH = "./models_saved"
DATA_PATH = "./data/datasets"

LAST_EPOCH = 0
NUM_EPOCHS = 100

SAVE_EPOCH = 5

parser = argparse.ArgumentParser()
parser.add_argument('-mp', '--model-path', type=str, default=MODEL_PATH)
parser.add_argument('-mn', '--model-name', type=str, default=MODEL_NAME)
parser.add_argument('-dp', '--data-path', type=str, default=DATA_PATH)
parser.add_argument('-cls', '--class-ids', type=str, default=CLASS_IDS_ALL)

parser.add_argument('-nv', '--num-views', type=int, default=NUM_VIEWS)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)

parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('-lr', '--learning-rate', type=float, default=LEARNING_RATE)

parser.add_argument('-ne', '--num-epochs', type=int, default=NUM_EPOCHS)
parser.add_argument('-le', '--last-epoch', type=int, default=LAST_EPOCH)
parser.add_argument('-se', '--save-epoch', type=int, default=SAVE_EPOCH)

parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)
parser.add_argument('-ll', '--lambda-laplacian', type=float, default=LAMBDA_LAPLACIAN)
parser.add_argument('-lf', '--lambda-flatten', type=float, default=LAMBDA_FLATTEN)

parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

if args.num_views <= 1:
    raise Exception

train_dataset = datasets.ShapeNet(args.num_views, DATA_PATH, args.class_ids.split(','), 'train')
val_dataset = datasets.ShapeNet(args.num_views, DATA_PATH, args.class_ids.split(','), 'val')

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

model = models.Model('./data/obj/sphere/sphere_642.obj', args=args)
optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)

model_dir = args.model_path+"/"+args.model_name
if args.last_epoch != 0:
  checkpoint = torch.load(model_dir+"_"+str(args.last_epoch)+".pth")
  model.load_state_dict(checkpoint["model"])
  optimizer.load_state_dict(checkpoint['optimizer'])

if torch.cuda.is_available():
    model = model.cuda()

epoch_train_losses = []
epoch_val_losses = []
epoch_train_ious = []
epoch_val_ious = []

for epoch in range(args.last_epoch + 1, args.num_epochs + 1):
    model.train()
    batch_losses = []
    batch_ious = []
    print('Epoch ' + str(epoch) + '/' + str(args.num_epochs))
    for batch, data in enumerate(train_dataloader):
        #print('Epoch ' + str(epoch) + '/' + str(args.num_epochs) + ', Batch ' + str(batch) + '/' + str(len(train_dataloader)))
        # soft render images
        optimizer.zero_grad()
        
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
            
            loss += (multiview_iou_loss(render_images, images, images_) + \
            args.lambda_laplacian * laplacian_loss + \
            args.lambda_flatten * flatten_loss) / (args.num_views-1)

        # compute gradient and optimize
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss.data.item())
        batch_ious.append(batch_iou.mean())
    
    mean_batch_loss = np.mean(batch_losses)
    mean_batch_iou = np.mean(batch_ious)
    print('Mean Training Loss: ' + str(mean_batch_loss))
    print('Mean Training IOU: ' + str(mean_batch_iou))
    epoch_train_losses.append(mean_batch_loss)
    epoch_train_ious.append(mean_batch_iou)
    
    model.eval()
    val_losses = []
    val_ious = []
    
    with torch.no_grad():
        for batch, data in enumerate(val_dataloader):
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
                
                loss += (multiview_iou_loss(render_images, images, images_) + \
                args.lambda_laplacian * laplacian_loss + \
                args.lambda_flatten * flatten_loss) / (args.num_views-1)
                
            val_losses.append(loss.data.item())
            val_ious.append(batch_iou.mean())
   
    mean_val_loss = np.mean(val_losses)
    mean_val_iou = np.mean(val_ious)
    print('Mean Validation Loss: ' + str(mean_val_loss))
    print('Mean Validation IOU: ' + str(mean_val_iou))
    epoch_val_losses.append(mean_val_loss)
    epoch_val_ious.append(mean_val_iou)
    
    if (epoch) % args.save_epoch == 0: 
        torch.save({'model': model.state_dict(),'optimizer': optimizer.state_dict()}, model_dir+"_"+str(epoch)+".pth")

plt.plot(range(1+args.last_epoch,1+args.last_epoch+len(epoch_train_losses)),epoch_train_losses,label="Training Loss")
plt.plot(range(1+args.last_epoch,1+args.last_epoch+len(epoch_val_losses)),epoch_val_losses,label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Change over Epochs")
plt.legend()
plt.savefig('Train_Val_Loss_v3.png')
#plt.show()

plt.clf()

plt.plot(range(1+args.last_epoch,1+args.last_epoch+len(epoch_train_ious)),epoch_train_ious,label="Training IOU")
plt.plot(range(1+args.last_epoch,1+args.last_epoch+len(epoch_val_ious)),epoch_val_ious,label="Validation IOU")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation IOU Change over Epochs")
plt.legend()
plt.savefig('Train_Val_IOU_v3.png')
#plt.show()