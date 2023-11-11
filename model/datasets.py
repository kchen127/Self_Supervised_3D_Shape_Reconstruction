import torch
import numpy as np
import tqdm
import os
import math
from torch.utils.data import Dataset

class_ids_map = {
    '02691156': 'Airplane',
    '02828884': 'Bench',
    '02933112': 'Cabinet',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03211117': 'Display',
    '03636649': 'Lamp',
    '03691459': 'Loudspeaker',
    '04090263': 'Rifle',
    '04256520': 'Sofa',
    '04379243': 'Table',
    '04401088': 'Telephone',
    '04530566': 'Watercraft',
}

class ShapeNet(Dataset):
    def __init__(self, num_views, directory, class_ids, set_name):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732

        self.class_ids_map = class_ids_map
        self.num_views = num_views

        images_all = [] #all images
        viewpoints_all = []
        voxels_all = [] #all voxels

        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop: 
            pic = list(np.load(
                os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items())[0][1]
            
            for i in range(pic.shape[0]):
                viewpoint_ids = np.random.choice(24, self.num_views, replace=False)
                viewpoints = []
                images = []
                
                for v in range(self.num_views):
                    viewpoint_id = viewpoint_ids[v]
                    viewpoints.append(self.get_points_from_angles(self.distance, self.elevation, -viewpoint_id * 15))
                    images.append(pic[i,viewpoint_id,:,:,:])
                    
                viewpoints_all.append(viewpoints)
                images_all.append(images)

            voxels_all.append(list(np.load(
                os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))).items())[0][1])
        
        images_all = np.concatenate(images_all, axis=0).reshape((-1, self.num_views, 4, 64, 64))
        viewpoints_all = np.concatenate(viewpoints_all, axis=0).reshape((-1, self.num_views, 3))
        images_all = np.ascontiguousarray(images_all)
        viewpoints_all = np.ascontiguousarray(viewpoints_all)
        
        self.images_all = images_all
        self.viewpoints_all = viewpoints_all
        self.voxels_all = np.ascontiguousarray(np.concatenate(voxels_all, axis=0))

        assert(self.images_all.shape[0]==self.viewpoints_all.shape[0]==self.voxels_all.shape[0])
        
        del images_all
        del viewpoints_all
        del voxels_all
    
    def __len__(self):
        return self.images_all.shape[0]
    
    def __getitem__(self,index): #index and class_id
        image = torch.from_numpy(self.images_all[index].astype('float32') / 255.) #(num_views,4,64,64)
        viewpoint = torch.from_numpy(self.viewpoints_all[index].astype('float32')) #(num_views,3)
        voxel = torch.from_numpy(self.voxels_all[index].astype('float32')) #(32,32,32)
        
        return {"img":image, "view":viewpoint, "vox":voxel}

    def get_points_from_angles(self, distance, elevation, azimuth, degrees=True):
        if isinstance(distance, float) or isinstance(distance, int):
            if degrees:
                elevation = math.radians(elevation)
                azimuth = math.radians(azimuth)
            return (
                distance * math.cos(elevation) * math.sin(azimuth),
                distance * math.sin(elevation),
                -distance * math.cos(elevation) * math.cos(azimuth))
        else:
            if degrees:
                elevation = math.pi / 180. * elevation
                azimuth = math.pi / 180. * azimuth
            return torch.stack([
                distance * torch.cos(elevation) * torch.sin(azimuth),
                distance * torch.sin(elevation),
                -distance * torch.cos(elevation) * torch.cos(azimuth)
                ]).transpose(1, 0)