import os
import h5py
import numpy as np

in_path = 'tall_c3d_features.hdf5'
out_path = 'c3d'
# test = '/devdata1/VideoCaption/PDVC/data/anet/features/c3d/v___c8enCfzqw.npy'
# d = np.load(test)
# print(d.shape)


if not os.path.exists(out_path):
    os.mkdir(out_path)

d = h5py.File(in_path, 'r')
for key in d.keys():
    print(key[:-4],d[key][:].shape)
    v_d = d[key][:].astype('float32')
    np.save(os.path.join(out_path, key[:-4]+'.npy'), v_d)