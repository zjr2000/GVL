import os
import h5py
import numpy as np

in_path = 'makeup_i3d_rgb_stride_1s.hdf5'
out_path = 'i3d_rgb'

if not os.path.exists(out_path):
    os.mkdir(out_path)

d = h5py.File(in_path)
print(d.keys())
for key in d.keys():
    v_d = d[key]['i3d_rgb_features'][:].astype('float32')
    np.save(os.path.join(out_path, 'v_'+key+'_rgb.npy'), v_d)