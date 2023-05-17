import os.path
import numpy as np
import h5py
d = h5py.File('/data1/wangteng/dataset/swin/anet_swin_fps_15_len_16_stride_16.h5')
in_path = '/data1/wangteng/dataset/swin/swin'
for k,v in d.items():
    out_path = os.path.join(in_path, k + '.npy')
    np.save(out_path, v)
    print(out_path)
pass