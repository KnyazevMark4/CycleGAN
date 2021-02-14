from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import os
import numpy as np
from skimage.transform import resize

class ImageDataset(Dataset):

  def __init__(self, root_dir, size):
    super().__init__()
    self.size = size
    self.root_dir = root_dir
    files = []
    for root, dirs, files in os.walk(root_dir):
      files.extend(files)
    self.files = sorted(files)
    self.len_ = len(self.files)
  
  def __len__(self):
    return self.len_
      
  def load_sample(self, file):
    return resize(imread(os.path.join(self.root_dir, file)), self.size, mode='constant', anti_aliasing=True)
  
  def __getitem__(self, index):
    x = self.load_sample(self.files[index])
    x = np.array(x, dtype='float32')
    return x
