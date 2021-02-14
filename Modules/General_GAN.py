import torch
from torch.utils.data import Dataset, DataLoader

class GAN():
  def __init__(self, gen, discr, device):
    self.G = gen().to(device)
    self.F = gen().to(device)
    self.D_A = discr().to(device)
    self.D_B = discr().to(device)
    self.level = 0
  
  def level_up(self, increase):
    self.level += increase
  
  def weight_init(self, mean, std):
    self.G.weight_init(mean=mean, std=std)
    self.F.weight_init(mean=mean, std=std)
    self.D_A.weight_init(mean=mean, std=std)
    self.D_B.weight_init(mean=mean, std=std)

  def save_weights(self, path, name, iter):
    whole_path = path + '/' + name + '_' + str(iter)
    for net, label in [(self.G, 'G'), (self.F, 'F'), (self.D_A, 'D_A'), (self.D_B, 'D_B')]:
      torch.save(net.state_dict(), whole_path+ '_'+ label + '.pt')

  def load_weights(self, path, name, level):
    self.level = level
    whole_path = path + '/' + name + '_' + str(level)
    for net, label in [(self.G, 'G'), (self.F, 'G'), (self.D_A, 'D_A'), (self.D_B, 'D_B')]:
      net.load_state_dict(torch.load(whole_path + '_' + label + '.pt'))


class GAN_Opt():
  def __init__(self, gan, optimizer, lr):
    self.G = optimizer(gan.G.parameters(), lr=lr)
    self.F = optimizer(gan.F.parameters(), lr=lr)
    self.D_A = optimizer(gan.D_A.parameters(), lr=lr)
    self.D_B = optimizer(gan.D_B.parameters(), lr=lr)


class GAN_Loader():
  def __init__(self, source, size, batch_size, dataset_class):
    # size = (256, 256)
    # base = 'Dataset2'
    # batch_size = 3

    trainA_dataset = dataset_class(source + '/trainA', size)
    trainB_dataset = dataset_class(source + '/trainB', size)
    testA_dataset = dataset_class(source + '/testA', size)
    testB_dataset = dataset_class(source + '/testB', size)

    self.trainA = DataLoader(trainA_dataset, batch_size=batch_size, shuffle=True)
    self.trainB = DataLoader(trainB_dataset, batch_size=batch_size, shuffle=True)
    self.testA = DataLoader(testA_dataset, batch_size=batch_size, shuffle=True)
    self.testB = DataLoader(testB_dataset, batch_size=batch_size, shuffle=True)

    if len(self.trainA) > len(self.trainB):    
      self.trainA = DataLoader(trainB_dataset, batch_size=batch_size, shuffle=True)
      self.trainB = DataLoader(trainA_dataset, batch_size=batch_size, shuffle=True)
      self.testA = DataLoader(testB_dataset, batch_size=batch_size, shuffle=True)
      self.testB = DataLoader(testA_dataset, batch_size=batch_size, shuffle=True)
    
    print(f'Первый класс: {len(self.trainA)}')
    print(f'Второй класс: {len(self.trainB)}')
    assert len(self.trainA) <= len(self.trainB), 'Далее предполагается, что класс trainA не больше класса trainB'
