import torch
import numpy as np
from PIL import Image
from operator import itemgetter
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TranDataset(Dataset):
    def __init__(self, opt, features, image_ids, labels, is_train=True):
        self.is_train = is_train
        self.data = []
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])

        temp = []
        for feature, image_id, label in zip(features, image_ids, labels):
            feature = torch.from_numpy(feature.astype(float))
            duration, is_observed = label[0], label[1]

            img = Image.open(opt.image_dir+image_id+'.png').convert('RGB')
            img = self.transform(img)

            temp.append([duration, img, is_observed, feature])
        sorted_temp = sorted(temp, key=itemgetter(0))

        if self.is_train:
            new_temp = sorted_temp
        else:
            new_temp = temp

        for duration, img, is_observed, feature in new_temp:
            if is_observed:
                mask = opt.max_time * [1.]
                # label = duration * [1.] + (opt.max_time - duration) * [0.]
                label = duration
                # feature = torch.stack(opt.max_time * [feature])
                self.data.append(
                    [feature.cuda(), img.cuda(), torch.tensor(duration).float().cuda(),
                     torch.tensor(mask).float().cuda(), torch.tensor(label).cuda(),
                     torch.tensor(is_observed).byte().cuda()])
            else:
                # NOTE plus 1 to include day 0
                mask = (duration + 1) * [1.] + (opt.max_time - (duration + 1)) * [0.]
                # label = opt.max_time * [1.]
                label = duration
                # feature = torch.stack(opt.max_time * [feature])
                self.data.append(
                    [feature.cuda(), img.cuda(), torch.tensor(duration).float().cuda(),
                     torch.tensor(mask).float().cuda(), torch.tensor(label).cuda(),
                     torch.tensor(is_observed).byte().cuda()])

    def __getitem__(self, index_a):
        if self.is_train:
            if index_a == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                index_b = np.random.randint(index_a + 1, len(self.data))
            return [[self.data[index_a][i], self.data[index_b][i]] for i in range(len(self.data[index_a]))]
        else:
            return self.data[index_a]

    def __len__(self):
        return len(self.data)
