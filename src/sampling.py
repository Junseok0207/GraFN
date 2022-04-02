import torch
import numpy as np
from collections import defaultdict



def set_batch_size(keepdict):

    num_sample = [len(v) for v in keepdict.values()]
    batch_size = min(num_sample)

    return batch_size


def make_label_dict(targets, mask):

    label_dict = defaultdict(list)
    for idx in range(len(mask)):
        if mask[idx].item():
            label_dict[f'{targets[idx].item()}'].append(idx)

    return label_dict


def make_labels_matrix(batch_size, num_classes, device):

    total = batch_size*num_classes
    
    label_matrix = torch.zeros(total, num_classes).to(device)

    for i in range(num_classes):
        label_matrix[i*batch_size : (i+1)*batch_size][:, i] = 1. 

    return label_matrix



class Sampler():
    def __init__(self, args, data, labels, mask):
        
        self.data = data
        self.num_classes = len(torch.unique(labels))
  
        self.device = args.device
        self.labeled_node_list = self.data.nodes[mask].tolist()

        self.label_dict = make_label_dict(labels, mask)
        self.batch_size = set_batch_size(self.label_dict)
        
        self.label_matrix = make_labels_matrix(self.batch_size, self.num_classes, self.device)
        

    def sample(self):
        
        samples = []

        for t in range(self.num_classes):
            t_indices = self.label_dict[f'{t}']
            try:
                sample = np.random.choice(t_indices, self.batch_size, replace=False)
            except:
                sample = np.random.choice(t_indices, self.batch_size, replace=True)
            samples.append(sample)
            
        idx_sample = np.concatenate(samples)

        return self.label_matrix, idx_sample, self.batch_size

    


    
    

