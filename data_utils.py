import pickle
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader

class CollateFn(object):
    def __init__(self, frame_size):
        self.frame_size = frame_size

    def make_frames(self, tensor):
        base, pos, neg = tensor[:,0,...], tensor[:,1,...], tensor[:,2,...]
        out_base = base.view(base.size(0), base.size(1) // self.frame_size, self.frame_size * base.size(2))
        out_base = out_base.transpose(1, 2)
        out_pos = pos.view(pos.size(0), pos.size(1) // self.frame_size, self.frame_size * pos.size(2))
        out_pos = out_pos.transpose(1, 2)
        out_neg = neg.view(neg.size(0), neg.size(1) // self.frame_size, self.frame_size * neg.size(2))
        out_neg = out_neg.transpose(1, 2)
        out = torch.stack([out_base, out_pos, out_neg], dim=1)
        return out 

    def __call__(self, l):
        data_tensor = torch.from_numpy(np.array(l))
        segment = self.make_frames(data_tensor)
        return segment

def get_data_loader(dataset, batch_size, frame_size, shuffle=True, num_workers=4, drop_last=False):
    _collate_fn = CollateFn(frame_size=frame_size) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, collate_fn=_collate_fn, pin_memory=True)
    return dataloader

class PickleDataset(Dataset):
    def __init__(self, pickle_path, sample_index_path, segment_size):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(sample_index_path, 'r') as f:
            self.indexes = json.load(f)
        self.segment_size = segment_size

    def __getitem__(self, ind):
        utt_id, t, pos_path, pos_t, neg_path, neg_t = self.indexes[ind]
        segment = self.data[utt_id][t:t + self.segment_size]
        pos_segment = self.data[pos_path][pos_t:pos_t + self.segment_size]
        neg_segment = self.data[neg_path][neg_t:neg_t + self.segment_size]
        return (segment, pos_segment, neg_segment)

    def __len__(self):
        return len(self.indexes)