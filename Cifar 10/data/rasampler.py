# Path: data/rasampler.py
import torch, torch.distributed as dist, math

class RASampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None: num_replicas = dist.get_world_size() if dist.is_available() else 1
        if rank is None: rank = dist.get_rank() if dist.is_available() else 0
        self.dataset, self.num_replicas, self.rank, self.shuffle = dataset, num_replicas, rank, shuffle
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist() if self.shuffle else list(range(len(self.dataset)))
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch