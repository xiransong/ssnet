import torch


class ValTestDataloader:

    def __init__(self, src, dst, neg_dst, batch_size):
        self.src_dl = torch.utils.data.DataLoader(dataset=src, batch_size=batch_size)
        self.dst_dl = torch.utils.data.DataLoader(dataset=dst, batch_size=batch_size)
        if neg_dst is None:
            self.neg_dst_dl = None
        else:
            self.neg_dst_dl = torch.utils.data.DataLoader(dataset=neg_dst, batch_size=batch_size)

    def __iter__(self):
        self.src_iter = iter(self.src_dl)
        self.dst_iter = iter(self.dst_dl)
        if self.neg_dst_dl is not None:
            self.neg_dst_iter = iter(self.neg_dst_dl)
        return self

    def __len__(self):
        return len(self.src_dl)

    def __next__(self):
        src = next(self.src_iter)
        dst = next(self.dst_iter)
        if self.neg_dst_dl is not None:
            neg_dst = next(self.neg_dst_iter)
        else:
            neg_dst = None
        
        return src, dst, neg_dst
