from Module.lossfunction import bpr_loss


class PasserLossWrapper:
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        pass
    
    def __call__(self, out_data):
        loss = out_data  # do nothing
        return loss


class DoNothingLoss:
    
    def __init__(self):
        pass
    
    def backward(self):
        pass
    
    def item(self):
        return 0


class DoNothingLossWrapper:
    
    def __init__(self):
        self.loss = DoNothingLoss()
    
    def build(self, data, config):
        pass
    
    def __call__(self, out_data):
        return self.loss


class BPRLossWrapper:
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        self.loss_fn = bpr_loss
        
    def __call__(self, out_data):
        pos_score, neg_score = out_data
        loss = self.loss_fn(pos_score, neg_score)
        return loss
