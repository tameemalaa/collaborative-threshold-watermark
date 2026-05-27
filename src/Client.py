import torch
from copy import deepcopy

class Client:
    def __init__(self,model, device, lr, flip_vectors, c, beta):
        self.device = device
        self.lr = lr
        self.model = deepcopy(model)
        self.model = self.model.float()
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        self.flip_vectors = flip_vectors
        self.c = c
        self.beta = beta
        self.grad_norm_ema = 0.0

    def get_parameters(self):
        return {key: value.cpu() for key, value in 
                self.model.state_dict().items()}

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_flip_vectors(self, flip_vectors):
        self.flip_vectors = flip_vectors

    def get_flip_vectors(self):
        return self.flip_vectors
    
    def move_to_device(self, device):
        self.model = self.model.to(device)
        self.flip_vectors = {k: v.to(device) for k, v in self.flip_vectors.items()}
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)