from src.Client import Client
import torch
import torch.nn as nn
from copy import deepcopy
class ClientResNet18(Client):
    def __init__(self,model, device, lr, flip_vectors, c, beta):
        model = deepcopy(model)
        super(ClientResNet18, self).__init__(model, device, lr, flip_vectors, c, beta)
        self.model = self.model.float()
        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.flip_vectors = flip_vectors

    def train(self, data_loader):
        self.model.train()
        epoch_loss = 0
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        return acc

    def train_with_adding_bias(self,data_loader):
        self.model.train()
        epoch_loss = 0
        old_model = deepcopy(self.model.state_dict())
        pseudo_grad = {}
        pseudo_grad_norms = []
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        with torch.no_grad():
            bias_norm = 0.0
            for name, param in self.model.named_parameters():
                if param.grad is None or self.flip_vectors[name] is None:
                    continue
                pseudo_grad[name] =  param.data - old_model[name]
                pseudo_grad_norms.append(pseudo_grad[name].norm().item())
            current_mean_grad_norm = sum(pseudo_grad_norms) / len(pseudo_grad_norms)
            self.grad_norm_ema = self.beta * self.grad_norm_ema + (1.0 - self.beta) * current_mean_grad_norm
            val_t = self.c * (self.grad_norm_ema)
            scale_factors = {name: 0.0 for name in self.flip_vectors}         
            if val_t > 0:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in self.flip_vectors and self.flip_vectors[name] is not None  :
                            scale_factors[name] = pseudo_grad[name].norm().item() * val_t 
                            old_pamram = param.data.clone()
                            param.data.add_(self.flip_vectors[name] * scale_factors[name])
                            new_param = param.data.clone()
                            bias_norm += (new_param - old_pamram).norm().item()
        return epoch_loss / len(data_loader), val_t, self.grad_norm_ema, bias_norm

    def train_with_calcluting_bias(self, data_loader):
        self.model.train()
        epoch_loss = 0
        old_model = deepcopy(self.model.state_dict())
        pseudo_grad = {}
        pseudo_grad_norms = []
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if self.flip_vectors[name] is None:
                    continue
                pseudo_grad[name] =  param.data - old_model[name]
                pseudo_grad_norms.append(pseudo_grad[name].norm().item())
            current_mean_grad_norm = sum(pseudo_grad_norms) / len(pseudo_grad_norms)
            self.grad_norm_ema = self.beta * self.grad_norm_ema + (1.0 - self.beta) * current_mean_grad_norm
            val_t = self.c * (self.grad_norm_ema)
            scale_factors = {name: 0.0 for name in self.flip_vectors}         
            if val_t > 0:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in self.flip_vectors and self.flip_vectors[name] is not None  :
                            scale_factors[name] = pseudo_grad[name].norm().item() * val_t 
        return epoch_loss / len(data_loader) , val_t, self.grad_norm_ema, scale_factors

    def add_bias(self,scale_factors):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None or self.flip_vectors[name] is None:
                    continue
                # Ensure all tensors are on the same device as param (GPU)
                flip_vec = self.flip_vectors[name].to(param.device)
                scale = torch.tensor(scale_factors[name], device=param.device, dtype=param.data.dtype)
                param.data.add_(flip_vec * scale)
            return self.get_parameters() 
