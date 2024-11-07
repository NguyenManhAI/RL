import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class BlockResMLP(nn.Module):
    def __init__(self, in_features, outs_features) -> None:
        super().__init__()
        features = [in_features] + outs_features
        self.block = nn.Sequential(
            *[nn.Sequential(nn.Linear(ins, outs), nn.ReLU() if i < len(features) - 2 else nn.Identity())
              for i, (ins, outs) in enumerate(zip(features[:-1], features[1:]))]
        )
        self.skip_conection = nn.Linear(in_features, outs_features[-1])
    def forward(self,X):
        out = self.block(X)
        out += self.skip_conection(X)
        out = F.relu(out)
        return out

class LayerResMLP(nn.Module):
    def __init__(self, num_blocks, in_features, outs_features) -> None:
        super().__init__()
        ins_feature = [in_features] + [outs_features[-1]]*(num_blocks-1)
        self.layer = nn.Sequential(
            *[BlockResMLP(in_features= ins, outs_features= outs_features) for ins in ins_feature]
        )
    def forward(self, X):
        out = self.layer(X)
        return out

class ResMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        layer = []
        for i in range(1, len(config)):
            config_params_layer = config[f'layer{i}']
            layer.append(LayerResMLP(**config_params_layer))
        
        config_params_layer_out = config['layer_out']
        layer.append(nn.Linear(**config_params_layer_out))
        
        self.model = nn.Sequential(*layer)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Sử dụng Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,X):
        out = self.model(X)
        return out
if __name__ == '__main__':
    def test_ResMLP():
        X = torch.Tensor(4,9)
        model = ResMLP(config_ResMLP_for_DQN)
        out = model(X)
        print(out.shape)
        print(model)
    test_ResMLP()