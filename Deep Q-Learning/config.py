import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_ResMLP_for_DQN = {
    'layer1':{
        'num_blocks': 2,
        'in_features': 9,
        'outs_features': [32,16,32],
    },
    'layer2':{
        'num_blocks': 2,
        'in_features': 32,
        'outs_features': [64,32,64],
    },
    'layer3':{
        'num_blocks': 2,
        'in_features': 64,
        'outs_features': [128,64,128],
    },
    'layer4':{
        'num_blocks': 2,
        'in_features': 128,
        'outs_features': [256,128,256],
    },
    'layer_out':{
        'in_features': 256,
        'out_features': 9 
    }
}

config_train_DQN_for_TicTacToe = {
    'batch_size': 32,
    'M': 5000, # num episodes
    'C': 100,
    'size_board': 3,
    'gamma': 0.9
}
config_decay_epsilon = {
    'mode': 'curve', 
    'rate_curve': 4, 
    'curved_direction': 'left'
}
