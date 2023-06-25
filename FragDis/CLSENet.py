# coding = utf-8

import copy
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# helper functions
def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor
def MLP(dim, projection_size, hidden_size=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

# FragEncoder class for fragment encoder
class FragEncoder(nn.Module):
    def __init__(self,
                 PathEmbedding_size,
                 PathAttention_factor):
        super().__init__()

        self.PathAttention_factor = PathAttention_factor
        self.attention_W = nn.Parameter(torch.Tensor(
            PathEmbedding_size, self.PathAttention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.PathAttention_factor))
        self.projection_h = nn.Parameter(
            torch.Tensor(self.PathAttention_factor, 1))
        for tensor in [self.attention_W, self.projection_h]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

    def forward(self, x):

        attention_input = x  # path vectors of the fragment
        attention_temp = F.relu(torch.tensordot(
            attention_input, self.attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score = F.softmax(torch.tensordot(
            attention_temp, self.projection_h.clone(), dims=([-1], [0])), dim=1)
        attention_output = torch.sum(
            self.normalized_att_score * attention_input, dim=1)

        return attention_output


# a wrapper class for the base neural network
class NetWrapper(nn.Module):
    def __init__(self,
                 encoder_net,
                 PathEmbedding_size,
                 PathAttention_factor,
                 projection_size,
                 projection_hidden_size,
                 use_simsiam_mlp = False):
        super().__init__()

        "Encoder"
        self.encoder = encoder_net
        self.PathEmbedding_size = PathEmbedding_size
        self.PathAttention_factor = PathAttention_factor

        "Projector"
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp


    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):

        encoder_output = self.encoder(x)

        assert encoder_output is not None, f'FragEncoder never emitted an output'
        return encoder_output


    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)  #[FC->BN->ReLU->FC]
        projection = projector(representation)
        return projection, representation

# main class
class CLSE_Net(nn.Module):
    def __init__(self,
                 encoder_net,
                 PathEmbedding_size,
                 PathAttention_factor,
                 projection_size = 256,
                 projection_hidden_size = 512,
                 moving_average_decay = 0.99,
                 use_momentum = True):
        super().__init__()

        self.online_encoder = NetWrapper(encoder_net, PathEmbedding_size, PathAttention_factor, projection_size, projection_hidden_size,
                                         use_simsiam_mlp=not use_momentum)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.Target_Encoder = NetWrapper(encoder_net, PathEmbedding_size, PathAttention_factor, projection_size, projection_hidden_size,
                                         use_simsiam_mlp=not use_momentum)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(encoder_net)
        self.to(device)

    @singleton('target_encoder')
    def _get_target_encoder(self, encoder):
        #target_encoder = copy.deepcopy(self.online_encoder)
        sd = self.online_encoder.state_dict()
        encoder.load_state_dict(sd)
        target_encoder = encoder
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)

        path_num = int(x.shape[1]/2)
        #xx = copy.deepcopy(x)
        frag_one = x[:, 0:path_num, :]  # frag1 paths
        frag_two = x[:, path_num:, :]  # frag2 paths

        online_proj_one, _ = self.online_encoder(frag_one)
        online_proj_two, _ = self.online_encoder(frag_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        #ttt = copy.deepcopy(self.online_encoder)
        with torch.no_grad():
            _, _ = self.Target_Encoder(frag_one)
            target_encoder = self._get_target_encoder(self.Target_Encoder) if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(frag_one)
            target_proj_two, _ = target_encoder(frag_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()

# training function
def train(train_loader, num_epochs, optimizer):

    encoder_net = FragEncoder(   # define encoder
        PathEmbedding_size = 768,
        PathAttention_factor = 4
    )
    learner = CLSE_Net(
        encoder_net,
        PathEmbedding_size = 768,
        PathAttention_factor = 4,
        use_momentum=True
    )

    if torch.cuda.is_available():
        learner = learner.cuda()

    train_loss_list = []
    loss_min = 999999
    print("Start Training...")
    Flag = False # used to record if training is stopped halfway
    for epoch in range(1, num_epochs+1):
        learner.train()
        train_loss_temp_list = []
        for i, data in enumerate(train_loader, 0):
            " Prepare data "
            train_x = data[0]
            if torch.cuda.is_available():
                train_x = train_x.cuda(0)
            " Forward "
            loss = learner(train_x)
            train_loss_temp = loss.data.item()
            train_loss_temp_list.append(train_loss_temp)
            " Backward "
            optimizer.zero_grad()
            loss.backward()
            " Update "
            optimizer.step()
            learner.update_moving_average()  # update moving average of target encoder

        if epoch % 1 == 0:
            train_loss = np.mean(train_loss_temp_list)
            print(f'Results----Train_Loss_temp:{train_loss_temp:.6f}, Train_Loss:{train_loss:.6f}')

            if train_loss < loss_min:
                loss_min = train_loss
                epoch_best = epoch
                learner_best = copy.deepcopy(learner)
                encoder_net_best = copy.deepcopy(encoder_net)
            train_loss_list.append(train_loss)

        if len(train_loss_list) >= 2:
            if train_loss_list[-1] > train_loss_list[-2]:
                add += 1
            else:
                add = 0
            if add >= 5:
                print("Exit Training...")
                print("epoch for minimum loss：", epoch_best, "minimum loss：", loss_min)
                Flag = True
                break

    if not Flag:
        print("Training Ending")
        print("epoch for minimum loss：", epoch_best, "minimum loss：", loss_min)

    # return your trained network
    return learner_best, encoder_net_best

