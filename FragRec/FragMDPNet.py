# coding = utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from sklearn.metrics import r2_score


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

class FragMDP_Net(nn.Module):
    def __init__(self, encoder_net, embedding_size, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super().__init__()
        "Fragment Encoder"
        self.encoder = encoder_net

        "Matching Degree Prediction"
        in_dim = embedding_size * 2
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim))

    def forward(self, inputs):

        "Fragment Encoder"
        encoder_output = self.encoder(inputs[:, 1:, :])

        "Matching Degree Prediction"
        demand_embed = inputs[:, 0, :]
        pred_input = torch.cat((demand_embed, encoder_output), dim=1)

        hidden_1_out = F.relu(self.layer1(pred_input))
        hidden_2_out = F.relu(self.layer2(hidden_1_out))
        hidden_3_out = F.relu(self.layer3(hidden_2_out))
        hidden_4_out = F.relu(self.layer4(hidden_3_out))
        out = torch.sigmoid(self.layer5(hidden_4_out))

        return out

def evaluate(data_loader, net, criterion):
    net.eval()
    eval_loss = 0
    eval_r2 = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            " Prepare data "
            eval_x, eval_y = data
            if torch.cuda.is_available():
                eval_x = eval_x.cuda(0)
                eval_y = eval_y.cuda(0)
            " Forward "
            out = net(eval_x)
            loss = criterion(out, eval_y)

            eval_loss += loss.data.item()
            true = eval_y.cpu().numpy()
            pred = out.cpu().detach().numpy()
            eval_r2 += r2_score(true, pred)

        eval_loss = eval_loss / len(data_loader)
        eval_r2 = eval_r2 / len(data_loader)
        return eval_loss, eval_r2


def train(train_loader, val_loader, test_loader, num_epochs, criterion, optimizer):

    encoder_net = FragEncoder(
        PathEmbedding_size = 768,
        PathAttention_factor = 4
    )
    encoder_net.load_state_dict(torch.load("PreTrained_FragEncoder.pt", map_location=torch.device('cuda')))
    model = FragMDP_Net(
        encoder_net,
        embedding_size = 768,
        n_hidden_1 = 512,
        n_hidden_2 = 512,
        n_hidden_3 = 256,
        n_hidden_4 = 256,
        out_dim = 1
    )

    if torch.cuda.is_available():
        model = model.cuda()

    val_r2_max = -1000
    train_loss_list = []
    val_loss_list = []
    val_r2_list = []

    add = 0
    print("Start Training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        for i, data in enumerate(train_loader, 0):
            " Prepare data "
            train_x, train_y = data
            if torch.cuda.is_available():
                train_x = train_x.cuda(0)
                train_y = train_y.cuda(0)
            " Forward "
            out = model(train_x)
            loss = criterion(out, train_y)
            train_loss = loss.data.item()
            " Backward "
            optimizer.zero_grad()
            loss.backward()
            " Update "
            optimizer.step()

        if epoch % 1 == 0:
            val_loss, val_r2 = evaluate(val_loader, model, criterion)  # evaluate the validation set
            train_loss1, train_r2 = evaluate(train_loader, model, criterion)  # evaluate the training set
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            val_r2_list.append(val_r2)

            if val_r2 > val_r2_max:
                val_r2_max = val_r2
                model_best = copy.deepcopy(model)
                epoch_best = epoch
            print(
                f'epoch: {epoch}, Train Loss:{train_loss:.6f}, Train R2:{train_r2:.6f}, Val Loss:{val_loss:.6f}, Val R2:{val_r2:.6f}')

        if len(train_loss_list) >= 2:
            add = add + 1 if train_loss_list[-1] > train_loss_list[-2] else 0
            if add >= 5:
                print("Exit Training...")
                print(epoch_best, val_r2_max)
                test_loss, test_r2 = evaluate(test_loader, model, criterion)  # evaluate the testing set
                print(f'Results----Test Loss:{test_loss:.6f}, Test R2:{test_r2:.6f}')
                break

    return model_best

