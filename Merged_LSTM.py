import torch
import torch.nn as nn

class LSTM_layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, output_weight):

        super(LSTM_layer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.output_weight = output_weight
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.activ = nn.ReLU()
        self.scalar_weight = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.activ(out)

        if self.output_weight:
            out = self.min_max_normalize(out)
            out = out * self.scalar_weight
    
        return out

    def min_max_normalize(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        
        return normalized_tensor


   
class LSTM_main(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, continue_factor):

        super(LSTM_main, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.continue_factor = continue_factor
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout , batch_first=True)
        self.activ = nn.ReLU()
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        continue_out = self.activ(out)
        out = continue_out[:, -1, :]
        out = self.fc(out)

        if self.continue_factor:
            return out, continue_out
        
        return out
    


class MergedLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers_loc, num_layers_main, output_dim, dropout, seq_list, output_weight, continue_factor):

        super(MergedLSTM, self).__init__()
        
        input_dims = [i.shape[2] for i in seq_list]
        self.lstm_layers = nn.ModuleList(
            [LSTM_layer(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers_loc,
                        output_dim=hidden_dim,
                        dropout=dropout,
                        output_weight=output_weight) for input_dim in input_dims])
        
        # 메인 LSTM 정의
        self.lstm_main = LSTM_main(input_dim=hidden_dim,
                                   hidden_dim=hidden_dim,
                                   num_layers=num_layers_main,
                                   output_dim=output_dim,
                                   dropout=dropout,
                                   continue_factor=continue_factor)

    def forward(self, *inputs):
        out = None
        for i, (lstm_layer, input_data) in enumerate(zip(self.lstm_layers, inputs)):
            if out is None:
                out = lstm_layer(input_data)
            else:
                out = out + lstm_layer(input_data)

        out = self.lstm_main(out)
        
        return out

    def show_weights(self):
        weights_origin_list = []
        for i, lstm_layer in enumerate(self.lstm_layers):
            print(f"LSTM_layer {i + 1} weighted_out weights:{lstm_layer.scalar_weight.item()}")
            weights_origin_list.append(lstm_layer.scalar_weight.item())
        return weights_origin_list