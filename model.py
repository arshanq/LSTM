import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.n_layers = num_layers
        self.lstm = nn.LSTM(embed_size, self.hidden_size, self.n_layers, dropout = 0.5,
                           batch_first = True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.hidden_size, vocab_size)
    
    def forward(self, features, captions):
        batch_size = features.shape[0]
        #here the size of dimensions is batch_size * caption_length and we're removing the end token from it.
        captions = captions[:,:-1]
        features = features.unsqueeze(1)
        embedding = self.embedding(captions)
#        print("Dim of captions after embedding " + str(captions.shape))
        #dim of captions = batch_size * (caption_len - 1 ) * embed_size

        self.hidden = self.init_hidden(batch_size) 
        
        embedding = torch.cat([features, embedding], dim=1)
        ltsm_output, self.hidden = self.lstm(embedding, self.hidden)
        return self.fc(ltsm_output)
        
#         print("Features dim: "+str(features.shape) + "hidden_state dim: " +str(hidden_state.shape))
#         out, hc = self.lstm(features, (hidden_state, cell_state))
        
        
#         output_tensor = torch.empty((batch_size, captions.shape[1], self.vocab_size)).cuda()
        
#         output_tensor = torch.cat([output_tensor, self.fc(out)], 1)
#         print("Dim of the output tensor is " + str(output_tensor.shape))
        
#         # Now we need to pass one word at a time to the lstm
#         for word_idx in range(captions.shape[1]):
#             caption = captions[:,word_idx,:].unsqueeze(1)
#             print("Dim of caption to lstm is " + str(caption.shape))
#             print("Dim of output of lstm is " + str(out.shape))
#             out, hc = self.lstm(caption, out)
#            # out = out.permute(1,0,2)
#             vocab_pred = self.fc(out)
#             output_tensor = torch.cat([output_tensor, vocab_pred], 1)
            
#         return output_tensor

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         start_word="<start>",
#         end_word="<end>",
        output = []
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)
        
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden)
            fc_out = self.fc(lstm_out)
            value, target_index = fc_out.max(2)
            idx = target_index.item()
            if idx == 1:
                break;
            
            output.append(idx)
            inputs = self.embedding(target_index)
            
        return output
            