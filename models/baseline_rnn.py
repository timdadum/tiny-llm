import torch.nn as nn
import torch
import preprocess as pre
import torch.nn.functional as F

class BaselineModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_size, num_layers):
        super(BaselineModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.h = hidden_size
        self.n = num_layers

        self.lstm = nn.LSTM(embedding_dim, self.h, self.n, batch_first=True)

        self.linear = nn.Linear(self.h, vocab_size)

    def forward(self, x):
        embed = self.embedding(torch.Tensor.long(x))
        
        # Initialize hidden and cell state with zeros: no information pertained at start of pass
        h0 = torch.zeros(self.n, x.size(0), self.h).to(x.device)
        c0 = torch.zeros(self.n, x.size(0), self.h).to(x.device)

        out, _ = self.lstm(embed, (h0, c0))
        logits = self.linear(out)[:, -1, :]

        return logits
    
    def sample(self, prompt: str, tokenizer, encoding, decoding, sequence_length, generation_length, temperature: int=1.0):
        tokenized_prompt = tokenizer.encode(last).tokens
        print(f"Tokenized prompt: {tokenized_prompt}")
        encoded_prompt = torch.tensor([encoding])
        print(f"Encoded prompt: {encoded_prompt}")
        
        # Start adding subwords to prompt
        for i in range(generation_length):
            last = prompt[:sequence_length]

            # Forward pass
            X, _ = pre.get_features_and_labels(last, sequence_length)

            logits = self(X)
            softmaxed_logits = F.softmax(logits / temperature, dim=0)
            sampled_class = torch.multinomial(softmaxed_logits.view(-1), 1)
            sampled_token = decoding[sampled_class]
            prompt += sampled_token

        return prompt

