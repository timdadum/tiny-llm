import torch.nn as nn
import torch
import utils.preprocess as pre
import torch.nn.functional as F
import utils.evaluate as ev

class BaselineModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_size, num_layers):
        super(BaselineModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.h = hidden_size
        self.n = num_layers

        self.lstm = nn.LSTM(embedding_dim, self.h, self.n, batch_first=True)

        self.linear = nn.Linear(self.h, vocab_size)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        embed = self.embedding(torch.Tensor.long(x))
        
        # Initialize hidden and cell state with zeros: no information pertained at start of pass
        h0 = torch.zeros(self.n, x.size(0), self.h).to(x.device)
        c0 = torch.zeros(self.n, x.size(0), self.h).to(x.device)

        out, _ = self.lstm(embed, (h0, c0))
        logits = self.linear(out)[:, -1, :]

        return logits
    
    def sample(self, prompt: str, tokenizer, sequence_length, generation_length, temperature: int=1.0):
        encoded_input = tokenizer.encode(prompt)
        input_ids = encoded_input.ids
        
        # Start adding subwords to prompt
        if len(input_ids) > sequence_length:
            input_ids = input_ids[-sequence_length:]
        elif len(input_ids) < sequence_length:
            padding = torch.zeros((1, sequence_length - len(input_ids)), dtype=torch.long)
            input_ids = torch.cat((padding, input_ids), dim=1)

        # Initialie the tensor for generated IDs
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        generated_ids = torch.tensor(input_ids, dtype=torch.float32).to(device)

        for _ in range(generation_length):
            # Get output logits
            logits = self(generated_ids.unsqueeze(0)).squeeze(0)

            # Apply temperature scaling, sample from prediction
            softmaxed_logits = F.softmax(logits / temperature, dim=-1)
            sampled_token_id = torch.multinomial(softmaxed_logits, num_samples=1)
            generated_ids = torch.cat((generated_ids, sampled_token_id))

        generated_text = tokenizer.decode(generated_ids.long().squeeze().tolist(), skip_special_tokens=True)
        clean_text = ev.clean_bpe_output(generated_text)

        return clean_text