import os
os.chdir('/scratch/sagarsj42')

import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CBOW_Dataset(Dataset):
    def __init__(self):
        super(CBOW_Dataset, self).__init__()
        
        self.reviews = list()
        self.grouped_reviews = list()
        self.data_pairs = list()
        self.word_count = {'<UNK>': 0}
        self.word_list = list()
        self.vocab_size = 0
        
        self.load_reviews()
        self.group_reviews()
        self.prepare_data_pairs()
        self.compile_vocab()
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, index):
        context_words = self.data_pairs[index][0]
        word = self.data_pairs[index][1]
        context_vector = self.encode_words(context_words)
        target = self.encode_word(word)
        
        return (context_vector, target)
    
    def load_reviews(self):
        start = time.time()

        for i in range(7):
            filename = 'review_words-' + str(i+1) + '.pkl'
            print('Opening', filename, end='  ')
            with open(filename, 'rb') as f:
                reviews_set = pickle.load(f)
                print('Contains', len(reviews_set), 'entries')
                self.reviews.extend(reviews_set)

        end = time.time()
        print('Load data', 'Time taken:', end - start)
        print('No. of reviews:', len(self.reviews))

    def group_reviews(self):
        start = time.time()

        for review in self.reviews:
            review_words = list()
            for sentence in review:
                review_words.extend(sentence)
            if len(review_words) > 6:
                self.grouped_reviews.append(review_words)
            
        end = time.time()
        print('Grouping reviews', 'Time taken:', end - start)
        print('No. of grouped reviews:', len(self.grouped_reviews))
    
    def prepare_data_pairs(self):
        start = time.time()
        for review in self.grouped_reviews:
            for ind, word in enumerate(review):
                win_size = min(ind, len(review)-ind-1, 3)

                if win_size >= 3:
                    left_ind = ind - win_size
                    right_ind = ind + win_size
                    context = list()
                    
                    for cont_i in range(left_ind, right_ind+1):
                        if cont_i != ind:
                            context.append(review[cont_i])
                            
                    self.data_pairs.append((context, word))
        end = time.time()
        print('Preparing data pairs', 'Time taken:', end - start)
        print('No. of data pairs:', len(self.data_pairs))
                    
    def compile_vocab(self):
        start = time.time()
        
        for review in self.reviews:
            for sentence in review:
                for word in sentence:
                    if not word in self.word_count:
                        self.word_count[word] = 1
                    else:
                        self.word_count[word] += 1

        self.word_list = list(self.word_count.keys())
        self.vocab_size = len(self.word_list)
        print('Vocab size:', self.vocab_size)
        end = time.time()
        print('Preparing vocab', 'Time taken:', end - start)
        
    def encode_word(self, word):
        return torch.tensor([self.word_list.index(word)], dtype=torch.long)
    
    def encode_words(self, words):
        indices = [self.word_list.index(word) for word in words]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_word(self, index):
        return self.word_list[index.item()]
    
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.activation_function = nn.LogSoftmax(dim=-1)
        
    def forward(self, inputs):
        out = self.embeddings(inputs)
        ct_size = out.shape[1]
        out = (out.sum(1) / ct_size).view(out.shape[0], -1)
        out = self.linear(out)
        out = self.activation_function(out)
        
        return out
    
    def get_embedding(self, word, vocab):
        word_ind = torch.tensor(vocab.index(word), dtype=torch.long)
        return self.embeddings(word_ind).view(1, -1)
    
dataset = CBOW_Dataset()
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

model_args = {'vocab_size': dataset.vocab_size, 'embedding_dim': 750}
model = CBOW(**model_args).cuda()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)
mbatch_losses = list()
epoch_losses = list()

n_epochs = 10
for epoch in range(n_epochs):
    print('Epoch', epoch)
    total_loss = 0.0
    
    for i, (context_vector, target) in enumerate(dataloader):
        if i % 100 == 0:
            print('\tStep', i)
            if i % 100000 == 0 and i > 0:
                torch.save({
                    'vocab': dataset.word_list,
                    'model_args': model_args,
                    'state_dict': model.state_dict(),
                    'n_epochs': n_epochs,
                    'mini_batch_losses': mbatch_losses,
                    'epoch_losses': epoch_losses
                },
                    'checkpoint-'+str(epoch)+'-'+str(i)+'.pt')
        
        context_vector = context_vector.cuda()
        target = target.view(-1).cuda()
        model.zero_grad()
        
        log_probs = model(context_vector)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        mbatch_losses.append(loss.item())
    
    epoch_losses.append(total_loss)
    
    torch.save({
        'vocab': dataset.word_list,
        'model_args': model_args,
        'state_dict': model.state_dict(),
        'n_epochs': n_epochs,
        'mini_batch_losses': mbatch_losses,
        'epoch_losses': epoch_losses
    },
        'checkpoint-'+str(epoch)+'.pt')
    
    print('Total epoch loss', total_loss)
    
torch.save({
    'vocab': dataset.word_list,
    'model_args': model_args,
    'state_dict': model.state_dict(),
    'n_epochs': n_epochs,
    'mini_batch_losses': mbatch_losses,
    'epoch_losses': epoch_losses
    },
    'final-checkpoint.pt'
)

checkpoint = torch.load('final-checkpoint.pt')

vocab = checkpoint['vocab']
print('Vocab size', len(vocab))
print('Samples', vocab[:5])

model_args = checkpoint['model_args']
model = CBOW(**model_args)
print('Model initialized', model)

model.load_state_dict(checkpoint['state_dict'])
model.eval()
print('Model loaded with trained weights', model.state_dict())

n_epochs = checkpoint['n_epochs']
print('Epochs', n_epochs)

mini_batch_losses = checkpoint['mini_batch_losses']
print('No. of mini-batch losses', len(mini_batch_losses))
print('Samples', mini_batch_losses[:5])

epoch_losses = checkpoint['epoch_losses']
print('No. of epoch losses', len(epoch_losses))
print('Samples', epoch_losses[:5])

embed = model.get_embedding('camera', vocab).detach()
print(embed.shape)
print(embed)
