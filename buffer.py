import torch
import pickle
from operator import itemgetter


class Buffer():
    def __init__(self, episode_length, buffer_size =1000000, batch_size=128):
        self.data = None
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.buffer_is_full = False
        self.batch_size = batch_size
        self.names = ['S', 'A', 'S_prime', 'R', 'terminal']
        self.num_episodes = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def finish_episode(self):
        self.num_episodes += 1

    def append(self, s, a, s_prime, r, terminal):
        _data = [s, a, s_prime, r, terminal]
        if type(self.data) == type(None):
            self.data = {}
            for i, (name, dat) in enumerate(zip(self.names, _data)):
                self.data[name] = dat
        else:
            for i, (name, dat) in enumerate(zip(self.names, _data)):
                self.data[name] = torch.cat((self.data[name], dat), dim= 0)

            if self.buffer_is_full:
                for i, (name, dat) in enumerate(zip(self.names, _data)):
                    self.data[name] = self.data[name][len(s):]
            else:
                self.buffer_is_full = (len(self.data['S']) >= self.buffer_size)

    def sample(self):
        idx = torch.randint(len(self.data['S']), (self.batch_size, ))
        data = (self.data[name][idx].to(self.device) for name in self.names)
        return data




