
import numpy as np
import torch


class LocalLogger():

    def __init__(self):
        # TODO: I do not like to have hardwired the elements to keep track of; maybe improving it
        self.epoch_log = dict( total_loss=0, ncorrect=0, ntokens=0, niterations=0, accuracy=0)
        self.log = dict(accuracy=list(), total_loss=list())

    def update_epoch_log(self, output, y, loss, VERBOSE=True):
        # Aggregate the current batch statistics to the previous ones.
        # TODO: Maybe this computations should be done elsewhere, them should still be stored here
        self.epoch_log['total_loss'] += loss.item()
        self.epoch_log['ncorrect'] += (torch.max(output, 1)[1] == y).sum().item()
        self.epoch_log['ntokens'] += y.numel()
        self.epoch_log['niterations'] += 1

        if VERBOSE:
            if self.epoch_log['niterations'] == 200 or self.epoch_log['niterations'] == 500 or self.epoch_log['niterations'] % 1000 == 0:
                wpb = self.epoch_log['ntokens'] // self.epoch_log['niterations']
                accuracy = 100 * self.epoch_log['ncorrect'] / self.epoch_log['ntokens']
                loss = self.epoch_log['total_loss'] / self.epoch_log['ntokens']

                print(f"Train: wpb={wpb}, num_updates={self.epoch_log['niterations']}, accuracy={accuracy:.1f}, loss={loss:.2f}")

    def finish_epoch(self, VERBOSE=True):
        self.epoch_log['total_loss'] = self.epoch_log['total_loss'] / self.epoch_log['ntokens']
        self.epoch_log['accuracy'] = 100 * self.epoch_log['ncorrect'] / self.epoch_log['ntokens']

        self.log['total_loss'].append(self.epoch_log['total_loss'])
        self.log['accuracy'].append(self.epoch_log['accuracy'])

        if VERBOSE:
            wpb = self.epoch_log['ntokens'] // self.epoch_log['niterations']
            
            print(f"Train: wpb={wpb}, num_updates={self.epoch_log['niterations']}, accuracy={self.epoch_log['accuracy']:.1f}, loss={self.epoch_log['total_loss']:.2f}")

    def __getitem__(self, key):
        try:
            return self.log[key]
        except:
            return self.epoch_log[key]
    
    def get_epoch_log(self):
        return self.epoch_log

    def get_last_epoch_log(self, prefix=None):
        prefix = prefix or ''
        return {f'{prefix}{k}' : v[-1] for k, v in self.log.items()}

    def get_one_epoch_log(self, epoch, prefix=None):
        prefix = prefix or ''
        return {f'{prefix}{k}' : v[epoch] for k, v in self.log.items()}

    def new_epoch(self):
        for key in self.epoch_log.keys():
            self.epoch_log[key] = 0

    def best_epochs(self, key='accuracy', num_elems=1, offset=0, maximize=True):
        vector = self.log[key].copy()
        epochs = np.argsort(vector) + offset
        if maximize : epochs = epochs[::-1]
        
        return epochs[:num_elems].tolist()

