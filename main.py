from types import SimpleNamespace
import pathlib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from data_loading import Vocabulary, batch_generator, load_preprocessed_dataset
from local_logger import LocalLogger
from model import CBOW


DATASET_VERSION = 'ca-100'
#COMPETITION_ROOT = '../input/vectors3'
COMPETITION_ROOT = 'data/word_vectors' # I downloaded it from https://www.kaggle.com/c/vectors3/data
#DATASET_ROOT = f'../input/text-preprocessing/data/{DATASET_VERSION}'
DATASET_ROOT = f'data/text-preprocessing/data/{DATASET_VERSION}'
#WORKING_ROOT = f'data/{DATASET_VERSION}'
MODELS_ROOT = f'model_parameters/{DATASET_VERSION}'
OUTPUTS_ROOT = f'outputs/{DATASET_VERSION}'
DATASET_PREFIX = 'ca.wiki'


params = SimpleNamespace(
    embedding_dim = 100,
    window_size = 7,
    batch_size = 1000,
    epochs = 4,
    preprocessed = f'{DATASET_ROOT}/{DATASET_PREFIX}',
    #working = f'{WORKING_ROOT}/{DATASET_PREFIX}',
    working = f'{OUTPUTS_ROOT}/{DATASET_PREFIX}', # Not used anywhere
    #modelname = f'{WORKING_ROOT}/{DATASET_VERSION}.pt',
    modelname = f'{MODELS_ROOT}/{DATASET_VERSION}.pt',
    train = True
)


def train(model, criterion, optimizer, idata, target, batch_size, device, local_logger, log=False):
    model.train()
    local_logger.new_epoch()

    for X, y in batch_generator(idata, target, batch_size, shuffle=True):
        # Get input and target sequences from batch
        X = torch.tensor(X, dtype=torch.long, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        model.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Training statistics
        local_logger.update_epoch_log(output, y, loss, VERBOSE=True)

    local_logger.finish_epoch(VERBOSE=log)
    return local_logger['accuracy'], local_logger['total_loss']

def validate(model, criterion, idata, target, batch_size, device, local_logger=None):
    model.eval()
    if local_logger is not None : local_logger.new_epoch()

    y_pred = []
    with torch.no_grad():
        for X, y in batch_generator(idata, target, batch_size, shuffle=False):
            # Get input and target sequences from batch
            X = torch.tensor(X, dtype=torch.long, device=device)
            output = model(X)

            if target is not None:
                y = torch.tensor(y, dtype=torch.long, device=device)
                loss = criterion(output, y)
                
                local_logger.update_epoch_log(output, y, loss, VERBOSE=False)
            else:
                pred = torch.max(output, 1)[1].detach().to('cpu').numpy()
                y_pred.append(pred)

    if target is not None:
        local_logger.finish_epoch(VERBOSE=False)
        return local_logger['accuracy'], local_logger['total_loss']

    else:
        return np.concatenate(y_pred)

if __name__ == "__main__":

    # TODO: all main into main function with parameters and here call docopts from docopts_parser.py and call the function as needed

    # Create working dir
    #pathlib.Path(WORKING_ROOT).mkdir(parents=True, exist_ok=True)
    pathlib.Path(MODELS_ROOT).mkdir(parents=True, exist_ok=True)
    pathlib.Path(OUTPUTS_ROOT).mkdir(parents=True, exist_ok=True)

    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("WARNING: Training without GPU can be very slow!")

    vocab, data = load_preprocessed_dataset(params.preprocessed)

    # 'El Periodico' validation dataset
    valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_valid.csv')
    tokens = valid_x_df.columns[1:]
    valid_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')
    valid_y_df = pd.read_csv(f'{COMPETITION_ROOT}/y_valid.csv')
    valid_y = valid_y_df['token'].apply(vocab.get_index).to_numpy(dtype='int32')

    # 'El Periodico' test dataset
    valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_test.csv')
    test_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')

    model = CBOW(len(vocab), params.embedding_dim).to(device)
    print(model)
    for name, param in model.named_parameters():
        print(f'{name:20} {param.numel()} {list(param.shape)}')
    print(f'TOTAL{" " * 16}{sum(p.numel() for p in model.parameters())}')

    criterion = nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters())

    # TODO: make and use a wandb logger

    train_accuracy = LocalLogger()
    wiki_accuracy = LocalLogger()
    valid_accuracy = LocalLogger()
    for epoch in range(params.epochs):
        acc, loss = train(model, criterion, optimizer, data[0][0], data[0][1], params.batch_size, device, train_accuracy, log=True)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={loss:.2f}')
        
        acc, loss = validate(model, criterion, data[1][0], data[1][1], params.batch_size, device, wiki_accuracy)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (wikipedia)')
        
        acc, loss = validate(model, criterion, valid_x, valid_y, params.batch_size, device, valid_accuracy)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (El Periódico)')

    # Save model
    torch.save(model.state_dict(), params.modelname)

    # TODO: Get best epoch model from wandb in order to generate the submission

    # Submission generation
    y_pred = validate(model, None, test_x, None, params.batch_size, device)
    y_token = [vocab.idx2token[index] for index in y_pred]

    submission = pd.DataFrame({'id':valid_x_df['id'], 'token': y_token}, columns=['id', 'token'])
    print(submission.head())
    submission.to_csv(f'{OUTPUTS_ROOT}/submission.csv', index=False)

    # TODO: Save submission on wandb artifact
