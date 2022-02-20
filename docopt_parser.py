
from datetime import datetime
from docopt import docopt


WEIGHTS = [1] * 6
TRAIN_WEIGHTS = False
VECTOR = False
EMBEDDING_DIM = 100
BATCH_SIZE = 1000
EPOCHS = 4
LEARNING_RATE = 0.001
TORCH_SEED = 0
RANDOM_SEED = 0
NUMPY_SEED = 0


DOCTEXT = f"""
Usage:
  main [--weights=<w>...|--train_weights=<tw>] [--vector=<v>] [--embedding_dim=<ed>] [--batch_size=<bs>] [--epochs=<e>] [--lr=<lr>] [--torch_seed=<ts>] [--random_seed=<rs>] [--numpy_seed=<ns>]
  main -h | --help

########################################################
vector options like [--weights=<w>...] have to be inputed like: --weights=3 --weights=6 --weights=10 for a [3, 6, 10] input
########################################################

Options:
  --weights=<w>               Float vector. All the needed weights (6) have to be set in order if the model will use non-trainable hand-picked weights [default: {' '.join([str(x) for x in WEIGHTS])}].
  --train_weights=<tw>        Bool. Set to True in order to train the weights. Set to False in order to not train them [default: {TRAIN_WEIGHTS}].
  --vector=<v>                Bool. Set to True if the weights ponder the embedding dimensions. Set to False if the weights are only position-wise [default: {VECTOR}]. 
  --embedding_dim=<ed>        Int. Number of dimensions for the embedding space [default: {EMBEDDING_DIM}].
  --batch_size=<bs>           Int. Number of samples to be processed simultaneously (before applying a optimization step) [default: {BATCH_SIZE}].
  --epochs=<e>                Int. Number of epochs (how many times the model sees all the training data) [default: {EPOCHS}].
  --lr=<lr>                   Positive float. It controlls the optimization speed and it is reallly important [default: {LEARNING_RATE}]. 
  --torch_seed=<ts>           Int. Set the state of the pseudo-random number generator in order to have reproducibility [default: {TORCH_SEED}].
  --random_seed=<rs>          Int. Set the state of the pseudo-random number generator in order to have reproducibility [default: {RANDOM_SEED}].
  --numpy_seed=<ns>           Int. Set the state of the pseudo-random number generator in order to have reproducibility [default: {NUMPY_SEED}].

"""


def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv, help=True, version=None, options_first=False)
    
    train_weights = True if opts['--train_weights'] == 'True' else False
    weights = None if train_weights else [float(x) for x in opts['--weights']]
    vector = True if opts['--vector'] == 'True' else (False if opts['--vector'] == 'False' else None)
    embedding_dim = int(opts['--embedding_dim'])
    batch_size = int(opts['--batch_size'])
    epochs = int(opts['--epochs'])
    lr = float(opts['--lr'])
    torch_seed = int(opts['--torch_seed'])
    random_seed = int(opts['--random_seed']) # It seems that it is not used, nevertheless, let's keep it here.
    numpy_seed = int(opts['--numpy_seed'])

    if train_weights:
        if vector:
            experiment_name = datetime.now().strftime("%Y%m%d%H%M_trained-vector")
        else:
            experiment_name = datetime.now().strftime("%Y%m%d%H%M_trained-scalar")
    else:
        experiment_name = datetime.now().strftime("%Y%m%d%H%M_fixed")

    args = (experiment_name, weights, vector, train_weights, embedding_dim, batch_size, epochs, lr, torch_seed, random_seed, numpy_seed)

    return args
