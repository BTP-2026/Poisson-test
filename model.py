import os

get_wd = os.getcwd()
os.chdir(get_wd)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import wandb
import numpy as np
import pandas as pd
from utils import get_train_data
from py_pde import PdeModel
import argparse

np.random.seed(1234)
tf.random.set_seed(1234)

parser = argparse.ArgumentParser(description='Train PINTO model for Poisson equation.')
parser.add_argument('--data_dir', type=str, default='./data/poisson_5000.h5', help='Path to the data file.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
parser.add_argument('--seq-len', type=int, default=60, help='Sequence length for sensor points.')
parser.add_argument('--saved-model', type=bool, default=False, help='Load a pre-trained model if True.')
parser.add_argument('--model-path', type=str, default='./output/Poisson_model.keras', help='Path to the pre-trained model.')
parser.add_argument('--sensor-dir', type=str, default=None, help='Path to sensor indices file if any.')
parser.add_argument('--num-indcies', type=int, default=100, help='Number of indices to use')
parser.add_argument('--learning-rate', type=float, default=1e-5, help='Initial learning rate for the optimizer.')
parser.add_argument('--units', type=int, default=64, help='Number of units in each layer.')

args = parser.parse_args()

data_dir = args.data_dir
epochs = args.epochs
batches = args.batch_size
seq_len = args.seq_len
saved_model = args.saved_model
model_path = args.model_path
sensor_dir = args.sensor_dir
num_indices = args.num_indcies
initial_learning_rate = args.learning_rate
units = args.units

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, training will run on CPU")

train_indices = np.arange(num_indices)
test_indices = [85, 90, 99, 100]
val_indices = np.arange(80, 100)
domain_samples = 1022 


ivals, ovals, idx_si, umin, umax = get_train_data(data_dir, domain_samples=domain_samples,
                                        seq_len=seq_len,
                                        indices=train_indices,
                                        val_indices=val_indices,
                                        sensor_data_dir=sensor_dir)

print(umin, umax)
parameters = {'test_ind': test_indices}

# Building the PINTO model using functional API
initializer = tf.keras.initializers.GlorotUniform(seed=1234)


def get_model(model_name, layer_names, layer_units, activation='swish'):
    sq = keras.Sequential(name=model_name)
    for i in range(len(layer_units)):
        sq.add(layers.Dense(units=layer_units[i], kernel_initializer=initializer,
                            name=layer_names + str(i + 1)))
        sq.add(layers.Activation(activation=activation))
    return sq


# Lifting operator for query values
input1 = layers.Input(shape=(1,), name='x_input')
input4 = layers.Input(shape=(1,), name='f_input')
rescale_input1 = layers.Rescaling(scale=2, offset=-1)(input1)
# rescale_input4 = layers.Rescaling(scale=0.2, offset=-2)(input4)


sp_trans = get_model(model_name='spatial_transformation',
                     layer_names='spatial_layer',
                     layer_units=[units], activation='tanh')

sp = layers.Concatenate()([input4, rescale_input1])
sp = layers.Reshape(target_shape=(1, -1))(sp)
spq = sp_trans(sp)
residual = spq

# Lifting operator for boundary condition values
input2 = layers.Input(shape=(None, 1,), name='Xbc_layer')
rescale_input3 = layers.Rescaling(scale=2, offset=-1)(input2)
pe = layers.Concatenate()([rescale_input3])
pe = get_model(model_name='BPE',
               layer_names='bpe_layer',
               layer_units=[units], activation='tanh')(pe)

# Lifting operator for boundary condition values
input3 = layers.Input(shape=(None, 1,), name='ubc_layer')
ce = layers.Dense(units=units, kernel_initializer=initializer, activation='tanh',
                  name='bve_layer_1')(input3)


# Cross Attention units
spk = layers.MultiHeadAttention(num_heads=2, key_dim=units)(query=spq, key=pe, value=ce)
spk = layers.Add()([residual, spk])
residual = spk
spk = layers.Dense(units=units, activation='tanh', kernel_initializer=initializer)(spk)
spk = layers.Dense(units=units, activation='tanh', kernel_initializer=initializer)(spk)
spk = layers.Add()([spk, residual])
residual = spk
spk = layers.MultiHeadAttention(num_heads=2, key_dim=units)(query=spk, key=pe, value=ce)
spk = layers.Add()([residual, spk])
spk = layers.Dense(units=units, activation='tanh', kernel_initializer=initializer)(spk)
ct = layers.Flatten()(spk)
residual = ct

# Projection operator for u function space
ou = get_model(model_name='U', layer_units=[units],
               layer_names='ou', activation='tanh')(ct)
ou = layers.Add()([residual, ou])
ou = layers.Dense(units=1, kernel_initializer=initializer, name='output_u')(ou)


if saved_model:
# building the PINTO model
    model = tf.keras.models.load_model(model_path, safe_mode=False)
else:
    model = keras.Model([input1, input4, input2, input3], [ou])
# metrics to track the performance of the model during training
metrics = {"loss": keras.metrics.Mean(name='loss'),
           "bound_loss": keras.metrics.Mean(name='bound_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "val_loss": keras.metrics.Mean(name='val_loss'),
           "val_data_loss": keras.metrics.Mean(name='val_data_loss'),
           "val_res_loss": keras.metrics.Mean(name='val_res_loss'),
           }


## Defining different Learning rate schedulers for different experiments
## Exponential Decay

decay_steps = 10000
decay_rate = 0.9
staircase = True

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=staircase)

# initiating the optimizer and loss function
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = keras.losses.MeanAbsoluteError()

model.summary()
model_dict = {"nn_model": model}

# initiating the PdeModel class
cm = PdeModel(inputs=ivals, outputs=ovals, get_models=model_dict, loss_fn=loss_fn,
              optimizer=optimizer, metrics=metrics,
              parameters=parameters, batches=batches, umin=umin, umax=umax)

vf = 100000  # validtation frequency
pf = 1000  # plot frequency
wb = False  # wandb logging

configuration = {
    '#_total_initial_and_boundary_points': len(ivals['xb'][0]),
    '#_total_domain_points': len(ivals['xin'][0]),
    "optimizer": "Adam",
    'initial_learning_rate': initial_learning_rate,
    # 'lr_Schedule': 'Exponential Decay',
    # 'decay_steps': decay_steps,
    # 'decay_rate': decay_rate,
    # 'staircase': staircase,
    "batches": batches,
    "Epochs": epochs,
    "Activation": 'swish',
    "model_name": 'Advection_model',
    "trainable_parameters": np.sum([np.prod(lay.shape) for lay in model.trainable_weights]),
    "non_trainable_parameters": np.sum([np.prod(lay.shape) for lay in model.non_trainable_weights]),
    'test_indices': test_indices,
    "sequence_length": seq_len}

print(configuration)

if wb:
    wandb.init(project='Finalising_results', config=configuration)

log_dir = 'output/Poisson_PINTO/'
try:
    os.makedirs(log_dir)
except FileExistsError:
    pass

history = cm.run(epochs=epochs, log_dir=log_dir, wb=wb, validation_freq=vf)

sdata = pd.DataFrame({'sensor_indices': idx_si.flatten()})
sdata.to_csv(path_or_buf=log_dir + 'sensor.csv')

if wb:
    wandb.finish()

# Evaluation
cm.nn_model.save(log_dir + 'Poisson_model.keras')