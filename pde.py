import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import wandb

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
tf.random.set_seed(1234)


class PdeModel:
    def __init__(self, inputs, outputs, get_models, loss_fn, optimizer, metrics, parameters,
                 batches=1, val_batches=50):

        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batches = batches
        self.parameters = parameters

        # Create efficient data pipelines
        self.inner_data = self.create_data_pipeline(inputs['xin'], inputs['fin'], inputs['xbc_in'], inputs['fbc_in'], inputs['ubc_in'],batch=batches).cache()
        self.bound_data = self.create_data_pipeline(inputs['xb'], inputs['fb'], outputs['ub'],inputs['xbc_b'], inputs['fbc_b'], inputs['ubc_b'],batch=batches).cache()
        self.val_data = self.create_data_pipeline(inputs['xval'], inputs['fval'], outputs['uval'],inputs['xbc_val'], inputs['fbc_val'], inputs['ubc_val'], batch=val_batches).cache()

        self.nn_model = get_models['nn_model']

        self.loss_tracker = metrics['loss']
        self.bound_loss_tracker = metrics['bound_loss']
        self.residual_loss_tracker = metrics['residual_loss']
        self.val_loss_tracker = metrics['val_loss']
        self.val_data_loss_tracker = metrics['val_data_loss']
        self.val_residual_loss_tracker = metrics['val_res_loss']

    @staticmethod
    def create_data_pipeline(*args, batch):
        dataset = tf.data.Dataset.from_tensor_slices(args)
        dataset = dataset.shuffle(buffer_size=len(args[0]))
        dataset = dataset.batch(np.ceil(len(args[0]) / batch))
        return dataset

    @tf.function
    def Pde_residual(self, input_data, beta, training=True):
        x, f, xbc, fbc, ubc = input_data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)
                u = self.nn_model([x, f, xbc, fbc, ubc], training=training)
            ux = tape2.gradient(u, x)

        uxx = tape.gradient(ux, x)
        del tape
        del tape2
        if uxx is None:
            print("waring u''(x) is zero")
            uxx = tf.zeros_like(ux)

        ge = -uxx - f
        residual_loss = tf.square(ge)
        return residual_loss

    @staticmethod
    def get_repeated_tensors(x_sen, t_sen, val_sen, size):
        return (tf.repeat(x_sen, [size], axis=0),
                tf.repeat(t_sen, [size], axis=0),
                tf.repeat(val_sen, [size], axis=0))

    @tf.function
    def train_step(self, bound_data, inner_data, beta):

        xb, fb, ub, xbc, fbc, ubc = bound_data

        with (tf.GradientTape(persistent=True) as tape):
            ub_pred = self.nn_model([xb, fb, xbc, fbc, ubc], training=True)
            residual_loss = tf.reduce_mean(self.Pde_residual(inner_data, beta, training=True))
            bound_loss = self.loss_fn(ub, ub_pred)
            loss = residual_loss + bound_loss

        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))

        del tape

        self.loss_tracker.update_state(loss)
        self.bound_loss_tracker.update_state(bound_loss)
        self.residual_loss_tracker.update_state(residual_loss)

        return {"loss": self.loss_tracker.result(),
                "bound_loss": self.bound_loss_tracker.result(),
                "residual_loss": self.residual_loss_tracker.result()}

    @tf.function
    def test_step(self, inp_data):

        x, f, u, xbc, fbc, ubc = inp_data
        upred = self.nn_model([x, f, xbc, fbc, ubc], training=False)
        val_data_loss = self.loss_fn(u, upred)
        val_res_loss = tf.reduce_mean(self.Pde_residual([x, f, xbc, fbc, ubc], self.parameters['beta'], training=False))
        val_loss = val_data_loss + val_res_loss

        self.val_loss_tracker.update_state(val_loss)
        self.val_data_loss_tracker.update_state(val_data_loss)
        self.val_residual_loss_tracker.update_state(val_res_loss)
        return {'val_loss': self.val_loss_tracker.result(), 'val_data_loss': self.val_data_loss_tracker.result(),
                'val_res_loss': self.val_residual_loss_tracker.result()}

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.bound_loss_tracker.reset_state()
        self.val_loss_tracker.reset_state()
        self.val_data_loss_tracker.reset_state()
        self.val_residual_loss_tracker.reset_state()

    def get_model_graph(self, log_dir, wb=False):
        pass

    def run(self, epochs, log_dir, wb=False, validation_freq=1000):

        history = {"loss": [], "residual_loss": [], "bound_loss": []}
        val_history = {"val_loss": [], "val_data_loss": [], "val_res_loss": []}

        self.get_model_graph(log_dir=log_dir, wb=wb)
        beta = self.parameters['beta']


        for epoch in range(epochs):
            start_time = time.time()
            self.reset_metrics()

            for j, (bound_data, inner_data) in enumerate(zip(
                     self.bound_data, self.inner_data)):
                logs = self.train_step(bound_data, inner_data, beta)

            if wb:
                wandb.log(logs, step=epoch + 1)

            if (epoch+1) % validation_freq == 0:
                for j, val_data in enumerate(self.val_data):
                    val_logs = self.test_step(val_data)
                if wb:
                    wandb.log(val_logs, step=epoch + 1)
            tae = time.time() - start_time
            for key, value in logs.items():
                history[key].append(value.numpy())
            if (epoch+1) % validation_freq == 0:
                for key, value in val_logs.items():
                    val_history[key].append(value.numpy())
            print(f'''Epoch:{epoch + 1}/{epochs}''')
            for key, value in logs.items():
                print(f"{key}: {value:.4f} ", end="")
            if (epoch + 1) % validation_freq == 0:
                for key, value in val_logs.items():
                    print(f"{key}: {value:.4f} ", end="")
            print(f"Time: {tae / 60:.4f}min")


        odata = pd.DataFrame(history)
        val_odata = pd.DataFrame(val_history)
        odata.to_csv(path_or_buf=log_dir + 'history.csv')
        val_odata.to_csv(path_or_buf=log_dir + 'val_history.csv')

        plt.figure()
        plt.plot(range(1, len(odata) + 1), np.log(odata['loss']))
        plt.xlabel('Epochs')
        plt.ylabel('Log_Loss')
        plt.title('log loss plot')
        plt.savefig(log_dir + '_log_loss_plt.png', dpi=300)
        if wb:
            wandb.log({"loss_plot": wandb.Image(log_dir + '_log_loss_plt.png')}, step=epochs)
        return history

    def predictions(self, inputs):
        u_pred = self.nn_model.predict(inputs, batch_size=32, verbose=False)
        return u_pred
