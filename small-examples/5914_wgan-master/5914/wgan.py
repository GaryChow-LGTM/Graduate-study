import os
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina' # enable hi-res output
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.losses import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.initializers import *
from keras.callbacks import *
from keras.utils.generic_utils import Progbar
from load_data import load_data
from normalization_data import normalization_data, unnormalization_data
# random seed
RND = 777

# output settings
RUN = 'B'
OUT_DIR = 'out/' + RUN
TENSORBOARD_DIR = './tensorboard/wgans/' + RUN
SAVE_SAMPLE_IMAGES = False

# GPU # to run on
GPU = "0"

BATCH_SIZE = 5
ITERATIONS = 5000

# size of the random vector used to initialize G
Z_SIZE = 15
D_ITERS = 5

# create output dir
if not os.path.isdir(OUT_DIR): os.makedirs(OUT_DIR)

# make only specific GPU to be utilized
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

# seed random generator for repeatability
np.random.seed(RND)

# load dataset
data_filepath = "./data.csv"
# type(data_X1)==np.float64, type(data_X2)==np.int64
# data_X1, data_X2, data_Y = load_data(data_filepath)
dataset = normalization_data(data_filepath)
data_dim = dataset.shape[1]

def d_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def create_D():
    # weights are initlaized from normal distribution with below params
    weight_init = RandomNormal(mean=0., stddev=0.02)

    input_ = Input(shape=(data_dim, ), name='input_')
    x = Dense(64)(input_)
    x = LeakyReLU(0.1)(x)
    output_fake = Dense(1, activation = "linear", name = "output_fake")(x)

    return Model(
        inputs=input_, outputs=output_fake, name='D')

def create_G(Z_SIZE=Z_SIZE):
    input_ = Input(shape=(Z_SIZE, ), name='input_')
    x = Dense(32)(input_)
    x = LeakyReLU(0.1)(x)
    output_gen = Dense(data_dim, activation = "linear", name = "output_gen")(x)

    return Model(
        inputs=input_, outputs=output_gen, name='G')

D = create_D()

D.compile(
    optimizer=RMSprop(lr=0.00005),
    # loss=[d_loss, 'binary_crossentropy']
    loss=['binary_crossentropy']
)
# D.summary()

input_z = Input(shape=(Z_SIZE, ), name='input_z_')

G = create_G()
# G.summary()
# create combined D(G) model
output_is_fake= D(G(inputs=input_z))
DG = Model(inputs=input_z, outputs=output_is_fake, name='DG')

DG.compile(
    optimizer=RMSprop(lr=0.00005),
    # loss=[d_loss, 'binary_crossentropy']
    loss=['binary_crossentropy']
)
# DG.summary()

start = 0
for step in range(ITERATIONS):
    random_latent_vectors = np.random.normal(0., 1., (BATCH_SIZE, Z_SIZE))
    generated_datas = G.predict(random_latent_vectors)
    stop = start + BATCH_SIZE
    real_datas = dataset[start:stop]
    combined_datas = np.concatenate([generated_datas, real_datas])
    labels = np.concatenate([np.ones((BATCH_SIZE, 1)),
                            np.zeros((BATCH_SIZE, 1))]
            )
    d_loss = D.train_on_batch(combined_datas, labels)

    random_latent_vectors = np.random.normal(0., 1., (BATCH_SIZE, Z_SIZE))
    misleading_targets = np.zeros((BATCH_SIZE, 1))
    dg_loss = DG.train_on_batch(random_latent_vectors, misleading_targets)

    start+=BATCH_SIZE
    if start>len(dataset) - BATCH_SIZE:
        start = 0

G.save("./Gmodel.h5")
random_latent_vectors = np.random.normal(0., 1., (10, Z_SIZE))
fakedata = unnormalization_data(G.predict(random_latent_vectors))
#np.savetxt('./fakedata.csv', np.around(fakedata, decimals=4), delimiter = ',')
np.savetxt('./fakedata.csv', np.around(fakedata, decimals=4), delimiter = ',')
print(fakedata)

# # write tensorboard summaries
# sw = tf.summary.FileWriter(TENSORBOARD_DIR)
# def update_tb_summary(step, write_sample_images=True):

#     s = tf.Summary()

#     # losses as is
#     for names, vals in zip((('D_real_is_fake', 'D_real_class'),
#                             ('D_fake_is_fake', 'D_fake_class'), ('DG_is_fake',
#                                                                  'DG_class')),
#                            (D_true_losses, D_fake_losses, DG_losses)):

#         v = s.value.add()
#         v.simple_value = vals[-1][1]
#         v.tag = names[0]

#         v = s.value.add()
#         v.simple_value = vals[-1][2]
#         v.tag = names[1]

#     # D loss: -1*D_true_is_fake - D_fake_is_fake
#     v = s.value.add()
#     v.simple_value = -D_true_losses[-1][1] - D_fake_losses[-1][1]
#     v.tag = 'D loss (-1*D_real_is_fake - D_fake_is_fake)'

#     # generated image
#     if write_sample_images:
#         img = generate_samples(step, save=True)
#         s.MergeFromString(tf.Session().run(
#             tf.summary.image('samples_%07d' % step,
#                              img.reshape([1, *img.shape, 1]))))

#     sw.add_summary(s, step)
#     sw.flush()

# progress_bar = Progbar(target=ITERATIONS)

# DG_losses = []
# D_true_losses = []
# D_fake_losses = []

# for it in range(ITERATIONS):

#     if len(D_true_losses) > 0:
#         progress_bar.update(
#             it,
#             values=[ # avg of 5 most recent
#                     ('D_real_is_fake', np.mean(D_true_losses[-5:], axis=0)[1]),
#                     ('D_real_class', np.mean(D_true_losses[-5:], axis=0)[2]),
#                     ('D_fake_is_fake', np.mean(D_fake_losses[-5:], axis=0)[1]),
#                     ('D_fake_class', np.mean(D_fake_losses[-5:], axis=0)[2]),
#                     ('D(G)_is_fake', np.mean(DG_losses[-5:],axis=0)[1]),
#                     ('D(G)_class', np.mean(DG_losses[-5:],axis=0)[2])
#             ]
#         )
        
#     else:
#         progress_bar.update(it)

#     # 1: train D on real+generated images

#     if (it % 1000) < 25 or it % 500 == 0: # 25 times in 1000, every 500th
#         d_iters = 100
#     else:
#         d_iters = D_ITERS

#     for d_it in range(d_iters):

#         # unfreeze D
#         D.trainable = True
#         for l in D.layers: l.trainable = True

#         # clip D weights

#         for l in D.layers:
#             weights = l.get_weights()
#             weights = [np.clip(w, -0.01, 0.01) for w in weights]
#             l.set_weights(weights)

#         # 1.1: maximize D output on reals === minimize -1*(D(real))

#         # draw random samples from real images
#         index = np.random.choice(len(X_train), BATCH_SIZE, replace=False)
#         real_images = X_train[index]
#         real_images_classes = y_train[index]

#         D_loss = D.train_on_batch(real_images, [-np.ones(BATCH_SIZE), 
#           real_images_classes])
#         D_true_losses.append(D_loss)

#         # 1.2: minimize D output on fakes 

#         zz = np.random.normal(0., 1., (BATCH_SIZE, Z_SIZE))
#         generated_classes = np.random.randint(0, 10, BATCH_SIZE)
#         generated_images = G.predict([zz, generated_classes.reshape(-1, 1)])

#         D_loss = D.train_on_batch(generated_images, [np.ones(BATCH_SIZE),
#           generated_classes])
#         D_fake_losses.append(D_loss)

#     # 2: train D(G) (D is frozen)
#     # minimize D output while supplying it with fakes, 
#     # telling it that they are reals (-1)

#     # freeze D
#     D.trainable = False
#     for l in D.layers: l.trainable = False

#     zz = np.random.normal(0., 1., (BATCH_SIZE, Z_SIZE)) 
#     generated_classes = np.random.randint(0, 10, BATCH_SIZE)

#     DG_loss = DG.train_on_batch(
#         [zz, generated_classes.reshape((-1, 1))],
#         [-np.ones(BATCH_SIZE), generated_classes])

#     DG_losses.append(DG_loss)

#     if it % 10 == 0:
#         update_tb_summary(it, write_sample_images=(it % 250 == 0))