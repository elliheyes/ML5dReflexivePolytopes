import math
import time
import numpy as np
import tensorflow as tf
from itertools import product
from datetime import timedelta
from sklearn.model_selection import train_test_split
        

#%% Data


# define a function to load the vertex coordinate data 
def load_vert_data(num_vertices):
    
    path = '/Users/elliheyes/Documents/PhD/Data/KS/4d/v'+str(num_vertices)+'_dataset.txt'
    f = open(path, 'r').read()
    raw_data = f.split('\n')
    raw_data = [x.split(' ') for x in raw_data]
    
    data_X = []
    for i in range(int(len(raw_data)/5)):
        row1 = list(map(int,list(filter(None, raw_data[i*5+1]))))
        row2 = list(map(int,list(filter(None, raw_data[i*5+2]))))
        row3 = list(map(int,list(filter(None, raw_data[i*5+3]))))
        row4 = list(map(int,list(filter(None, raw_data[i*5+4]))))
        data_X.append(np.transpose([row1,row2,row3,row4]))
             
    return data_X


# define a function to load the adjacency matrix data
def load_adj_data(num_vertices):
    
    path = '/Users/elliheyes/Documents/PhD/Data/KS/4d/v'+str(num_vertices)+'_adj.txt'
    f = open(path, 'r').read()
    raw_data = f.split('\n')
    raw_data = [x.split(' ') for x in raw_data]
    del raw_data[-1]

    data_A = []
    for i in range(int(len(raw_data)/num_vertices)):
        a = []
        for j in range(num_vertices):
            b = []
            for k in range(num_vertices):
                b.append(int("".join(filter(str.isdigit, raw_data[i*num_vertices+j][k]))))
            a.append(b)
        data_A.append(np.array(a))
    
    return data_A


data_A = load_adj_data(6)
data_X = load_vert_data(6)


# shuffle and split data into training and test sets
A_train, A_test, X_train, X_test = train_test_split(data_A, data_X, test_size=0.2, random_state=42)


#%% Models


epochs = 3
num_samples = 10
batch_size = 64
noise_dim = 5
vertexes = 6
nodes = 4


def make_generator_model(vertexes, nodes, dropout_rate=0.):
    inputs = tf.keras.Input(shape=(5,))
    x = tf.keras.layers.Dense(units=128,activation=tf.keras.activations.tanh)(inputs)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.Dense(units=256,activation=tf.keras.activations.tanh)(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.Dense(units=512,activation=tf.keras.activations.tanh)(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    
    adj_outputs = tf.keras.layers.Dense(units=vertexes*vertexes)(x)
    adj_outputs = tf.reshape(adj_outputs, (-1, vertexes, vertexes))
    adj_outputs = (adj_outputs + tf.linalg.matrix_transpose(adj_outputs)) / 2
    adj_outputs = tf.keras.layers.Dropout(rate=dropout_rate)(adj_outputs)
    
    feat_outputs = tf.keras.layers.Dense(units=vertexes*nodes)(x)
    feat_outputs = tf.reshape(feat_outputs, (-1, vertexes, nodes))
    feat_outputs = tf.keras.layers.Dropout(rate=dropout_rate)(feat_outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=[adj_outputs,feat_outputs])
    
    return model

generator = make_generator_model(vertexes=6, nodes=4, dropout_rate=0.5)


def make_discriminator_model(vertexes, nodes, dropout_rate=0.):
    adj_tensor = tf.keras.Input(shape=(vertexes,vertexes,))
    feat_tensor = tf.keras.Input(shape=(vertexes,nodes,))
    
    # graph convolution layers
    hidden_tensor = tf.matmul(adj_tensor,feat_tensor)
    hidden_tensor = tf.keras.layers.Dense(units=128,activation=tf.keras.activations.tanh)(hidden_tensor)
    hidden_tensor = tf.keras.layers.Dropout(rate=dropout_rate)(hidden_tensor)
    
    node_feature_tensor = tf.concat((hidden_tensor, feat_tensor), -1)
    
    hidden_tensor = tf.matmul(adj_tensor,node_feature_tensor)
    hidden_tensor = tf.keras.layers.Dense(units=64,activation=tf.keras.activations.tanh)(hidden_tensor)
    hidden_tensor = tf.keras.layers.Dropout(rate=dropout_rate)(hidden_tensor)

    node_feature_tensor = tf.concat((hidden_tensor, feat_tensor), -1)
    
    # graph aggregation layer
    i = tf.keras.layers.Dense(units=128,activation=tf.keras.activations.sigmoid)(node_feature_tensor) 
    j = tf.keras.layers.Dense(units=128,activation=tf.keras.activations.tanh)(node_feature_tensor) 
    
    outputs = tf.reduce_sum(i * j, 1)
    outputs = tf.keras.activations.tanh(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout_rate)(outputs)
    
    # multi dense layers
    outputs = tf.keras.layers.Dense(units=128,activation=tf.keras.activations.tanh)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(units=64,activation=tf.keras.activations.tanh)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout_rate)(outputs)
        
    outputs = tf.keras.layers.Dense(units=1)(outputs)
    
    model = tf.keras.Model(inputs=[adj_tensor,feat_tensor],outputs=outputs)
    
    return model

discriminator = make_discriminator_model(vertexes=6, nodes=4, dropout_rate=0.5)


def make_value_model(vertexes, nodes, dropout_rate=0.):
    adj_tensor = tf.keras.Input(shape=(vertexes,vertexes,))
    feat_tensor = tf.keras.Input(shape=(vertexes,nodes,))
    
    # graph convolution layers
    hidden_tensor = tf.matmul(adj_tensor,feat_tensor)
    hidden_tensor = tf.keras.layers.Dense(units=128,activation=tf.keras.activations.tanh)(hidden_tensor)
    hidden_tensor = tf.keras.layers.Dropout(rate=dropout_rate)(hidden_tensor)
    
    node_feature_tensor = tf.concat((hidden_tensor, feat_tensor), -1)
    
    hidden_tensor = tf.matmul(adj_tensor,node_feature_tensor)
    hidden_tensor = tf.keras.layers.Dense(units=64,activation=tf.keras.activations.tanh)(hidden_tensor)
    hidden_tensor = tf.keras.layers.Dropout(rate=dropout_rate)(hidden_tensor)

    node_feature_tensor = tf.concat((hidden_tensor, feat_tensor), -1)
    
    # graph aggregation layer
    i = tf.keras.layers.Dense(units=128,activation=tf.keras.activations.sigmoid)(node_feature_tensor) 
    j = tf.keras.layers.Dense(units=128,activation=tf.keras.activations.tanh)(node_feature_tensor) 
    
    outputs = tf.reduce_sum(i * j, 1)
    outputs = tf.keras.activations.tanh(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout_rate)(outputs)
    
    # multi dense layers
    outputs = tf.keras.layers.Dense(units=128,activation=tf.keras.activations.tanh)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(units=64,activation=tf.keras.activations.tanh)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout_rate)(outputs)
        
    outputs = tf.keras.layers.Dense(units=1)(outputs)
    
    model = tf.keras.Model(inputs=[adj_tensor,feat_tensor],outputs=outputs)
    
    return model

valuer = make_value_model(vertexes=6, nodes=4)


def scorer(dataset, alpha=0.5, beta=0.5, gamma=0.5):
    adj_list = dataset[0]
    feat_list = dataset[1]
    
    scores = []
    for i in range(len(feat_list)):
        vertices = []
        for j in range(len(feat_list[i])):
            vertex = []
            for k in range(len(feat_list[i][j])):
                vertex.append(feat_list[i][j][k])
            vertices.append(vertex)
        poly = Polyhedron(vertices)
        adj = poly.adjacency_matrix()
        
        dual = poly.polar()
        dual_adj = dual.adjacency_matrix()
        dual_vertices = []
        for i in range(len(dual.vertices())):
            dual_vertices.append([float(dual.vertices()[i][j]) for j in range(len(dual.vertices()[i]))])
    
        vertices_dist = []
        for j in range(len(vertices)):
            cum_sum = 0
            for k in range(len(vertices[j])):
                cum_sum += (abs(vertices[j][k]) % 1)**2
            vertices_dist.append(math.sqrt(cum_sum))
        
        dual_vertices_dist = []
        for j in range(len(dual_vertices)):
            cum_sum = 0
            for k in range(len(dual_vertices[j])):
                cum_sum += (abs(dual_vertices[j][k]) % 1)**2
            dual_vertices_dist.append(math.sqrt(cum_sum))
        
        lattice_points = []
        for a, b, c, d in product(range(2),repeat=4):
            lattice_points.append([a,b,c,d])
        
        poly_interior, dual_interior = 0, 0
        for j in range(len(lattice_points)):
            if poly.contains(lattice_points[j]):
                poly_interior += 1
            if dual.contains(lattice_points[j]):
                dual_interior += 1
                
        equal_edge_sum = 0
        for j in range(len(adj)):
            for k in range(len(adj[j])):
                if adj_list[i][j][k] == adj[j][k]:
                    equal_edge_sum += 1
        equal_edges = equal_edge_sum / len(adj)**2
        
        score1 = alpha * np.mean(vertices_dist) + (1 - alpha) * np.mean(dual_vertices_dist)
        score2 = beta * poly_interior/len(lattice_points) + (1 - beta) * dual_interior/len(lattice_points)
        score3 = gamma * equal_edge_sum
        scores.append(score1 + score2 + score3)
     
    return scores


def scorer_temp(dataset):
    feat_list = dataset[1]
    
    scores = []
    for i in range(len(feat_list)):
        vertices = []
        for j in range(len(feat_list[i])):
            vertex = []
            for k in range(len(feat_list[i][j])):
                vertex.append(float(feat_list[i][j][k]))
            vertices.append(vertex)
            
        vertices_dist = []
        for j in range(len(vertices)):
            cum_sum = 0
            for k in range(len(vertices[j])):
                cum_sum += (abs(vertices[j][k]) % 1)**2
            vertices_dist.append(math.sqrt(cum_sum))
        
        score = np.mean(vertices_dist) 
        scores.append(1-score)
     
    return scores
    
# use the untrained generator to create polytopes
noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
generated = generator(noise, training=False)
scores = scorer_temp(generated)
print('Untrained Score: ',np.mean(scores))

#%% Losses

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_disc_output, fake_disc_output):
    real_loss = cross_entropy(tf.ones_like(real_disc_output), real_disc_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_disc_output), fake_disc_output)
    D_loss = real_loss + fake_loss
    return D_loss

def generator_loss(fake_disc_output, fake_value_output):
    WGAN_loss = cross_entropy(tf.ones_like(fake_disc_output), fake_disc_output) 
    RL_loss = cross_entropy(tf.ones_like(fake_value_output), fake_value_output) 
    return 0.5 * WGAN_loss + 0.5 * RL_loss

def value_loss(real_value_output, fake_value_output, true_real_value, true_fake_value):
    V_loss = tf.reduce_mean((real_value_output - true_real_value) ** 2 + (fake_value_output - true_fake_value)) ** 2
    return V_loss


#%% Train

gen_optim = tf.keras.optimizers.Adam(1e-4)
disc_optim = tf.keras.optimizers.Adam(1e-4)
value_optim = tf.keras.optimizers.Adam(1e-4)

def train(adj_dataset, feat_dataset, epochs):
    
    for epoch in range(epochs):
        step = 0
        epoch_start_time = time.time()
        
        for i in range(0,len(adj_dataset),batch_size):
            step += 1
            
            dataset = [tf.convert_to_tensor(adj_dataset[i:i+batch_size]),
                       tf.convert_to_tensor(feat_dataset[i:i+batch_size])]
            
            noise = np.random.normal(0, 1, size=(len(dataset[0]), noise_dim))
            
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as val_tape:
                real_disc_output = discriminator(dataset,training=True)
                real_value_output = valuer(dataset,training=True)
                true_real_value = scorer_temp(dataset)
                
                generated = generator(noise, training=True)
                
                fake_disc_output = discriminator(generated,training=True)
                fake_value_output = valuer(generated,training=True)
                true_fake_value = scorer_temp(generated)
                
                disc_loss = discriminator_loss(real_disc_output, fake_disc_output)
                gen_loss = generator_loss(fake_disc_output, fake_value_output)
                val_loss = value_loss(real_value_output, fake_value_output, true_real_value, true_fake_value)
            
            grad_of_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            grad_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
            grad_of_val = val_tape.gradient(val_loss, valuer.trainable_variables)
        
            if step % 5 == 0:
                gen_optim.apply_gradients(zip(grad_of_gen, generator.trainable_variables))
                value_optim.apply_gradients(zip(grad_of_val, valuer.trainable_variables))
            else:                
                disc_optim.apply_gradients(zip(grad_of_disc, discriminator.trainable_variables))
        
        eta = timedelta(seconds=int((time.time() - epoch_start_time) * (epochs - epoch)))
        print('Epoch {}, ETA: {}'.format(epoch+1, eta))

train(A_train[:2000], X_train[:2000], epochs)


#%% Test

noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
generated = generator(noise, training=False)
scores = scorer(generated)
print('Trained Score: ',np.mean(scores))


    
