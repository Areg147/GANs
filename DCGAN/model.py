import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

class DCGAN(Model):

    def __init__(self,latent_dim,k_steps,*args,**kwargs):
        super().__init__(args,kwargs)

        self.latent_dim = latent_dim # noise prior pg(z)
        self.G = self.make_generator() #Generator
        self.D = self.make_discriminator() # Discriminator
        self.k_steps = k_steps # K steps to train D and 1 step G

    def make_discriminator(self):
        inputs = layers.Input(shape = (73,73,3))
        Conv_1_D = layers.Conv2D(128,(3,3),(2,2))(inputs)
        batch_norm_1_D = layers.BatchNormalization()(Conv_1_D)
        LeakyReLU_1 = layers.LeakyReLU(alpha=0.2)(batch_norm_1_D)
        

        Conv_2_D = layers.Conv2D(256,(3,3),(2,2))(LeakyReLU_1)
        batch_norm_2_D = layers.BatchNormalization()(Conv_2_D)
        LeakyReLU_2 = layers.LeakyReLU(alpha=0.2)(batch_norm_2_D)
        

        Conv_3_D = layers.Conv2D(512,(3,3),(2,2))(LeakyReLU_2)
        batch_norm_3_D = layers.BatchNormalization()(Conv_3_D)
        LeakyReLU_3 = layers.LeakyReLU(alpha=0.2)(batch_norm_3_D)
        
        
        Flatten = layers.Flatten()(LeakyReLU_3)
        outputs = layers.Dense(1,activation="sigmoid")(Flatten)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Discriminator")
        return model
    
    def make_generator(self):
        inputs  = layers.Input(shape = (self.latent_dim,))
        projection_layer_G = layers.Dense(4*4*1024,use_bias=False)(inputs)
        reshape_layer_G = layers.Reshape((4,4,1024))(projection_layer_G)
        batch_norm_1_G = layers.BatchNormalization()(reshape_layer_G)

        Conv_1_G = layers.Conv2DTranspose(512,(2,2),(2,2),activation="relu")(batch_norm_1_G)
        batch_norm_2_G = layers.BatchNormalization()(Conv_1_G)

        Conv_2_G = layers.Conv2DTranspose(256,(2,2),(2,2),activation="relu")(batch_norm_2_G)
        batch_norm_3_G = layers.BatchNormalization()(Conv_2_G)

        Conv_3_G = layers.Conv2DTranspose(128,(2,2),(2,2),activation="relu")(batch_norm_3_G)
        batch_norm_4_G = layers.BatchNormalization()(Conv_3_G)
        
        Conv_4_G = layers.Conv2DTranspose(64,(5,5),(1,1),activation="relu")(batch_norm_4_G)
        batch_norm_5_G = layers.BatchNormalization()(Conv_4_G)
        

        outputs = layers.Conv2DTranspose(3,(3,3),(2,2),activation="tanh")(batch_norm_5_G)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Generator")
        return model
    
    def compile(self,G_loss,D_loss,optimizer_G,optimizer_D):

        super().compile()
        self.G_loss = G_loss
        self.D_loss = D_loss
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G

    
    def train_step(self,input_):

        #Sample minibatch of m noise samples {z (1) , . . . , z (m)} from noise prior pg(z)
        noise = tf.random.normal(shape=(128,self.latent_dim))
        lables_real = tf.ones(shape=(128,1))
        lable_fake = tf.zeros(shape=(128,1))
        concated_label = tf.concat((lables_real,lable_fake),axis=0)

        # Update the discriminator by ascending its stochastic gradient:
        for i in range(self.k_steps):
            with tf.GradientTape() as tape_D:
                D_X_i = self.D(input_)
                D_G_Z_i = self.D(self.G(noise))
                concated_pred = tf.concat((D_X_i,D_G_Z_i),axis=0)
                # Update the discriminator by ascending its stochastic gradient: equivalently descending - D_loss_
                D_loss_ = self.D_loss(concated_label,concated_pred)

            gradients_of_discriminator = tape_D.gradient(D_loss_, self.D.trainable_variables)
            self.optimizer_D.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

        
        with tf.GradientTape() as tape_G:
            D_G_Z_i = self.D(self.G(noise))
            G_loss_ = self.G_loss(lables_real,D_G_Z_i)
            
        # Update the generator by ascending its stochastic gradient: equivalently descending - G_loss_ -Log(D(G(z_i)))
        gradients_of_generator = tape_G.gradient(G_loss_, self.G.trainable_variables)
        self.optimizer_G.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
        
        return {"G_loss_":G_loss_, "D_loss_":D_loss_}
