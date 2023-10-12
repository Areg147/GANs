# GANs
𝐈𝐦𝐩𝐥𝐞𝐦𝐞𝐧𝐚𝐭𝐚𝐭𝐢𝐨𝐧 𝐨𝐟 𝐭𝐡𝐢𝐬 𝐩𝐚𝐩𝐞𝐫 -- https://arxiv.org/abs/1511.06434
𝐦𝐨𝐝𝐞𝐥 𝐠𝐞𝐧𝐞𝐫𝐚𝐭𝐞 𝟕𝟑 𝐱 𝟕𝟑 𝐩𝐢𝐱𝐞𝐥 𝐩𝐞𝐨𝐩𝐥𝐞𝐬 𝐢𝐦𝐚𝐠𝐞𝐬

The authors of DCGAN offer us a family of convolutional neural network models that provide much more stable training in the context of GANs. In the 'model.py' file, you can find the proposed architecture, with almost everything matching the description in the paper. For example, they use Leaky Relu in the Discriminator, Batchnorms in both the Discriminator and Generator, ReLU in the Generator, and so on. The only change I made was to the generator architecture; I added one more Deconvolution layer before the final layer.

In the GAN paper, which is available at https://arxiv.org/abs/1406.2661, the authors proposed an algorithm to train models based on the following loss function:

min G max D 𝐋(𝐃, 𝐆) = 𝐄[𝐥𝐨𝐠(𝐃(𝐱))] + 𝐄[𝐥𝐨𝐠(1 − 𝐃(𝐆(𝐳)))],

where 𝐱 is sampled from the data distribution 𝐩𝐝𝐚𝐭𝐚(𝐱), and 𝐳 is sampled from the noise distribution 𝐩𝐳(𝐳). The training process involves K steps to update D and one step to update G. In my implementation, I set K=1, and I sample only one noisy vector from a Normal distribution, as opposed to a uniform distribution, for updating the weights of both networks.

𝐓𝐫𝐚𝐢𝐧 𝐜𝐨𝐧𝐟𝐢𝐠𝐬:
Batch_size = 128
G_opt = Adam lr=0.0002, beta_1=0.5
D_opt = Adam lr=0.0002, beta_1=0.5
Leaky(alpha=0.2)
k_steps_to_train_D = 1

𝐑𝐄𝐒𝐔𝐋𝐓𝐒
![results](https://github.com/Areg147/GANs/assets/131033594/7f0bb009-cd45-493a-870f-b8a13ba41508)
