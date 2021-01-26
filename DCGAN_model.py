import torch.nn  as nn
import torch

#Discriminator model according to DCGAN Paper
class Discriminator(nn.Module):
	def __init__(self, img_channels, out_features):
		super(Discriminator,self).__init__()
		self.disc=nn.Sequential(
			nn.Conv2d(img_channels, out_features, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2),
			self.block(out_features,out_features*2,4,2,1),
			self.block(out_features*2,out_features*4,4,2,1),
			self.block(out_features*4,out_features*8,4,2,1),
			nn.Conv2d(out_features*8,1,kernel_size=4,stride=2,padding=0),
			nn.Sigmoid())

	def block(self,in_channel, out_channel, kernel, strides, padding):
		return nn.Sequential(
			nn.Conv2d(in_channel, out_channel, kernel, strides, padding, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.LeakyReLU(0.2))

	def forward(self,x):
		return self.disc(x)

#Generator model according to DCGAN Paper
class Generator(nn.Module):
	def __init__(self, noise_channels, out_features, img_channels):
		super(Generator, self).__init__()
		self.gen=nn.Sequential(
			self.block(noise_channels,out_features*16,4,1,0),
			self.block(out_features*16,out_features*8,4,2,1),
			self.block(out_features*8,out_features*4,4,2,1),
			self.block(out_features*4,out_features*2,4,2,1),
			nn.ConvTranspose2d(out_features*2, img_channels, kernel_size=4, stride=2, padding=1),
			nn.Tanh()
			)

	def block(self, in_channel, out_channel, kernel, strides, padding):
		return nn.Sequential(
			nn.ConvTranspose2d(in_channel, out_channel, kernel, strides, padding, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU())

	def forward(self,x):
		return self.gen(x)

#Initializing weights to the model
def Initialize_weights(model):
	for m in model.modules():
		if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
			nn.init.normal_(m.weight.data, 0.0, 0.02)

#Testing whether the models return proper outputs
def test():
	N, img_channels, height, width= 4, 3, 64, 64
	noice_channels=100
	X = torch.randn([N, img_channels, height, width])
	disc = Discriminator(img_channels, 8)
	Initialize_weights(disc)
	assert disc(X).shape == (N, 1, 1, 1), "Discriminator didn't produce the correct output"
	gen = Generator(noice_channels, 8, img_channels)
	Initialize_weights(gen)
	Z = torch.randn([N, noice_channels, 1, 1])
	assert gen(Z).shape == (N, img_channels, height, width), "Generator didn't produce the correct output"
