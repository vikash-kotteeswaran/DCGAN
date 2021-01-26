import torch
import torch.optim as optimizer
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DCGAN_model import Discriminator, Generator, Initialize_weights

#Set the device to which things are sent (cpu or gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
lr = 2e-4
batch_size = 128
num_epochs = 10
img_size = 64
img_channels = 1
noise_channels = 100
disc_features = 64
gen_features = 64

#Transform the data to needs
transforms=transforms.Compose([
	transforms.Resize(img_size),
	transforms.ToTensor(),
	transforms.Normalize([0.5 for i in range(img_channels)], [0.5 for i in range(img_channels)])
	])

#Required Dataset
dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms, download = True)

#If the dataset is our own
#dataset = datasets.ImageFolder(root = "[folder_name]", transform = transforms)

#Dataloader with batches
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

#Referncing Models and Initializing them
gen= Generator(noise_channels, gen_features, img_channels).to(device)
disc= Discriminator(img_channels, disc_features).to(device)
Initialize_weights(gen)
Initialize_weights(disc)

#Optimizer
gen_opt=optimizer.Adam(gen.parameters(), lr = lr, betas = (0.5, 0.999))
disc_opt=optimizer.Adam(disc.parameters(), lr = lr, betas = (0.5, 0.999))
criterion = torch.nn.BCELoss()

#Noise
test_noise = torch.randn([32, noise_channels, 1, 1]).to(device)

#Logs on tensorboard
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step=0

#Setting to train mode
gen.train()
disc.train()

#Train the GAN
for epoch in range(num_epochs):
	for batch_idx, (real, _) in enumerate(dataloader):
		real = real.to(device)
		Noise = torch.randn([batch_size, noise_channels, 1, 1]).to(device)
		fake = gen(Noise)

		#Train the Discriminator
		disc_real = disc(real).reshape(-1)
		loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
		disc_fake = disc(fake.detach()).reshape(-1)
		loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
		disc_loss = (loss_disc_real + loss_disc_fake)/2
		disc.zero_grad()
		disc_loss.backward(retain_graph = True)
		disc_opt.step()

		#Train the Generator
		output = disc(fake).reshape(-1)
		gen_loss = criterion(output, torch.ones_like(output))
		gen.zero_grad()
		gen_loss.backward()
		gen_opt.step()

		#Progess of learning 
		if batch_idx%100 ==0:
			print(f"EPOCH : [{epoch}/{num_epochs}])    \
				BATCH : [{batch_idx}/len(Dataloader)]    \
				Disc_Loss : {disc_loss:.4f}  Gen_Loss : {gen_loss:.4f}"
				)

			#Watching the Progress through displaying the Reals and the Fakes
			with torch.no_grad():
				fake = gen(test_noise)
				img_grid_real = torchvision.utils.make_grid(real[:32], normalize = True)
				img_grid_fake =  torchvision.utils.make_grid(fake[:32], normalize = True)

				writer_real.add_image("Real", img_grid_real, global_step = step)
				writer_fake.add_image("Fake", img_grid_fake, global_step = step)
			step+=1