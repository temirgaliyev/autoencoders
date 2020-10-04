import torch
from torchvision.utils import save_image

from timeit import default_timer as timer
from os import path

from .utils import create_folders, get_dataloader, get_model, get_loss, train_epoch, test_epoch


def train(epochs=1000, batch_size=1024, cuda=True, is_loss_bce=True, is_pool_conv=False):
	print("Initialization...")
	WEIGHT_FILENAME_PREFIX = "WEIGHT_{}_{}_".format(
							'BCE' if is_loss_bce else 'MSE',
							'CONV' if is_pool_conv else 'MAXP')
	cuda_available = torch.cuda.is_available()
	device = torch.device("cuda" if cuda_available and cuda else "cpu")

	print("Creating folders...")
	create_folders("data", 
		path.join("results", "rand"), 
		path.join("results", "test"), 
		path.join("results", "weights"))

	print("Loading MNIST...")
	train_loader = get_dataloader("data", True, batch_size)
	test_loader = get_dataloader("data", False, batch_size)

	random_test = torch.randn(64, 16).to(device)

	model = get_model(is_pool_conv).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	loss_function = get_loss(is_loss_bce)

	print("Training...")
	train_losses, test_losses = [], []
	for epoch in range(1, epochs+1):

		time_started = timer()

		train_loss = train_epoch(epoch, train_loader, model, loss_function, device, optimizer)
		test_loss = test_epoch(epoch, test_loader, model, loss_function, device, True)

		time_elapsed = timer() - time_started

		print(f"Epoch: {epoch}/{epochs:02d} | Train loss: {train_loss:02.7f} | Test loss: {test_loss:02.7f} | Time: {time_elapsed}")

		train_losses.append(train_loss)
		test_losses.append(test_loss)

		with torch.no_grad():
			random_sample = model.decoder(random_test).cpu()
			save_image(random_sample, f"results/rand/{epoch:02d}.png")

		if epoch%100 == 0 or epoch == epochs:
			filename = f"{epoch}_{test_loss:02.7f}.torch"
			weight_path = path.join("results", "weights", WEIGHT_FILENAME_PREFIX+filename)
			torch.save(model, weight_path)
			print(f"SAVING weights at {weight_path}")

	print("Completed!")
