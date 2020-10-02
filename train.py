import torch

from .models import (
	MaxPoolEncoder, ConvPoolEncoder, Decoder, VAE,
	BCE_KLD_loss, MSE_KLD_loss
	)
from .utils import create_folders, get_dataloader, train, test
from timeit import default_timer as timer


BATCH_SIZE = 10000
CUDA = torch.cuda.is_available()
NUM_EPOCHS = 1000
IS_LOSS_BCE = True
IS_POOL_CONV = False


def main():
	device = torch.device("cuda" if CUDA else "cpu")
	loss_function = BCE_KLD_loss if IS_LOSS_BCE else MSE_KLD_loss
	encoder_class = ConvPoolEncoder if IS_POOL_CONV else MaxPoolEncoder

	create_folders("data", "results/rand", "results/test")

	train_loader = get_dataloader("data", True, BATCH_SIZE)
	test_loader = get_dataloader("data", False, BATCH_SIZE)

	random_test = torch.randn(64, 16).to(device)

	encoder = encoder_class()
	decoder = Decoder()
	model = VAE(encoder, decoder).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	train_losses, test_losses = [], []

	for epoch in range(1, NUM_EPOCHS+1):

	    time_started = timer()
	    
	    train_loss = train(epoch, train_loader, model, loss_function, device, optimizer)
	    test_loss = test(epoch, test_loader, model, loss_function, device, True)

	    time_elapsed = timer() - time_started

	    print(f"Epoch: {epoch}/{NUM_EPOCHS:02d} | Train loss: {train_loss:02.7f} | Test loss: {test_loss:02.7f} | Time: {time_elapsed}")

	    train_losses.append(train_loss)
	    test_losses.append(test_loss)

	    with torch.no_grad():
	        random_sample = model.decoder(random_test).cpu()
	        save_image(random_sample, f"results/rand/{epoch:02d}.png")


