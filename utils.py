import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pathlib
from typing import List
from .models import (
	BCE_KLD_loss, MSE_KLD_loss,
	ConvPoolEncoder, MaxPoolEncoder, 
	Decoder, VAE
	)


def create_folders(*folders: List[str], parents=True, exist_ok=True):
	for folder in folders:
		pathlib.Path(folder).mkdir(parents=parents, exist_ok=exist_ok)


def get_dataloader(folder: str, is_train: bool, batch_size: int):
	dataset = datasets.MNIST(folder, 
							train=is_train, 
							download=True,
							transform=transforms.ToTensor())

	dataloader = torch.utils.data.DataLoader(dataset,
											batch_size=batch_size,
											shuffle=is_train)
	return dataloader


def get_model(is_pool_conv=False):
	encoder_class = ConvPoolEncoder if is_pool_conv else MaxPoolEncoder
	encoder = encoder_class()
	decoder = Decoder()
	model = VAE(encoder, decoder)
	return model


def get_loss(is_loss_bce=True):
	return BCE_KLD_loss if is_loss_bce else MSE_KLD_loss


def train_epoch(epoch: int, dataloader, model, loss_function, device, optimizer) -> int:
    model.train()
    train_loss = 0

    for batch_idx, (target, _) in enumerate(dataloader):
        target = target.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(target)
        loss = loss_function(output, target, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
            
    train_loss /= len(dataloader.dataset)

    return train_loss


def test_epoch(epoch: int, dataloader, model, loss_function, device, save: bool) -> int:
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for i, (target, _) in enumerate(dataloader):
            target = target.to(device)
            output, mu, logvar = model(target)
            loss = loss_function(output, target, mu, logvar).item()
            test_loss += loss

            if i == 0 and save:
                n = min(target.shape[0], 16)
                comparison = torch.cat([target[:n], output[:n]])
                save_image(comparison.cpu(), f'results/test/{epoch:02d}.png', nrow=n)

    test_loss /= len(dataloader.dataset)

    return test_loss


def create_gif(model, test_X: torch.Tensor, filename: str, size: int, N: int=10):
    model.eval()
    img = None
    imgs = []

    resize = transforms.Resize(size)
    to_pil = transforms.ToPILImage()
    apply_transforms = lambda x: (resize(to_pil(x))) 

    with torch.no_grad():
        for i in range(1, len(test_X)):
            starting = test_X[i-1]
            ending = test_X[i]

            starting_mu, starting_logvar = model.encoder(starting.unsqueeze(0))
            ending_mu, ending_logvar = model.encoder(ending.unsqueeze(0))

            starting_vector = starting_mu + torch.exp(0.5*starting_logvar)
            ending_vector = ending_mu + torch.exp(0.5*ending_logvar)

            starting_decoded = last_activation(model.decoder(starting_vector))
            ending_decoded = last_activation(model.decoder(ending_vector))

            part = (ending_vector-starting_vector)/N
            middle_vector = starting_vector
            
            if img is None:
                decoded = model.decoder(middle_vector)
                img = apply_transforms(decoded.squeeze())

            for j in range(N):
                middle_vector += part 
                middle_decoded = model.decoder(middle_vector)
                imgs.append(apply_transforms(middle_decoded.squeeze()))

    img.save(filename, format="GIF", append_images=imgs, save_all=True, duration=200, loop=0)

    return filename