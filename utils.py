import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pathlib


def create_folders(*folders, parents=True, exist_ok=True):
	for folder in folders:
		pathlib.Path(folder).mkdir(parents=parents, exist_ok=exist_ok)


def get_dataloader(folder: str, is_train: bool, batch_size: int):
	dataset = datasets.MNIST(folder, 
							train=is_train, 
							download=True,
                            transform=transforms.ToTensor())

	dataloader = torch.utils.data.DataLoader(dataset,
	                                        batch_size=dataset, 
	                                        shuffle=is_train)
	return dataloader


def train(epoch: int, dataloader, model, loss_function, device, optimizer) -> int:
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


def test(epoch: int, dataloader, model, loss_function, device, save: bool) -> int:
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


def create_gif(model, test_X: torch.Tensor, N=10):
	model.eval()

	with torch.no_grad():
	    for i in range(1, len(test_X)+1):
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

	        for j in range(N):
	            print(i*N+j)
	            middle_vector += part 
	            middle_decoded = last_activation(model.decoder(middle_vector))
	            save_image(middle_decoded, f"results/gif/{(i-1)*N+j+1:02d}.png")
	