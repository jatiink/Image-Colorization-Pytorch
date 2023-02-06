import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from dataset import *
from visual import *
from support_functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

board = SummaryWriter("board")

train_data = MyDataset("train_data.txt")
test_data = MyDataset("test_data.txt")

train_loader = DataLoader(train_data, batch_size=20, shuffle=True, num_workers=3)
test_loader = DataLoader(test_data, batch_size=20, shuffle=False, num_workers=3)

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=1, out_channels=2, init_features=32, pretrained=False)

model.load_state_dict(torch.load('model_saves\Epoch_8.pt', map_location=torch.device('cpu')))
model.eval()

model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def training():

    epochs = 10
    i = 0
    b = 0

    for epoch in range(9,epochs):

        for step, data in enumerate(train_loader, 1):
            inputs, labels, names = data["image"].to(torch.float), data["target"], data["name"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            print(f'Epoch: {epoch+1} Step: {step} Loss: {loss.item()}')
            if step == 1:
                img_list = get_visual(inputs, labels, outputs)
                board_add_images(board, "Train", img_list, i, names = names)
            if step % 5000 == 0:
                img_list = get_visual(inputs, labels, outputs)
                board_add_images(board, "Train", img_list, i, names = names)

        with torch.no_grad():
            for vstep, vdata in enumerate(test_loader, 1):
                vinputs, vlabels, vnames = vdata["image"].to(torch.float), vdata["target"], vdata["name"]
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                b += 1
                if vstep % 50 == 0:
                    img_list = get_visual(vinputs, vlabels, voutputs)
                    board_add_images(board, "Test", img_list, b, names=vnames)
                    board.add_scalars("Test_losses", {"Net_loss": vloss.item()}, b)

        torch.save(model.state_dict(), ("model_saves/Epoch_" + str(epoch) + '.pt'))

if __name__ == '__main__':
    training()