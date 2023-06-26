import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets2 import KeyPointDatasets
from modeleyenew import IRModel
from utils import Visualizer, compute_loss,ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# h, w
IMG_SIZE = 376, 376

vis = Visualizer(env="keypoint")
K = [0.01, 0.03]
L = 255
window_size =3

def train(model, epoch, dataloader, optimizer, criterion, scheduler):
    model.train()

    for itr, (image, hm) in enumerate(dataloader):

        device = torch.device('cuda:0')

        hm = hm.to(device)
        image = image.to(device)


        bs = image.shape[0]

        output,output1 = model(image)


        #output=output.float()

        loss = criterion(output,hm)
        loss1 = criterion(output1, hm)
        loss=loss+loss1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if itr % 2 == 0:
            print("epoch:%2d|step:%04d|loss:%.6f" %
                  (epoch, itr, loss.item()/(bs)))
            vis.plot_many_stack({"train_loss": loss.item()/bs})


def test(model, epoch, dataloader, criterion):
    model.eval()
    sum_loss = 0.
    n_sample = 0
    for itr, (image, hm) in enumerate(dataloader):

        hm = hm.cuda()
        image = image.cuda()

        output,output1 = model(image)

        loss = criterion(output, hm)
        loss1 = criterion(output1, hm)
        loss = loss + loss1


        sum_loss += loss.item()
        n_sample += image.shape[0]

    print("TEST: epoch:%02d-->loss:%.6f" % (epoch, sum_loss/n_sample))
    if epoch > 1:
        vis.plot_many_stack({"test_loss": sum_loss/n_sample})
    return sum_loss / n_sample


if __name__ == "__main__":

    total_epoch = 3000
    bs =4
    ########################################
    transforms_all = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((500,480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                             std=[0.2479, 0.2475, 0.2485])


    ])

    datasets = KeyPointDatasets(root_dir="./dataeyeCHARS", transforms=transforms_all)

    data_loader = DataLoader(datasets, shuffle=True,
                             batch_size=bs, collate_fn=datasets.collect_fn,drop_last=True)
    datasets1 = KeyPointDatasets(root_dir="./dataeyeCHARS", transforms=transforms_all)

    data_loader1 = DataLoader(datasets1, shuffle=True,
                             batch_size=1, collate_fn=datasets.collect_fn,drop_last=True)
    model = IRModel()


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model=model.to(device)
        #model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion =torch.nn.MSELoss()# compute_loss
    #criterion=torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=30,
                                                gamma=0.1)

    for epoch in range(total_epoch):
        train(model, epoch, data_loader, optimizer, criterion, scheduler)
        loss = test(model, epoch, data_loader1, criterion)

        if epoch % 1 == 0:
            torch.save(model.state_dict(),
                        "weightseyeCHAS/epoch_%d_%.3f.pt" % (epoch, loss*10000))
