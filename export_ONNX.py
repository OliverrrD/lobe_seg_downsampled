from torch.autograd import Variable
from models import unet256, unet512, unet1024
import torch.onnx
import torchvision
import torch


if __name__ == "__main__":

    dummy_input =torch.randn(1, 1, 96, 96, 96).cuda()
    state_dict = torch.load('/home/local/VANDERBILT/dongc1/lobe_seg/models/config_1018_TS/config_1018_TS_best_model.pth')
    device = torch.device("cuda:0")
    model = unet256(6).to(device)
    model.load_state_dict(state_dict)
    torch.onnx.export(model, dummy_input, "/home/local/VANDERBILT/dongc1/unet256.onnx")