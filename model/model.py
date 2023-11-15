#Model Module
import torch
from torch import nn

class CCN(nn.Module):
    def __init__(self):
        super(CCN, self).__init__()

        # Encoder
        self.encoder_conv1 = self.Conv_BN_Relu(2, 64)
        self.encoder_conv2 = self.Conv_BN_Relu(64, 128)
        self.encoder_conv3 = self.Conv_BN_Relu(128, 256)
        self.encoder_conv4 = self.Conv_BN_Relu(256, 512)

        # Decoder
        self.decoder_upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1,padding=1)
        self.decoder_conv1 = self.Conv_BN_Relu(512, 256)
        self.decoder_upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1,padding=1)
        self.decoder_conv2 = self.Conv_BN_Relu(256, 128)
        self.decoder_upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1,padding=1)
        self.decoder_conv3 = self.Conv_BN_Relu(128, 64)

        # Output Layer
        self.output = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=True)
       


    def Conv_BN_Relu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        encoder_conv1 = self.encoder_conv1(x)
        encoder_conv2 = self.encoder_conv2(encoder_conv1)
        encoder_conv3 = self.encoder_conv3(encoder_conv2)
        encoder_conv4 = self.encoder_conv4(encoder_conv3)

        # Decoder
        decoder_upsample1 = self.decoder_upsample1(encoder_conv4)
        # print(decoder_upsample1.shape)
        # print(encoder_conv1.shape)
        cat1 = torch.cat((decoder_upsample1, encoder_conv3), dim=1)
        decoder_conv1 = self.decoder_conv1(cat1)

        decoder_upsample2 = self.decoder_upsample2(decoder_conv1)
        cat2 = torch.cat((decoder_upsample2, encoder_conv2), dim=1)
        decoder_conv2 = self.decoder_conv2(cat2)

        decoder_upsample3 = self.decoder_upsample3(decoder_conv2)
        cat3 = torch.cat((decoder_upsample3, encoder_conv1), dim=1)
        decoder_conv3 = self.decoder_conv3(cat3)

        # Output Layer
        #output = self.output(decoder_conv3)

        return decoder_conv3


class CCN_Model(nn.Module):
    def __init__(self):
        super(CCN_Model, self).__init__()
        self.Net = CCN()
        self.RG = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.RB = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.GB = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=True)
    
    def forward(self,x):

        output_1 = self.Net(x[0])
        output_2 = self.Net(x[1])
        output_3 = self.Net(x[2])



        return self.RG(output_1),self.RB(output_2),self.GB(output_3)



def shuffle(tensor):
    # # Generate a random permutation of the channels
    # channel_permutation = torch.randperm(tensor.size(1))

    # # Shuffle the channels of the tensor
    # shuffled_tensor = tensor[:, channel_permutation]
    
    # return shuffled_tensor
    flattened_tensor = tensor.view(-1)

    # Shuffle the elements of the flattened tensor
    channel_permutation = torch.randperm(flattened_tensor.size(0))

    # Shuffled the channels of the tensor
    shuffled_tensor = flattened_tensor[channel_permutation].view(tensor.size())

    return shuffled_tensor


class MFCS(nn.Module):
    def __init__(self, channel=3):
        super(MFCS, self).__init__()
        self.channel = channel
        self.layer_1 = nn.Conv2d(channel, 64, 1, stride=1)
        self.layer_2 = nn.Conv2d(self.channel, 64, 3, padding=1)
        self.layer_3 = nn.Sequential(
            nn.Conv2d(self.channel, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(self.channel, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
        #self.shuffle = nn.PixelShuffle(upscale_factor=2)  # Add pixel shuffle layer for upsampling

    def forward(self, x):
        layer_1 = self.layer_1(x)
        layer_2 = self.layer_2(x)
        layer_3 = self.layer_3(x)
        layer_4 = self.layer_4(x)
        s1 = shuffle(layer_1)
        s2 = shuffle(layer_2)
        s3 = shuffle(layer_3)
        s4 = shuffle(layer_4)
        return torch.cat([s1, s2, s3, s4], 1)

class MFN(nn.Module):
    def __init__(self):
        super(MFN, self).__init__()
        self.mfcs = nn.Sequential(
            MFCS(),
            MFCS(512),
            MFCS(512),
            MFCS(512),
            MFCS(512),
            MFCS(512),
            MFCS(512),
            MFCS(512),
            MFCS(512)
        )
        self.weight_generator = nn.Conv2d(512, 3, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        layer_1 = self.mfcs(x)
        layer_1 = self.weight_generator(layer_1)
        enhanced_image = self.relu(x + layer_1)  # Adding a skip connection for better feature fusion
        return enhanced_image


class DEN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        

        self.conv_op = nn.Conv2d(64,3,3,stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        
        l1 = self.conv_relu(x)
        l2 = self.layer_2(l1)
        l3 = self.layer_3(l2)
        l4 = self.layer_4(l3)

        l5 = self.relu(self.conv_op(l4))

        return l5
    


        