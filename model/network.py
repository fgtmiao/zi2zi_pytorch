import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels=3, generator_dim=64):
        super(Generator, self).__init__()
        # Encode initial
        # Output in each layer: 64, 128, 256, 512, 512, 512, 512, 512
        out_channels = generator_dim
        channels_list = []
        self._encoder_layers = dict()
        #TODO: Check different between leakyrelu and relu
        for i in range(1, 9):
            channels_list.append(out_channels)
            self._encoder_layers["e%d"%i] = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            if out_channels < generator_dim * 8:
                out_channels *= 2

        # Decode initial
        channels_list = channels_list[::-1]
        channels_list[0] = channels_list[0] // 2
        self._decode_layers = dict()
        #TODO: Consider other batchnorm for decoder
        for i in range(1, 8):
            self._decode_layers["d%d"%i] = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(channels_list[i-1]*2, channels_list[i], kernel_size=5,
                                   stride=2, padding=2, output_padding=1),
            )
        self._decode_layers["d8"] = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(channels_list[-1]*2, in_channels, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.Tanh()
        )

    def encoder(self, inputs):
        encoder_outputs = dict()
        in_tmp = inputs
        for i in range(1, 9):
            in_tmp = self._encoder_layers["e%d" % i](in_tmp)
            encoder_outputs["e%d" % i] = in_tmp
        return in_tmp, encoder_outputs

    def decode(self, inputs, encoder_outputs):
        outputs = inputs
        for i in range(1, 8):
            outputs = self._decode_layers["d%d"%i](outputs)
            # Special of UNET
            outputs = torch.cat((outputs, encoder_outputs["e%d"%(8-i)]), 1)

        outputs = self._decode_layers["d%8"](outputs)
        return outputs

    def generator(self, inputs, embedding):
        last_encoder, encoder_outputs = self.encoder(inputs)
        # Concat embedding with last encoder layer
        encoder_embed = torch.cat((last_encoder, embedding), 1)
        outputs = self.decode(encoder_embed, encoder_outputs)
        return outputs, last_encoder


class Discriminator(nn.Module):
    def __init__(self, img_size, cat_num, in_channels=3, discriminator_dim=64):
        super(Discriminator, self).__init__()
        layers = []
        out_channels = discriminator_dim
        for i in range(4):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU())
            in_channels = out_channels
            out_channels *= 2

        layers.append(nn.Flatten())
        self.out = nn.Sequential(*layers)
        out_dim = ((img_size //16) ** 2) * discriminator_dim * 8
        self.out_real_fake = nn.Linear(out_dim, 1)
        self.out_cat = nn.Linear(out_dim, cat_num)

    def forward(self, input):
        out = self.out(input)
        # Check whether image is real or fake
        out_real_fake = self.out_real_fake(out)
        # Check category of the image (which fonts they belong to)
        out_cat = self.out_cat(out)
        return nn.Sigmoid()(out_real_fake), out_real_fake, out_cat




