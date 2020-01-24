import torch
import torch.nn as nn
import numpy as np


class zi2zi:
    def __init__(self, emb_size, emb_dim, output_dim=256, L1_penalty=100, Lconst_penalty=15,
                 Ltv_penalty=0.0, Lcategory_penalty=1.0):
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.category_embedding = self.embedding_lookup(emb_size, emb_dim)

        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
        self.Lcategory_penalty = Lcategory_penalty

        self.output_dim = output_dim

    def embedding_lookup(self, emb_size, emb_dim):
        W1 = torch.FloatTensor(np.random.uniform(-1, 1, size=(emb_size, emb_dim)))
        embedded_user = torch.nn.Embedding(emb_size, emb_dim, _weight=W1)
        embedded_user.weight.requires_grad = False
        embedded_users = torch.unsqueeze(embedded_user, -1)
        return embedded_users

    def build_model(self, data, optimizations, losses, generator, discriminator):
        real_A = data["real_A"]
        real_B = data["real_B"]
        embedding_ids = data["embedding_ids"]
        embdding = self.category_embedding[embedding_ids]

        # Generator
        optimizations['generator'].zero_grad()
        fake_B, encoded_real_A = generator.generator(real_A, embdding)
        encoded_fake_B, _ = generator.encoder(fake_B)
        fake_AB = torch.cat((real_A, fake_B), 1)

        fake_dis, fake_dis_logits, fake_cat_logits = discriminator(fake_AB)

        # 1. L1 loss between real and generated images
        l1_loss = losses['l1_loss'](fake_B-real_B)
        # 2. Category loss
        true_labels = torch.FloatTensor(real_A.size[0], self.emb_size).scatter_(1, embedding_ids, 1)
        fake_cat_loss = losses['cat_loss'](fake_cat_logits, true_labels)
        # 3. Encoding constant loss
        const_loss = losses['const_loss'](encoded_real_A-encoded_fake_B)
        # 4. TV loss
        # TODO: Check TV loss
        width = self.output_dim
        tv_loss = (losses['tv_loss'](fake_B[:, :, 1:, 1] - fake_B[:, :, :width-1, :]) +
                   losses['tv_loss'](fake_B[:, :, :, 1:] - fake_B[:, :, :, :width-1])) / width
        # 5. Discriminator fake
        cheat_loss = losses['dis_loss'](fake_dis_logits, torch.ones_like(fake_dis_logits))
        loss = l1_loss * self.L1_penalty + fake_cat_loss * self.Lcategory_penalty + \
                const_loss * self.Lconst_penalty + tv_loss * self.Ltv_penalty + cheat_loss

        loss.backward()
        optimizations['generator'].step()

        # Discriminator
        optimizations['discriminator'].zero_grad()
        fake_B, encoded_real_A = generator.generator(real_A, embdding)
        encoded_fake_B, _ = generator.encoder(fake_B)
        real_AB = torch.cat((real_A, real_B), 1)
        fake_AB = torch.cat((real_A, fake_B), 1)

        real_dis, real_dis_logits, real_cat_logits = discriminator(real_AB)
        fake_dis, fake_dis_logits, fake_cat_logits = discriminator(fake_AB)

        # 1. Dis loss
        d_loss_real = losses['dis_loss'](real_dis_logits, torch.ones_like(real_dis_logits))
        d_loss_fake = losses['dis_loss'](fake_dis_logits, torch.zeros_like(fake_dis_logits))
        # 2. Category loss
        cat_loss = losses['cat_loss'](real_cat_logits, true_labels)
        discriminator_loss = d_loss_real + d_loss_fake + cat_loss / 2.0
        discriminator_loss.backward()
        optimizations['discriminator'].step()



















