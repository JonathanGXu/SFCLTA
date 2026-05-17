import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist

class VAE(nn.Module):
    def __init__(self,input_dim, latent_dim, hidden_dim=400):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,latent_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,input_dim),
            nn.Sigmoid()
        )

    def guide(self,x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data",x.shape[0]):
            h = self.encoder(x.flatten(1))
            z_loc, z_logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
            z_scale = torch.exp(0.5 * z_logvar)
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    def model(self,x):
        pyro.module("decoder",self.decoder)
        with pyro.plate("data",x.shape[0]):
            z_loc = torch.zeros(x.shape[0],self.latent_dim).to("cuda")
            z_scale = torch.ones(x.shape[0],self.latent_dim).to("cuda")
            z = pyro.sample("z", dist.Normal(z_loc,z_scale).to_event(1))
            x_recon   = self.decoder(z)
            pyro.sample("obs",dist.Normal(x_recon, 0.1).to_event(1), obs=x.flatten(1))

