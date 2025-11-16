from torch import nn


'''
    Simple CNN Auto-Encoder to 
    use as input for Wave2Vec Model
'''
class CNN_Encoder(nn.Module):
    def __init__(self, in_channels:int = 3,*args, **kwargs):
        super(CNN_Encoder,self).__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels,16,3,stride=1,padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2,stride = 2),
            nn.Conv2d(16,8,3,stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,3,in_channels,stride = 2,padding=1, output_padding=1),
            nn.Sigmoid()
        )
    '''
        Compute Model Forward Pass
        Encoder -> Decoder
        input is (batch_size, channels, height, width)
    '''
    def forward(self,x):
        z_hat = self.encoder(x)
        y_hat = self.decoder(z_hat)
        return y_hat


if __name__ == "__main__":
    import torch
    x = torch.rand(size = (1,4,20,10)) 
    model = CNN_Encoder(in_channels=4)
    print(model.forward(x))