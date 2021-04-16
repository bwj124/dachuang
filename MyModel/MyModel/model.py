from torch import nn

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(300, 202),
            nn.LeakyReLU(inplace=True),
            nn.Linear(202, 121),
            nn.LeakyReLU(inplace=True),
            nn.Linear(121, 2)
        )

    def forward(self, x):
        x = self.model(x)

        return x
