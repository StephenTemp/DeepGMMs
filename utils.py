# IMPORTS
# ------------------
from torch import nn

# FUNCTION: Flatten( [] ) => []
# SUMMARY: provided a multidimensional array, flatten 
#          to one dimension for processing 
class Flatten(nn.Module):
    def flatten(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
        
    def forward(self, x):
        return self.flatten(x)