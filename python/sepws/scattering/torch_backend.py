import torch
import torch.nn.functional as F
from torch import Tensor
    
class TorchBackend:
    def __init__(self) -> None:
        pass
        
    def pad(self, x, sizes):
        """
        Pads the last n dimensions with reflection padded given a 2n size array in reverse dimension order (dim -1 before, dim -1 after, dim -2 before, dim -2 after, ...).
        """
        return F.pad(x, sizes, mode='reflection')    
    
    def fft1d(self, x, dim):
        """
        Takes the fft along a single arbitrary dimension
        """
        return torch.fft.fft(x, dim=dim)
    
    def ifft1d(self, x, dim):
        """
        Takes the ifft along a single arbitrary dimension
        """
        return torch.fft.ifft(x, dim=dim)
    
    def fft(self, x, dims):
        """
        Takes the fft along any number of arbitrary dimensions
        """
        return torch.fft.fftn(x, dim=dims)
    
    def ifft(self, x, dims):
        """
        Takes the ifft along any number of arbitrary dimensions
        """
        return torch.fft.ifftn(x, dim=dims)
    
    def freq_downsample1d(self, X: Tensor, d: int, dim: int):
        """
        Performs frequency periodisation to downsample in the frequency domain along a given dimension.
        
        Args:
            X (Tensor): Tensor assumed to be in the frequency domain along dimension dim. Must be of shape (Nbatch, ...), i.e, must start with a batch dimension.
            d (int): downsampling factor
            dim (int): dimension along which to perform downsampling
        
        Returns:
            A Tensor which has been downsampled by a factor d in the specified dimension. The returned tensor is still in the frequency domain.
        """        
        if dim == len(X.shape) - 1:
            return X.view(*X.shape[0:dim], d, X.shape[dim]//d).mean(dim=d, keepdim=False)
        else:
            return X.view(*X.shape[0:dim], d, X.shape[dim]//d, X.shape[dim+1:]).mean(dim=d, keepdim=False)
        
    def mul1d(self, x: Tensor, y: Tensor, dim: int):
        """
        Multiplies a multi-dimensional tensor x with a 1 dimensional tensor y along dimension dim, broadcasting to other dimensions.
        
        Args:
            X (Tensor): input tensor of shape (..., N, ...)
            y (Tensor): input tensor of shape (N,)
            dim (int): dimension of x along which to multiply, which must be the same size as the number of elements in y.
        
        Returns:
            x*y, broadcasted to all dimensions.
        """
        target_shape = [1 for _ in range(len(x.shape))]
        target_shape[dim] = y.shape[0]
        return x * y.view(target_shape)  
    
    def modulus(self, x: Tensor):
        return x.abs()          
    
    def unpad(self, x: Tensor):
        """
        Unpads a Tensor x which has been padded correctly (half of the time support of the filter on each side) and then accordingly downsampled.
        
        Args: 
            x (Tensor): a 2-4D tensor of shape (Nbatch, ...). Must be, at maximum, a 4D tensor. A first batch dimension is required.
        
        Returns:
            The unpadded tensor (fist and last samples discarded in each signal dimension).        
        
        """
        ndims = len(x.shape) - 1
        if ndims == 1:
            return x[:, 1:-1]
        if ndims == 2:
            return x[:, 1:-1, 1:-1]
        if ndims == 3:
            return x[:, 1:-1, 1:-1, 1:-1]
        raise NotImplementedError(f'A maximum 3 dimensions is supported. Got {ndims} dimensions.')
        