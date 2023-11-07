from scattering import scattering_1d as s1d
from scattering import morlet
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import matplotlib as mpl

from torch.nn import Conv2d

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update(mpl.rcParamsDefault)

class ScatteringFBTFConfig:
    """
    Configure a TF scattering FB using 2 1D filterbanks. The 2D wavelet is factorised as psi(t, log(lambda)) = psi(t) * psi(log(lambda)).
    """
    def __init__(self, Q_time:int, Q_quefrequency, T_time, T_bins_quefrequency, fs, approximation_support=5.0, ds_mode = s1d.DownsampleMode.MULTIPLE_OF_LPF) -> None:  
        self.time_config = s1d.ScatteringFB1DConfig(Q_time, T_time, fs, 
                                              approximation_support=approximation_support, downsample_mode = ds_mode)
        self.quefrequency_config = s1d.ScatteringFB1DConfig(Q_quefrequency, T_bins_quefrequency*2*np.pi, 1.0, approximation_support=approximation_support, 
                                                      downsample_mode = ds_mode)
    
    def from_1D(conf1d: s1d.ScatteringFB1DConfig, Q_time, Q_quefrequency, T_bins_quefrequency, approximation_support=5.0):
        return ScatteringFBTFConfig(Q_time, Q_quefrequency, conf1d.T, T_bins_quefrequency, conf1d.morlets[0].output_fs, approximation_support=approximation_support)    

 
        
class ScatteringFilterBankTF:
    def __init__(self, config: ScatteringFBTFConfig) -> None:
        self.config = config
        self.time_fb = s1d.ScatteringFB1D(config.time_config)
        self.qf_fb_pos = s1d.ScatteringFB1D(config.quefrequency_config)
        self.qf_fb_neg = s1d.ScatteringFB1D(config.quefrequency_config, dir=-1.0)
        self._sample()        
        
    def _sample(self):
        # 2 sides of the half-plane grid
        nfilters = self.config.time_config.number_of_wavelets*self.config.quefrequency_config.number_of_wavelets*2
        # 1 side each of the filters on their respective axes
        nfilters += self.config.time_config.number_of_wavelets + self.config.quefrequency_config.number_of_wavelets
        self.filters = np.zeros((self.time_fb.filters.shape[0], self.qf_fb_pos.filters.shape[0], nfilters), dtype=s1d.FB_data_type)
        self.filter_grid_indices = {}
        
        k = 0
        
        #the positive and negative sides of the quefrequency half-plane along with the time-plane
        for a in range(self.config.time_config.number_of_wavelets):
            for b in range(self.config.quefrequency_config.number_of_wavelets):
                psi_t = self.time_fb.filters[:, [a]]
                psi_log_lambda = self.qf_fb_pos.filters[:, [b]].T
                self.filters[:, :, k] =  psi_t * psi_log_lambda
                self.filter_grid_indices[k] = (a,b)
                k += 1
                psi_log_lambda = self.qf_fb_neg.filters[:, [b]].T
                self.filters[:, :, k] =  psi_t * (psi_log_lambda)
                self.filter_grid_indices[k] = (a,b)
                k += 1
                
        #the filters on the time axis
        for a in range(self.config.time_config.number_of_wavelets):
            psi_t = self.time_fb.filters[:, [a]]
            phi_log_lambda = morlet.sample_gauss(self.qf_fb_pos.t, self.config.quefrequency_config.lpf_t_sigma)
            phi_log_lambda = phi_log_lambda[np.newaxis, :] #configure the transpose of the log(lambda) LPF
            self.filters[:, :, k] = psi_t * phi_log_lambda
            self.filter_grid_indices[k] = (a,-1)
            k += 1
        
        #the filters on the quefrequency axis
        for b in range(self.config.quefrequency_config.number_of_wavelets):
            psi_qf = self.qf_fb_pos.filters[:, [b]].T
            phi_t = morlet.sample_gauss(self.time_fb.t, self.config.time_config.lpf_t_sigma)
            phi_t = phi_t[:, np.newaxis] #configure the transpose of the log(lambda) LPF
            self.filters[:, :, k] = phi_t * psi_qf
            self.filter_grid_indices[k] = (-1,b)
            k += 1
            
        self.number_filters = k
        assert(k == nfilters)
        
        #construct the LPF        
        self.lpf = self.time_fb.lpf[:, np.newaxis] * self.qf_fb_pos.lpf[np.newaxis,:]
            
    def to_torch(self) -> Conv2d:
        pass
    
    def plot(self, oversample = 8):
        for k in range(self.number_filters):
            psi = self.filters[:, :, k]
            Psi = fftshift(np.abs(fft2(psi, s=(psi.shape[0]*oversample, psi.shape[1]*oversample)))).T
            X = (np.arange(Psi.shape[1])/Psi.shape[1]) - 0.5 #qf
            Y = (np.arange(Psi.shape[0])/Psi.shape[0]) - 0.5 #time           
            print(np.max(Psi))
            plt.contour(X, Y, Psi, levels=[5.0])
            
        
        plt.xlabel("$\omega_t$")
        plt.ylabel("$\omega_{log \lambda}$")
        # plt.xlim([0,0.5])
        
        print(self.number_filters)   
        
        plt.show(block=True)
        
    def plot_wavelet(self, k):
        psi = self.filters[:, :, k].T
        plt.subplot(311)
        plt.imshow(np.real(psi), aspect='equal')
        plt.subplot(312)
        plt.imshow(np.imag(psi), aspect='equal')
        plt.subplot(313)
        plt.imshow(np.abs(psi), aspect='equal')
        plt.show(block=True)
        
    def plot_lpf(self):
        plt.imshow(self.lpf.T)
        plt.xlabel('Time')
        plt.ylabel('Quefrequency')
        plt.show(block=True)
        
        
def get_eligible_filter_indices(fb0: ScatteringFilterBankTF, k0 , fb1: ScatteringFilterBankTF):    
        """
        For a given filter with index k0 in a FB fb0, compute all the relevant filter indices 
        in fb1 that will capture significant information.
        """          
        pass
                