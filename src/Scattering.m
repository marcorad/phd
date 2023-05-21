classdef Scattering
    properties(Constant)
    Q = 4;
    T = 128;
    fs = 2000;

    Ninvariance = floor(Scattering.T*Scattering.fs);
    sigmaRange = 5; %number of std deviations in IR
    fl = 25; %minimum frequency to consider
    fh = Scattering.fs/2; %maximum frequency to consider
    n_max = Scattering.irLength(); %max support according to fl    
    Nir = 2*Scattering.n_max + 1;
    Nsig = 4096;
    Nconvpsi = Scattering.Nsig + 2*Scattering.Nir -1; %number of conv points, accounting for reflection boundary
    lambdas = Scattering.getLambdasExponential();
    fc = Scattering.lambdas/2/pi;
    psi = Scattering.constructFB();
    fpsi = (0:Scattering.Nconvpsi-1)/Scattering.Nconvpsi*Scattering.fs;
    Psi = fft(Scattering.psi, Scattering.Nconvpsi, 2);
    phi = Scattering.invarianceFilter();
    end

    methods(Static)

        function g = gauss(t, sigma)
            g = exp(-(t/sigma).^2);
        end

        function th = theta(t)
            Q = Scattering.Q;
            th = Scattering.gauss(t, 2*Q);%sigma_t = 2*Q, sigma_w = 1/Q            
        end         

        %if lamda=1, then this is for 1 rad/s
        %for now, we don't care about the minimum scale
        function [psi, n, t] = morlet(lambda)   
            n = -Scattering.n_max:Scattering.n_max;
            t = n / Scattering.fs;
            Q = Scattering.Q;
            Thm1 = Scattering.gauss(-1, 1/Q); 
            Th0 = Scattering.gauss(0, 1/Q);
            psi =  lambda *( exp(1j*lambda*t) - Thm1/Th0) .* Scattering.theta(t*lambda); 
        end

        function phi = invarianceFilter()
            sigma_n = Scattering.T;
            nmax = floor(sigma_n *Scattering.sigmaRange); 
            n = -nmax:nmax;
            phi = exp(-(n/sigma_n).^2);
        end


        function lambda = minimumLambdaForInvarianceScale()
        %lambda >= 2*pi*Q*T -> T >= 2*pi*Q/lambda -> fl = lambda
        lambda = 2  * pi * Scattering.Q / T;
        end

        function T = invarianceScaleForLowestFrequency(fl)
        %lambda >= 2*pi*Q*T -> T >= 2*pi*Q/lambda -> fl = lambda
        T = 2 * pi * Scattering.Q / fl;
        end

        function n_range = irLength()
            sigma_t = 2*Scattering.Q/Scattering.fl;
            t_range = Scattering.sigmaRange * sigma_t;
            n_range = floor(t_range * Scattering.fs);
        end

        function lambdas = getLambdasExponential()                     
            lambdas(1) = Scattering.fl*2*pi * 2^(0/Scattering.Q);
            k = 1;  
            while lambdas(k) < Scattering.fh*2*pi
                lambdas(k+1) = Scattering.fl*2*pi * 2^(k/Scattering.Q);
                k = k + 1;
            end
            lambdas = flipud(lambdas(1:end-1));
        end       

        function fb = constructFB()
            fb = zeros(numel(Scattering.fc), Scattering.Nir);
            for i = 1:numel(Scattering.fc)
                fb(i, :) = Scattering.morlet(Scattering.lambdas(i));
            end
        end

        function u = filterU(x)
            x = x(:)'; %make sure its 1xN
           
            Nir = Scattering.Nir;
            xrl = fliplr(x(2:Nir));
            xrr = fliplr(x(end-Nir:end-1));

            x = [xrl, x, xrr]; %reversed boundary
            X = fft(x);
            U = X.*Scattering.Psi;
            u = ifft(U, size(U, 2), 2);
            u = abs(u(:, (Nir + (Nir-1)/2):(end- (Nir-1)/2) - 1)); %align the convolution and strip away edges
        end

        function s = filterS(x)
            u = Scattering.filterU(u);

        end

    end
end