classdef SFB < handle

    % exp(-0.5*at^2) <-> sqrt(pi/0.5/a) exp(-w^2/2a)
    % let a = 1/sigma^2
    % exp(-(t/sigma)^2) <-> sqrt(pi)/sigma exp(-(w/(2/sigma))^2)

    properties
        Q %filters per octave
        T %time invariance in ms, i.e., 1/flp
        fs %sampling frequency
        fc %center frequencies
        psi %unmodified bandpass IRs
        fpsi %frequency indices of Psi
        Psi %modified bank of fft(psi)
        phi %invariance scale filter
        fphi %frequency indices of Phi
        Phi %fft(phi)
        N %signal length
        flow %lowest frequency
        fhigh %highest frequency
        lambdas %center frequencies in rad/s
        psiBWrad %bandwidths in rad/s
        psiBWHz %bandwidths of all the filters in Hz
        psiTimeSupport %time support of psi fb
        downsampleU %downsampling factor of U
        downsampleS %dowmsampling factor after calcuting s*phi
        Nu %number of resulting elements in U
        Ns %number of resulting elements in S
        alpha %factor to multiply Q to get BW
        allowDSU = true;
        oversample = 1; %oversampling factor of coefficients
    end

    properties(Constant)
        sigmaRange = 5
    end

    methods(Static)
        function g = gauss(t, sigma)
            g = single(exp(-0.5*(t/sigma).^2)/sqrt(2*pi)/sigma);
        end

        function y = convrefl(x, fb, Nir)            
            xrl = fliplr(x(:,2:Nir));
            xrr = fliplr(x(:, end-Nir:end-1));
            x = [xrl, x, xrr]; %reflected boundary
            X = fft(x, size(x, 2), 2);
            Y = X.*fb;
            y = ifft(Y, size(Y, 2), 2);
            half = ceil((Nir-1)/2);
            if mod(Nir, 2) == 0
                e = half;
            else
                e = half + 1;
            end
            y = y(:, Nir + half:end - e); %align the convolution and strip away edges
        end
    end

    methods
        function this = SFB(Q, T, fs, N, flow, fhigh, allowDSU, oversample)
            this.Q = Q;
            this.T = T;
            this.fs = fs;
            this.N = N;
            this.flow = flow;
            this.fhigh = fhigh;       
            this.alpha = 1;
            
            if nargin < 7
                this.allowDSU = true;
            else
                this.allowDSU = allowDSU;
            end

            if nargin < 8
                this.oversample = 1;
            else
                this.oversample = oversample;
            end
            this.setLambdasExponential();
            this.constructFB();
            this.constructInvarianceFilter();
            
        end

        function bw = filterBW(this)
            bw = this.alpha*this.Q;
        end

        function th = theta(this, t)
            th = SFB.gauss(t, this.filterBW());%sigma_t = Q, sigma_w = 1/Q
        end

        %if lamda=1, then this is for 1 rad/s
        %for now, we don't care about the minimum scale
        function [psi, t] = morlet(this, lambda, t)      
            Thm1 = this.gauss(-1, 1/this.filterBW());
            Th0 = this.gauss(0, 1/this.filterBW());
%             psi =  single(lambda *( exp(1j*lambda*t) - Thm1/Th0) .* this.theta(t*lambda));
              psi =  single(lambda * this.theta(t*lambda).*( exp(1j*lambda*t) - Thm1/Th0));
              psi = psi - mean(psi);
        end

        function constructInvarianceFilter(this)
            sigma_t = this.T/2/pi; %in the paper, T = 2*pi/flp -> T = -> a=T/2

            t_range = this.sigmaRange * this.T;
            Nphiir = floor(t_range * this.fs)*2 + 1;

            
            nmax = (Nphiir - 1)/2;
            n = -nmax:nmax;
            t = n / this.fs;

            this.phi = this.gauss(t, sigma_t);
            this.phi = this.phi(); %make sure its 1xNphiir

            this.phi = this.phi(1,1:this.downsampleU:end); %downsample to highest BW of filterbank
            this.phi = this.phi/sum(this.phi); %normalise to averaging filter

            Nphi = 2*size(this.phi, 2) + this.Nu - 1;
            
            this.Phi = fft(gpuArray(this.phi), Nphi, 2);
            this.fphi = (0:Nphi-1)/Nphi*this.fs/this.downsampleU;
            
            %critcally downsample
            fT = 1/this.T/(this.fs/this.downsampleU); %normalized bandwidth
            this.downsampleS = floor(1/fT/4/this.oversample); %downsampling factor
            
            this.Ns = ceil(this.Nu/this.downsampleS);

        end


        function setLambdasExponential(this)            

            this.lambdas(1) = this.flow*2*pi;
            %exponential
            k = 1;  
            while this.lambdas(k) < this.fhigh * 2*pi
                this.lambdas(k+1) = this.flow * 2*pi * 2^(k/this.Q);
                k = k + 1;
            end

            this.lambdas = flipud(this.lambdas(1:end-1));
            this.fc = this.lambdas/2/pi;
            this.psiBWrad = this.lambdas/this.filterBW();
            this.psiBWHz = this.psiBWrad/2/pi;

            bw = max(this.psiBWHz)*2;
            if this.allowDSU
%                 minDownsample = floor(this.fs*this.T/2);
                this.downsampleU = max(floor(this.fs/2/bw), 1);
            else
                this.downsampleU = 1; %downsampling factor
            end
            this.psiTimeSupport = 1./this.psiBWrad;
            this.Nu = ceil(this.N / this.downsampleU);
        end

        function constructFB(this)
            sigma_t = max(this.psiTimeSupport);
            t_range = this.sigmaRange * sigma_t;
            Npsiir = floor(t_range * this.fs)*2 + 1; 
            nmax = (Npsiir - 1)/2;
            n = -nmax:nmax;
            t = n / this.fs;

            %psi filterbank
            for i = 1:numel(this.fc)
                this.psi(i, :) = this.morlet(this.lambdas(i), t)/this.fs*2;
            end
                        
            Npsi = 2*Npsiir + this.N - 1;
            this.Psi = fft(gpuArray(this.psi), Npsi, 2);
            this.Psi = gpuArray(this.Psi);
            this.fpsi = (0:Npsi-1)/Npsi*this.fs;    

        end

        function u = filterU(this, x, filteridx)
            if nargin < 3
                u = abs(this.convrefl(x, this.Psi, size(this.psi, 2)));                
            else                               
                u = abs(this.convrefl(x, this.Psi(filteridx, :), size(this.psi, 2))); 
            end
            u = u(:, 1:this.downsampleU:end);
        end

        function t = getTime(this, s)
            t = (0:size(s,2)-1)*(this.downsampleS*this.downsampleU)/this.fs;
        end

        function fs = getSSamplingFreq(this)
            fs = this.fs/(this.downsampleS*this.downsampleU);
        end


        function s = filterS(this, x)
            %prepare x
            x = x(:)'; %make sure it's 1xN
            Norig = numel(x);
            Npad = this.N - Norig;  
            if Npad > 0
                warning("Signal is zero-padded with %d zeros to be of length %d.", Npad, this.N);
            end

            if Npad < 0
                error("Signal of length %d must have a maximum length of %d.", Norig, this.N);
            end

            x = [x, zeros(1, Npad)];
            x = gpuArray(single(x));

            %get Ux and downsample to find Sx
            u = this.filterU(x);
            s = real(this.convrefl(u, this.Phi, size(this.phi, 2)));
            
            s = gather(s(:, 1:this.downsampleS:end));
            if Npad > 0
                s = s(:, 1:floor(Norig/this.downsampleU/this.downsampleS));
            end
        end

        function [s, u] = filterSU(this, x, bwrad)
            %prepare x
            x = x(:)'; %make sure it's 1xN
            Norig = numel(x);
            Npad = this.N - Norig;  
            if Npad > 0
                warning("Signal is zero-padded with %d zeros to be of length %d.", Npad, this.N);
            end

            if Npad < 0
                error("Signal of length %d must have a maximum length of %d.", Norig, this.N);
            end

            x = [x, zeros(1, Npad)];
            x = gpuArray(single(x));

            if nargin < 3            
                %get Ux and downsample to find Sx
                u = this.filterU(x);            
            else
                filteridx = this.lambdas - this.psiBWrad < bwrad; 
                if all(filteridx == false)
                    u = nan;
                    s = nan;
                    return;
                end
                u = this.filterU(x, filteridx);
            end
            s = real(this.convrefl(u, this.Phi, size(this.phi, 2)));            
            s = gather(s(:, 1:this.downsampleS:end));
            u = gather(u);
        end

        function plot(this)
            figure
            subplot(321)
            plot(0:size(this.phi, 2)-1, this.phi)
            subplot(322)
            plot(this.fphi, abs(this.Phi))
            xlim([0, this.fs/2/this.downsampleU])

            subplot(323)
            plot(this.fpsi, abs(sum(this.Psi, 1)))
            subplot(324)
            plot(this.fpsi, abs(this.Psi))
            xlim([0, this.fs/2])

            subplot(3, 2, [5,6])
            plot(0:size(this.psi, 2)-1, abs(this.psi))
        end

        function plotS(this, s, logp, fig)
            if nargin < 3
                logp = true;
            end
            if nargin < 4
                fig = gca;
            end
            Tools.plotTF(s, this.fc, true, logp, this.getTime(s), fig);
            nt = floor(size(s, 1)/10);
            yticks(fig, round(this.fc(1:nt:end),0))
            xlabel(fig, "Time (s)");
            ylabel(fig, "Frequency (Hz)");
            ylim([min(this.fc), max(this.fc)])
        end

    end

end