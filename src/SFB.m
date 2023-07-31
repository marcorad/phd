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
        function obj = SFB(Q, T, fs, N, flow, fhigh, allowDSU)
            obj.Q = Q;
            obj.T = T;
            obj.fs = fs;
            obj.N = N;
            obj.flow = flow;
            obj.fhigh = fhigh;       
            obj.alpha = 2;
            if nargin < 7
                obj.allowDSU = true;
            else
                obj.allowDSU = allowDSU;
            end
            obj.setLambdasExponential();
            obj.constructFB();
            obj.constructInvarianceFilter();

        end

        function bw = filterBW(obj)
            bw = obj.alpha*obj.Q;
        end

        function th = theta(obj, t)
            th = SFB.gauss(t, obj.filterBW());%sigma_t = Q, sigma_w = 1/Q
        end

        %if lamda=1, then this is for 1 rad/s
        %for now, we don't care about the minimum scale
        function [psi, t] = morlet(obj, lambda, t)      
            Thm1 = obj.gauss(-1, 1/obj.filterBW());
            Th0 = obj.gauss(0, 1/obj.filterBW());
%             psi =  single(lambda *( exp(1j*lambda*t) - Thm1/Th0) .* obj.theta(t*lambda));
              psi =  single(lambda * obj.theta(t*lambda).*( exp(1j*lambda*t) - Thm1/Th0));
              psi = psi - mean(psi);
        end

        function constructInvarianceFilter(obj)
            sigma_t = obj.T/2/pi; %in the paper, T = 2*pi/flp -> T = -> a=T/2

            t_range = obj.sigmaRange * obj.T;
            Nphiir = floor(t_range * obj.fs)*2 + 1;

            
            nmax = (Nphiir - 1)/2;
            n = -nmax:nmax;
            t = n / obj.fs;

            obj.phi = obj.gauss(t, sigma_t);
            obj.phi = obj.phi(); %make sure its 1xNphiir

            obj.phi = obj.phi(1,1:obj.downsampleU:end); %downsample to highest BW of filterbank
            obj.phi = obj.phi/sum(obj.phi); %normalise to averaging filter

            Nphi = 2*size(obj.phi, 2) + obj.Nu - 1;
            
            obj.Phi = fft(gpuArray(obj.phi), Nphi, 2);
            obj.fphi = (0:Nphi-1)/Nphi*obj.fs/obj.downsampleU;
            
            %critcally downsample
            fT = 1/obj.T/(obj.fs/obj.downsampleU); %normalized bandwidth
            obj.downsampleS = floor(1/fT/4); %downsampling factor
            
            obj.Ns = ceil(obj.Nu/obj.downsampleS);

        end


        function setLambdasExponential(obj)            

            obj.lambdas(1) = obj.flow*2*pi;
            k = 1;  
            while obj.lambdas(k) < obj.fhigh * 2*pi
                obj.lambdas(k+1) = obj.flow * 2*pi * 2^(k/obj.Q);
                k = k + 1;
            end
            obj.lambdas = flipud(obj.lambdas(1:end-1));
            obj.fc = obj.lambdas/2/pi;
            obj.psiBWrad = obj.lambdas/obj.filterBW();
            obj.psiBWHz = obj.psiBWrad/2/pi;

            bw = max(obj.psiBWHz)*2;
            if obj.allowDSU
%                 minDownsample = floor(obj.fs*obj.T/2);
                obj.downsampleU = max(floor(obj.fs/2/bw), 1);
            else
                obj.downsampleU = 1; %downsampling factor
            end
            obj.psiTimeSupport = 1./obj.psiBWrad;
            obj.Nu = ceil(obj.N / obj.downsampleU);
        end

        function constructFB(obj)
            sigma_t = max(obj.psiTimeSupport);
            t_range = obj.sigmaRange * sigma_t;
            Npsiir = floor(t_range * obj.fs)*2 + 1; 
            nmax = (Npsiir - 1)/2;
            n = -nmax:nmax;
            t = n / obj.fs;

            %psi filterbank
            for i = 1:numel(obj.fc)
                obj.psi(i, :) = obj.morlet(obj.lambdas(i), t)/obj.fs*2;
            end
                        
            Npsi = 2*Npsiir + obj.N - 1;
            obj.Psi = fft(gpuArray(obj.psi), Npsi, 2);
            obj.Psi = gpuArray(obj.Psi);
            obj.fpsi = (0:Npsi-1)/Npsi*obj.fs;    

        end

        function u = filterU(obj, x, filteridx)
            if nargin < 3
                u = abs(obj.convrefl(x, obj.Psi, size(obj.psi, 2)));                
            else                               
                u = abs(obj.convrefl(x, obj.Psi(filteridx, :), size(obj.psi, 2))); 
            end
            u = u(:, 1:obj.downsampleU:end);
        end

        function t = getTime(obj, s)
            t = (0:size(s,2)-1)*(obj.downsampleS*obj.downsampleU)/obj.fs;
        end

        function fs = getSSamplingFreq(obj)
            fs = obj.fs/(obj.downsampleS*obj.downsampleU);
        end


        function s = filterS(obj, x)
            %prepare x
            x = x(:)'; %make sure it's 1xN
            Norig = numel(x);
            Npad = obj.N - Norig;  
            if Npad > 0
                warning("Signal is zero-padded with %d zeros to be of length %d.", Npad, obj.N);
            end

            if Npad < 0
                error("Signal of length %d must have a maximum length of %d.", Norig, obj.N);
            end

            x = [x, zeros(1, Npad)];
            x = gpuArray(single(x));

            %get Ux and downsample to find Sx
            u = obj.filterU(x);
            s = real(obj.convrefl(u, obj.Phi, size(obj.phi, 2)));
            
            s = gather(s(:, 1:obj.downsampleS:end));
            if Npad > 0
                s = s(:, 1:floor(Norig/obj.downsampleU/obj.downsampleS));
            end
        end

        function [s, u] = filterSU(obj, x, bwrad)
            %prepare x
            x = x(:)'; %make sure it's 1xN
            Norig = numel(x);
            Npad = obj.N - Norig;  
            if Npad > 0
                warning("Signal is zero-padded with %d zeros to be of length %d.", Npad, obj.N);
            end

            if Npad < 0
                error("Signal of length %d must have a maximum length of %d.", Norig, obj.N);
            end

            x = [x, zeros(1, Npad)];
            x = gpuArray(single(x));

            if nargin < 3            
                %get Ux and downsample to find Sx
                u = obj.filterU(x);            
            else
                filteridx = obj.lambdas - obj.psiBWrad < bwrad; 
                if all(filteridx == false)
                    u = nan;
                    s = nan;
                    return;
                end
                u = obj.filterU(x, filteridx);
            end
            s = real(obj.convrefl(u, obj.Phi, size(obj.phi, 2)));            
            s = gather(s(:, 1:obj.downsampleS:end));
            u = gather(u);
        end

        function plot(obj)
            figure
            subplot(321)
            plot(0:size(obj.phi, 2)-1, obj.phi)
            subplot(322)
            plot(obj.fphi, abs(obj.Phi))
            xlim([0, obj.fs/2/obj.downsampleU])

            subplot(323)
            plot(obj.fpsi, abs(sum(obj.Psi, 1)))
            subplot(324)
            plot(obj.fpsi, abs(obj.Psi))
            xlim([0, obj.fs/2])

            subplot(3, 2, [5,6])
            plot(0:size(obj.psi, 2)-1, abs(obj.psi))
        end

        function plotS(obj, s, logp, fig)
            if nargin < 3
                logp = true;
            end
            if nargin < 4
                fig = gca;
            end
            Tools.plotTF(s, obj.fc, true, logp, obj.getTime(s), fig);
            nt = floor(size(s, 1)/10);
            yticks(fig, round(obj.fc(1:nt:end),0))
            xlabel(fig, "Time (s)");
            ylabel(fig, "Frequency (Hz)");
            ylim([min(obj.fc), max(obj.fc)])
        end

    end

end