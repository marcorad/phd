classdef SFB < handle

    % exp(-at^2) <-> sqrt(pi/a) exp(-w^2/4a)
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
        psiBWHz %bandwidths of all the filters in Hz
        psiTimeSupport %time support of psi fb
        downsampleS %downsampling factor of U
        downsamplePhi %dowmsampling factor after calcuting s*phi
    end

    properties(Constant)
        sigmaRange = 4
    end

    methods(Static)
        function g = gauss(t, sigma)
            g = single(exp(-(t/sigma).^2)/sqrt(2*pi)/sigma);
        end

        function y = convrefl(x, fb, Nir)            
            xrl = fliplr(x(:,2:Nir));
            xrr = fliplr(x(:, end-Nir:end-1));
            x = [xrl, x, xrr]; %reflected boundary
            X = fft(x, size(x, 2), 2);
            Y = X.*fb;
            y = ifft(Y, size(Y, 2), 2);
            half = ceil((Nir-1)/2);
            y = y(:, Nir + half:end - half - 1); %align the convolution and strip away edges
        end
    end

    methods
        function obj = SFB(Q, T, fs, N, flow, fhigh)
            obj.Q = Q;
            obj.T = T;
            obj.fs = fs;
            obj.N = N;
            obj.flow = flow;
            obj.fhigh = fhigh;        
            obj.setLambdasExponential();
            obj.constructFB();
            obj.constructInvarianceFilter();
        end

        function th = theta(obj, t)
            th = SFB.gauss(t, 4*obj.Q);%sigma_t = 4*Q, sigma_w = 2/Q
        end

        %if lamda=1, then this is for 1 rad/s
        %for now, we don't care about the minimum scale
        function [psi, t] = morlet(obj, lambda, t)      
            Thm1 = obj.gauss(-1, 2/obj.Q);
            Th0 = obj.gauss(0, 2/obj.Q);
            psi =  single(lambda *( exp(1j*lambda*t) - Thm1/Th0) .* obj.theta(t*lambda));
        end

        function constructInvarianceFilter(obj)
            t_range = obj.sigmaRange * obj.T/2;
            Nphiir = floor(t_range * obj.fs)*2 + 1;

            sigma_t = obj.T/2; %in the paper, T = 1/flp -> a=T/2
            nmax = (Nphiir - 1)/2;
            n = -nmax:nmax;
            t = n / obj.fs;

            obj.phi = obj.gauss(t, sigma_t);
            obj.phi = obj.phi(); %make sure its 1xNphiir

            obj.phi = obj.phi(1,1:obj.downsampleS:end); %downsample to highest BW of filterbank
            obj.phi = obj.phi/sum(obj.phi); %normalise to averaging filter

            Nphi = 2*size(obj.phi, 2) + ceil(obj.N/obj.downsampleS) - 1;
            
            obj.Phi = fft(gpuArray(obj.phi), Nphi, 2);
            obj.fphi = (0:Nphi-1)/Nphi*obj.fs/obj.downsampleS;
            
            %critcally downsample
            fT = 1/obj.T/(obj.fs/obj.downsampleS); %normalized bandwidth
            obj.downsamplePhi = floor(1/fT/2); %downsampling factor

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
            obj.psiBWHz = obj.fc/obj.Q;

            bw = max(obj.psiBWHz);
            obj.downsampleS = floor(obj.fs/2/bw);
            obj.psiTimeSupport = 4 * obj.Q ./ obj.fc/2/pi;
        end

        function constructFB(obj)
            sigma_t = 4*obj.Q/obj.lambdas(1);
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

        function u = filterU(obj, x)
            u = abs(obj.convrefl(x, obj.Psi, size(obj.psi, 2)));
        end

        function t = getTime(obj, s)
            t = (0:size(s,2)-1)*(obj.downsamplePhi*obj.downsampleS)/obj.fs;
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
            u = u(:, 1:obj.downsampleS:end);
            s = real(obj.convrefl(u, obj.Phi, size(obj.phi, 2)));
%             s = s(:, 1:floor(Norig/obj.downsampleS));
            
            s = gather(s(:, 1:obj.downsamplePhi:end));            
        end

        function plot(obj)
            figure
            subplot(321)
            plot(0:size(obj.phi, 2)-1, obj.phi)
            subplot(322)
            plot(obj.fphi, abs(obj.Phi))
            xlim([0, obj.fs/2/obj.downsampleS])

            subplot(323)
            plot(obj.fpsi, abs(sum(obj.Psi, 1)))
            subplot(324)
            plot(obj.fpsi, abs(obj.Psi))
            xlim([0, obj.fs/2])

            subplot(3, 2, [5,6])
            plot(0:size(obj.psi, 2)-1, abs(obj.psi))
        end

    end

end