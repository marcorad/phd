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
        Npsi %number of points to convolve with psi
        Nphi %number of points to colvolve with phi
        N %signal length
        flow %lowest frequency
        fhigh %highest frequency
        Npsiir %number of points in psi impulse reponse
        Nphiir %number of points in phi impulse reponse
        lambdas %center frequencies in rad/s
    end

    properties(Constant)
        sigmaRange = 5
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
            y = y(:, (Nir + (Nir-1)/2):(end- (Nir-1)/2) - 1); %align the convolution and strip away edges
        end
    end

    methods
        function obj = SFB(Q, T, fs, N, flow, fhigh)
            obj.Q = Q;
            obj.T = T;
            obj.fs = fs;
            obj.N = N;
            obj.flow = flow;
            if nargin < 6
                obj.fhigh = obj.fs/2;
            else
                obj.fhigh = fhigh;
            end            
            obj.setLambdasExponential();
            obj.setIRLengths();
            obj.constructFB();
            obj.constructInvarianceFilter();
        end

        function th = theta(obj, t)
            th = SFB.gauss(t, 2*obj.Q);%sigma_t = 2*Q, sigma_w = 1/Q
        end

        %if lamda=1, then this is for 1 rad/s
        %for now, we don't care about the minimum scale
        function [psi, n, t] = morlet(obj, lambda)
            nmax = (obj.Npsiir - 1)/2;
            n = -nmax:nmax;
            t = n / obj.fs;
            Thm1 = obj.gauss(-1, 1/obj.Q);
            Th0 = obj.gauss(0, 1/obj.Q);
            psi =  single(lambda *( exp(1j*lambda*t) - Thm1/Th0) .* obj.theta(t*lambda));
        end

        function constructInvarianceFilter(obj)
            sigma_t = obj.T/2; %in the paper, T = 1/flp -> a=T/2
            nmax = (obj.Nphiir - 1)/2;
            n = -nmax:nmax;
            t = n / obj.fs;
            obj.phi = obj.gauss(t, sigma_t);
            obj.phi = obj.phi(); %make sure its 1xNphiir
            obj.Phi = fft(gpuArray(obj.phi), obj.Nphi, 2);
            obj.fphi = (0:obj.Nphi-1)/obj.Nphi*obj.fs;
        end

        function setIRLengths(obj)
            sigma_t = 2*obj.Q/obj.lambdas(1);
            t_range = obj.sigmaRange * sigma_t;
            obj.Npsiir = floor(t_range * obj.fs)*2 + 1;   

            t_range = obj.sigmaRange * obj.T/2;
            obj.Nphiir = floor(t_range * obj.fs)*2 + 1;

            obj.Npsi = 2*obj.Npsiir + obj.N - 1;
            obj.Nphi = 2*obj.Nphiir + obj.N - 1;
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
        end

        function constructFB(obj)
            %psi filterbank
            obj.psi = zeros(numel(obj.fc), obj.Npsiir);
            for i = 1:numel(obj.fc)
                obj.psi(i, :) = obj.morlet(obj.lambdas(i));
            end
            obj.Psi = fft(gpuArray(obj.psi), obj.Npsi, 2);
%             kmid = ceil(obj.Npsi/2);
%             obj.Psi(:, kmid+1:end) = 0;
            obj.Psi = gpuArray(obj.Psi);
            obj.fpsi = (0:obj.Npsi-1)/obj.Npsi*obj.fs;
        end

        function u = filterU(obj, x)
            u = abs(obj.convrefl(x, obj.Psi, obj.Npsiir));
        end

        function s = filterS(obj, x)
            x = x(:)'; %make sure it's 1xN
            Norig = numel(x);
            Npad = obj.N - Norig;            
            x = [x; zeros(1, Npad)];
            x = gpuArray(single(x));
            u = obj.filterU(x);
            s = real(obj.convrefl(u, obj.Phi, obj.Nphiir));
            s = s(:, 1:Norig);
            %downsample
            fT = 1/obj.T/obj.fs; %normalized bandwidth
            R = floor(1/fT/2); %downsampling factor
            s = gather(s(:, 1:R:end));
        end

        function plot(obj)
            figure
            subplot(221)
            plot(0:obj.Nphiir-1, obj.phi)
            subplot(222)
            plot(obj.fphi, abs(obj.Phi)/obj.Nphi)
            xlim([0, obj.fs/2])

            subplot(223)
            hold on
            plot(0:obj.Npsiir-1, real(obj.psi))
            plot(0:obj.Npsiir-1, imag(obj.psi))
            hold off
            subplot(224)
            plot(obj.fpsi, abs(obj.Psi)/obj.Npsi)
            xlim([0, obj.fs/2])
        end

    end

end