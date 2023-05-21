classdef SFB < handle

    % exp(-at^2) <-> sqrt(pi/a) exp(-w^2/4a)
    % let a = 1/sigma^2
    % exp(-(t/sigma)^2) <-> sqrt(pi)/sigma exp(-(w/(2/sigma))^2)

    properties
        Q %filters per octave
        T %time invariance in ms, i.e., 1/flp
        fs %sampling frequency
        fc %center frequencies
        psi %modified bandpass IRs
        phi %invariance scale filter
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

        %TODO USE conv ON GPU SINCE IT IS MUCH MUCH FASTER

        function y = convrefl(x, h, Nir)            
            xrl = fliplr(x(:, 2:Nir));
            xrr = fliplr(x(:, end-Nir:end-1));
            x = [xrl, x, xrr]; %reflected boundary
            Npad = (Nir-1)/2;
            if size(x, 1) == 1
                y = gpuArray(zeros(size(h,1), size(x,2) + 2*Npad));            
                for i = 1:size(h,1)
                    y(i, :) = conv(x, h(i,:));
                end
            else
                y = gpuArray(zeros(size(x,1), size(x,2) + 2*Npad));            
                for i = 1:size(x,1)
                    y(i, :) = conv(x(i,:), h);
                end
            end
            y = y(:, (Npad + Nir + (Nir-1)/2):(end - Npad - (Nir-1)/2) - 1); %align the convolution and strip away edges
        end
    end

    methods
        function obj = SFB(Q, T, fs, flow, fhigh)
            obj.Q = Q;
            obj.T = T;
            obj.fs = fs;
            obj.flow = flow;
            if nargin < 5
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
            obj.phi = obj.phi(:)'; %make sure its 1xNphiir
        end

        function setIRLengths(obj)
            sigma_t = 2*obj.Q/obj.lambdas(1);
            t_range = obj.sigmaRange * sigma_t;
            obj.Npsiir = floor(t_range * obj.fs)*2 + 1;
            t_range = obj.sigmaRange * obj.T/2;
            obj.Nphiir = floor(t_range * obj.fs)*2 + 1;
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
            %set energy content above nyquist to 0
            obj.psi = fft(gpuArray(obj.psi), obj.Npsiir, 2);
            kmid = ceil(obj.Npsiir/2);
%             obj.psi(:, kmid+1:end) = 0;
            obj.psi = gpuArray(ifft(obj.psi, obj.Npsiir, 2));
        end

        function u = filterU(obj, x)
            u = abs(obj.convrefl(x, obj.psi, obj.Npsiir));
        end

        function s = filterS(obj, x)
            x = x(:)'; %make sure its 1xN
            x = gpuArray(single(x));
            u = obj.filterU(x);
            s = obj.convrefl(u, obj.phi, obj.Nphiir);
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