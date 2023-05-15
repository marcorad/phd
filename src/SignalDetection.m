classdef SignalDetection
    % Contains all the methods to perform k-means based soft signal
    % detection using spectral entropy.

    properties(Constant)
        MaxGMMIter = 200;
        PosteriorThresh = 0.5;
    end

    methods(Static)
        function H = spectralEntropy(S)
            % Generate spectral entropy measure from TF decomposition S.
            % Requires S to be KxT matrix, where K is the frequency bins
            % and T is the number of time steps.
            P = S ./ sum(S,1); %treat as pdf
            Plog = log(P); %log pdf
            H = -sum(P.*Plog, 1);
        end

        function H = spectralEntropyMF(S, M)
            % Generate spectral entropy measure from TF decomposition S.
            % Requires S to be KxT matrix, where K is the frequency bins
            % and T is the number of time steps. Smooth it with median
            % filtering.
            P = S ./ sum(S,1); %treat as pdf
            Plog = log(P); %log pdf
            H = -sum(P.*Plog, 1);
            H = smoothdata(H, 2, "movmedian", M);
        end

        function [H, probs, mu_ns, mu_s, beta] = softSignalDetection(S, p, M)
            % Get the soft signal classification probabilities and entropy
            % class means, median-filtered with an ODD window of length M.
            H = SignalDetection.spectralEntropy(S);
            H = smoothdata(H, 2, "movmedian", M);
            [mu_ns, mu_s, beta] = SignalDetection.entropyKMeans(H);
            g = SignalDetection.kmProbGain(mu_s, mu_ns, p);
            probs = SignalDetection.entropyProb(H, mu_s, mu_ns, g);
        end

        function [H, probs, mu_ns, mu_s, sigma_ns, sigma_s, comp_ns, comp_s, converged, n_iter, LL, mon, fmin, fmax, Ht, contam] = softSignalDetectionGMM(S, M)
            % Get the soft signal classification probabilities and entropy
            % class means, median-filtered with an ODD window of length M.
            H = SignalDetection.spectralEntropy(S);
            H = smoothdata(H, 2, "movmedian", M);
            [probs, mu_s, mu_ns, sigma_s, sigma_ns, comp_s, comp_ns, n_iter, converged, LL, mon, fmin, fmax, Ht, contam] = SignalDetection.gmmSE(H);
        end

        function [H, probs, mu_ns, mu_s, beta, delta, converged] = softSignalDetectionIt(S, p, M, iter)
            % Get the soft signal classification probabilities and entropy
            % class means, median-filtered with an ODD window of length M.
            H = SignalDetection.spectralEntropy(S);
            H = smoothdata(H, 2, "movmedian", M);
            [mu_ns, mu_s, beta, delta, converged] = SignalDetection.entropyKMeansIt(H, iter);
            g = SignalDetection.kmProbGain(mu_s, mu_ns, p);
            probs = SignalDetection.entropyProb(H, mu_s, mu_ns, g);
        end

        function [mu_s, delta, converged] = entropyMeansIt(S, M, iter)
            % Get the entropy class signal mean, the delta from the noise
            % mean and whether the K-means algorithm converged
            H = SignalDetection.spectralEntropy(S);
            H = smoothdata(H, 2, "movmedian", M);
            [~, mu_s, ~, delta, converged] = SignalDetection.entropyKMeansIt(H, iter);
        end

        function [mu_ns, mu_s, beta] = entropyKMeans(E)
            % Perform k-means to determine the entropy class means.
            M = [max(E); min(E)];
            Mprev = M+100;
            epsilon = 1e-7;
            while max(abs(M - Mprev)) > epsilon
                D = (E - M).^2; %distances
                [~,S] = min(D,[],1); %assignment
                Mprev = M;
                M(1) = mean(E(S==1));
                M(2) = mean(E(S==2));
            end
            beta = M(1) + (M(2) - M(1))/2;
            mu_ns = M(1);
            mu_s = M(2);
        end

        function [mu_ns, mu_s, beta, delta, converged] = entropyKMeansIt(E, iter)
            % Perform k-means to determine the entropy class means.
            M = [max(E); min(E)];
            Mprev = M+100;
            epsilon = 1e-7;
            converged = false;
            for i=1:iter
                if max(abs(Mprev-M)) < epsilon
                    converged = true;
                    break;
                end
                D = (E - M).^2; %distances
                [~,S] = min(D,[],1); %assignment
                Mprev = M;
                M(1) = mean(E(S==1));
                M(2) = mean(E(S==2));
            end
            beta = M(1) + (M(2) - M(1))/2;
            mu_ns = M(1);
            mu_s = M(2);
            delta = mu_ns-mu_s;
        end

        function g = kmProbGain(mu_s, mu_ns, p)
            % Get the gain required to correspond to a signal probability p
            % given known class means.
            beta = mu_s + (mu_ns - mu_s)/2;
            g = (mu_ns - mu_s)/(mu_s - beta)/2*log(1/p-1);
        end

        function [P, beta] = entropyProb(H, mu_s, mu_ns, g)
            % Calculate the signal probabilities given the entropy measure,
            % entropy class means and gain hyper-parameter.
            beta = mu_s + (mu_ns - mu_s)/2;
            Hs = 2*g*(H - beta)/(mu_ns - mu_s);
            P = 1./(1 + exp(Hs));
        end


        function [mon, lim, dir] = isGMMMonotonic(Hmax, Hmin, mu_s, mu_n, sigma_s, sigma_n) 
            lim = (sigma_s^2*mu_n - sigma_n^2*mu_s)/(sigma_s^2 - sigma_n^2);
            if sigma_s > sigma_n
                dir = "max";
                mon = Hmax < lim;
            else
                dir = "min";
                mon = Hmin > lim;
            end
        end

        function [f, xt] = gmmPosterior(x, mu_s, mu_n, sigma_s, sigma_n, pi_s, pi_n, lim, dir)
            if nargin == 7
                [~, lim, dir] = SignalDetection.isGMMMonotonic(max(x), min(x), mu_s, mu_n, sigma_s, sigma_n);
            end
            ps = normpdf(x, mu_s, sigma_s) * pi_s;
            pn = normpdf(x, mu_n, sigma_n) * pi_n;
            f = ps./(pn+ps);
            fmax = max(f);
            fmin = min(f);
            if dir == "min"                
                f(f < lim) = fmax;
            elseif dir == "max"                
                f(f > lim) = fmin;
            end 
            xt = x(find(f <= SignalDetection.PosteriorThresh, 1, "first"));
        end

        function [probs, mu_s, mu_n, sigma_s, sigma_n, pi_s, pi_n, n_iter, converged, LL, mon, fmin, fmax, Ht, contam] = gmmSE(H)
            mu_n = max(H);
            mu_s = min(H);
            sigma = std(H);
            sigma_s = 0.5*sigma;
            sigma_n = sigma_s;
            LL = inf;
            pi_s = 0.5;
            pi_n = 0.5;
            N = numel(H);
            n_iter = 0;
            while n_iter < SignalDetection.MaxGMMIter                
                p_s = pi_s * normpdf(H, mu_s, sigma_s);
                p_ns = pi_n * normpdf(H, mu_n, sigma_n);
                LL_prev = LL;
                LL = sum(log(p_s + p_ns));
                if abs(LL - LL_prev) < 1e-6
                    break;
                end
                gamma_s = p_s ./ (p_s + p_ns);
                gamma_ns = 1 - gamma_s;
                N_s = sum(gamma_s);
                N_ns = N - N_s;
                mu_s = sum(gamma_s .* H) / N_s;
                mu_n = sum(gamma_ns .* H) / N_ns;
                sigma_s = sqrt(sum(gamma_s .* (H - mu_s).*(H - mu_s))/N_s);
                sigma_n = sqrt(sum(gamma_ns .*(H - mu_n).*(H - mu_n))/N_ns);
                pi_s = N_s/N;
                pi_n = 1 - pi_s;
                n_iter = n_iter + 1;
            end
            converged = n_iter < SignalDetection.MaxGMMIter;
            probs = gamma_s;
            if mu_s > mu_n
                [mu_n, mu_s] = swap(mu_n, mu_s);
                [sigma_n, sigma_s] = swap(sigma_n, sigma_s);
                [pi_n, pi_s] = swap(pi_n, pi_s);
                probs = gamma_ns;
            end

            %Take care of the monotonicity problems
            Hmin = min(H);
            Hmax = max(H);
            [mon, lim, dir] = SignalDetection.isGMMMonotonic(Hmax, Hmin, mu_s, mu_n, sigma_s, sigma_n);
            x = linspace(Hmin, Hmax, 1000);
            [f, Ht] = SignalDetection.gmmPosterior(x, mu_s, mu_n, sigma_s, sigma_n, pi_s, pi_n, lim, dir);
            fmax = max(f);
            fmin = min(f);
            if dir == "min"                
                probs(H < lim) = fmax;
            elseif dir == "max"                
                probs(H > lim) = fmin;
            end 

            contam = pi_n*normcdf(Ht, mu_n, sigma_n)/(pi_s*normcdf(Ht, mu_s, sigma_s)) ;


        end
    end
end