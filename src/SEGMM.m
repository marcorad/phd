classdef SEGMM < handle
    properties(Constant)
        MaxGMMIter = 200;
        PosteriorThresh = 0.5;
    end

    properties
        M %median filtering window size
        H %filtered spectral entropy
        probs %modified signal probability
        mu_n %noise mean
        mu_s %signal mean
        sigma_n %noise std
        sigma_s %signal std
        converged %GMM convergence
        n_iter %iterations for convergence
        LL %model log-likelihood
        mon %is the posterior monotonic
        fmin %minimum value of the posterior
        fmax %maximum value of the posterior
        Ht %thresholded to set entropy for binary detection
        contam %amount of class contaminamation
        Hmin %minimum entropy
        Hmax %maximum entropy
        dir %direction in which the posterior is not monotonic
        lim %the value at which monotonicity breaks
        pi_n %proportion of noise
        pi_s %proportion of signal
        f %the posterior function
        fmod %the modified posterior function
        x %values to evaluate the posterior function
        pdf %PDF evaluated at x
        beta %beta for k-means
        g %gain for k-means
    end

    methods
        function obj = SEGMM(M)
            obj.M = M;
        end

        %get the GMM-related values in a structure for easy analysis and
        %reduced memory usage
        function stats = getStatistics(obj)
            stats.mu_n = obj.mu_n;
            stats.mu_s = obj.mu_s;
            stats.sigma_n = obj.sigma_n;
            stats.sigma_s = obj.sigma_s;
            stats.converged = obj.converged;
            stats.n_iter = obj.n_iter;
            stats.LL = obj.LL;
            stats.mon = obj.mon;
            stats.contam = obj.contam;
            stats.pi_n = obj.pi_n;
            stats.pi_s = obj.pi_s;
            stats.delta = obj.mu_n - obj.mu_s;
        end

        function calculateSE(obj, S)
            % Generate spectral entropy measure from TF decomposition S.
            % Requires S to be KxT matrix, where K is the frequency bins
            % and T is the number of time steps.
            P = S ./ sum(S,1); %treat as pdf
            Plog = log(P); %log pdf
            obj.H = -sum(P.*Plog, 1);
        end

        function detect(obj, S)
            % Get the soft signal classification probabilities and entropy
            % class means, median-filtered with an ODD window of length obj.M.
            obj.calculateSE(S);
            obj.H = smoothdata(obj.H, 2, "movmedian", obj.M);
            obj.Hmin = min(obj.H);
            obj.Hmax = max(obj.H);
            obj.fitGMM();
        end

        function fitKMeans(obj)
            % Perform k-means to determine the entropy class means.
            mus = [max(obj.H); min(obj.H)];
            Mprev = mus+100;
            epsilon = 1e-7;
            obj.converged = false;
            for i=1:SEGMM.MaxGMMIter
                if max(abs(Mprev-mus)) < epsilon
                    obj.converged = true;
                    break;
                end
                D = (E - mus).^2; %distances
                [~,S] = min(D,[],1); %assignment
                Mprev = mus;
                mus(1) = mean(E(S==1));
                mus(2) = mean(E(S==2));
            end
            obj.beta = mus(1) + (mus(2) - mus(1))/2;
            obj.mu_n = mus(1);
            obj.mu_s = mus(2);
            p = 0.99;
            obj.g = (obj.mu_n - obj.mu_s)/(obj.mu_s - obj.beta)/2*log(1/p-1);
            Hs = 2*obj.g*(obj.H - obj.beta)/(obj.mu_n - obj.mu_s);
            obj.probs = 1./(1 + exp(Hs));
        end

        function fitGMM(obj)
            obj.mu_n = max(obj.H);
            obj.mu_s = min(obj.H);
            sigma = std(obj.H);
            obj.sigma_s = 0.5*sigma;
            obj.sigma_n = obj.sigma_s;
            obj.LL = inf;
            obj.pi_s = 0.5;
            obj.pi_n = 0.5;
            N = numel(obj.H);
            obj.n_iter = 0;
            while obj.n_iter < SignalDetection.MaxGMMIter
                p_s = obj.pi_s * normpdf(obj.H, obj.mu_s, obj.sigma_s);
                p_n = obj.pi_n * normpdf(obj.H, obj.mu_n, obj.sigma_n);
                LL_prev = obj.LL;
                obj.LL = sum(log(p_s + p_n));
                if abs(obj.LL - LL_prev) < 1e-6
                    break;
                end
                gamma_s = p_s ./ (p_s + p_n);
                gamma_n = 1 - gamma_s;
                N_s = sum(gamma_s);
                N_n = N - N_s;
                obj.mu_s = sum(gamma_s .* obj.H) / N_s;
                obj.mu_n = sum(gamma_n .* obj.H) / N_n;
                obj.sigma_s = sqrt(sum(gamma_s .* (obj.H - obj.mu_s).^2)/N_s);
                obj.sigma_n = sqrt(sum(gamma_n .*(obj.H - obj.mu_n).^2)/N_n);
                obj.pi_s = N_s/N;
                obj.pi_n = 1 - obj.pi_s;
                obj.n_iter = obj.n_iter + 1;
            end
            obj.converged = obj.n_iter < SEGMM.MaxGMMIter;
            obj.probs = gamma_s;
            if obj.mu_s > obj.mu_n
                [obj.mu_n, obj.mu_s] = swap(obj.mu_n, obj.mu_s);
                [obj.sigma_n, obj.sigma_s] = swap(obj.sigma_n, obj.sigma_s);
                [obj.pi_n, obj.pi_s] = swap(obj.pi_n, obj.pi_s);
                obj.probs = gamma_n;
            end

            %Take care of the monotonicity problems
            obj.determineMonotonicity();
            obj.modifiedPosterior()

            if ~obj.mon
                if obj.dir == "min"
                    obj.probs(obj.H < obj.lim) = obj.fmax;
                elseif obj.dir == "max"
                    obj.probs(obj.H > obj.lim) = obj.fmin;
                end
            end

%             minp = min(obj.probs);
%             maxp = max(obj.probs);
%             obj.probs = (obj.probs - minp)/(maxp - minp);
            
            obj.contam = obj.pi_n*normcdf(obj.Ht, obj.mu_n, obj.sigma_n)/(obj.pi_s*normcdf(obj.Ht, obj.mu_s, obj.sigma_s));           
            
            obj.calculatePDF()
        end

        function modifiedPosterior(obj)
            obj.x = linspace(obj.Hmin, obj.Hmax, 1000);
            ps = normpdf(obj.x, obj.mu_s, obj.sigma_s) * obj.pi_s;
            pn = normpdf(obj.x, obj.mu_n, obj.sigma_n) * obj.pi_n;
            obj.f = ps./(pn+ps);
            obj.fmod = obj.f;
            obj.fmax = max(obj.f);
            obj.fmin = min(obj.f);
            if ~obj.mon
                if obj.dir == "min"
                    obj.fmod(obj.x < obj.lim) = obj.fmax;
                elseif obj.dir == "max"
                    obj.fmod(obj.x > obj.lim) = obj.fmin;
                end
            end
            obj.Ht = obj.x(find(obj.f <= SEGMM.PosteriorThresh, 1, "first"));
        end

        function determineMonotonicity(obj)
            obj.lim = (obj.sigma_s^2*obj.mu_n - obj.sigma_n^2*obj.mu_s)/(obj.sigma_s^2 - obj.sigma_n^2);
            if obj.sigma_s > obj.sigma_n
                obj.dir = "max";
                obj.mon = obj.Hmax < obj.lim;
            else
                obj.dir = "min";
                obj.mon = obj.Hmin > obj.lim;
            end
        end

        function calculatePDF(obj)
            obj.pdf(1, :) = normpdf(obj.x, obj.mu_s, obj.sigma_s);
            obj.pdf(2, :) = normpdf(obj.x, obj.mu_n, obj.sigma_n);
        end

    end

end