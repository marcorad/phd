classdef Discriminant < handle

    properties
        Xtrain %training samples, an Nxd matrix
        ytrain %training lables, an Nx1 or 1xN string
        classes %this classes provided from ytrain
        prior %the prior probabilities of each class
        sigma %the shared cov matrix estimate
        corr %the correlation matrix
        mu %class means
        N
        d
        C %number of classes
        w %the weights to apply the transformation
        w0 %the constant to apply to the transformation
        gamma %regularisation
        mode %mode: linear, quadratic
        delta %discard insignificant correlations
        covtypes %the cov matrix types for quadratic mode (diag, full)
    end

    methods
        function this = Discriminant(gamma, mode, delta)
            this.gamma = gamma;
            this.mode = mode;
            this.delta = delta;
        end

        function fit(this, X, y, prior, covtypes)
            this.Xtrain = X;
            this.ytrain = y;
            this.prior = prior;
            this.classes = unique(this.ytrain);
            this.N = size(X, 1);
            this.d = size(X, 2);
            this.C = numel(this.classes);
            this.mu = zeros(this.d, this.C);
            if nargin < 5
                for c = 1:this.C
                    covtypes.(this.classes(c)) = "full";
                end
            end
            this.covtypes = covtypes;
            if this.mode == "linear"
                this.sigma = zeros(this.d, this.d);
                this.corr = zeros(this.d, this.d);
            elseif this.mode == "quadratic"
                this.sigma = zeros(this.d, this.d, this.C);
                this.corr = zeros(this.d, this.d, this.C);
            end
            this.w = zeros(this.d, this.C);
            this.w0 = zeros(this.C, 1);
            for c = 1:this.C
                idx = y == this.classes(c);
                x = this.Xtrain(idx, :);
                this.mu(:, c) = median(x, 1); %median estimator of means
                if this.mode == "linear"
                    this.sigma = this.sigma + cov(x)/this.C; %equal weighting of the covariance matrices
                elseif this.mode == "quadratic"
                    t = this.covtypes.(this.classes(c));
                    if t == "full"
                    this.sigma(:, :, c) = cov(x);
                    elseif t == "diag"
                    this.sigma(:, :, c) = diag(var(x, 1, 1));
                    end
                end
            end

            this.suppressCoefficients();
            this.regularise();            

            for c = 1:this.C
                pc = this.prior.(this.classes(c));
                m = this.mu(:, c);
                if this.mode == "linear"
                    s = inv(this.sigma);
                    this.w(:, c) = (s * m);
                    this.w0(c) = -0.5 * m' * s * m + log(pc);
                end
                
            end
        end

        function yhat = predict(this, X)
            if this.mode == "linear"
            post = this.w'*X' + this.w0;            
            elseif this.mode == "quadratic"
                post = zeros(this.C, size(X, 1));
                for c = 1:this.C
                    z = X' - this.mu(:, c);
                    s = this.sigma(:, :, c);
                    pc = this.prior.(this.classes(c));
                    sinv = pinv(s);
                    logpc = log(pc);
                    logdet = 0.5*log(det(s));
                    for n = 1:size(X, 1)
                        post(c, n) = -0.5*z(:, n)'*sinv*z(:, n) + logpc - logdet;
                    end
                end
            end
            [~, yhat] = max(post,[], 1);
            yhat = this.classes(yhat);
        end


    end

    methods(Hidden)
        function suppressCoefficients(this)
            if this.mode == "linear"
                D = diag(1./sqrt(diag(this.sigma)));
                this.corr = D * this.sigma * D;
            elseif this.mode == "quadratic"
                for c = 1:this.C
                    s = this.sigma(:, :, c);
                    D = diag(1./sqrt(diag(s)));
                    this.corr(:, :, c) = D * s * D;
                end
            end
            this.sigma(abs(this.corr) < this.delta) = 0;
        end


        function regularise(this)
            if this.mode == "linear"
                this.sigma = (1- this.gamma)*this.sigma + this.gamma*diag(diag(this.sigma)); %regularisation
            elseif this.mode == "quadratic"
                for c = 1:this.C
                    s = this.sigma(:, :, c);
                    this.sigma(:, :, c) = (1- this.gamma)*s + this.gamma*diag(diag(s)); %regularisation
                end
            end
        end
    end

end