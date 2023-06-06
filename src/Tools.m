classdef Tools

    methods(Static)
        
        function saveFig(fig, w, h, name, ctype)
            if nargin < 5
                ctype = 'vector';
            end
            f = "fig/" + name + ".pdf";
            fig.Position = [0 0 w h];
            exportgraphics(fig,f,'ContentType',ctype) 
            f = "fig/" + name + ".png";
            saveas(fig, f);
        end

        function y = blockConv(x, h, B, halfShift)
            if nargin < 4
                halfShift = false;
            end

            Lx = length(x);
            Lh = length(h);
            Nb = ceil(Lx/B); %number of blocks
            Npad = Nb*B - Lx; %padding to get to correct block size
            xr = [x;zeros(Npad,1)];
            xr = reshape(xr, [B, Nb]);
            Nfft = B + Lh - 1;
            X = fft(xr, Nfft, 1);
            H = fft(h, Nfft, 1);
            yb = ifft(X.*H);
            y = zeros(Lx + Lh + Nfft, 1);
            y(1:Nfft) = yb(:,1);
            i = B+1;
            for b = 2:Nb    
                y(i:i+Nfft-1) =  y(i:i+Nfft-1) + yb(:, b);
                i = i + B;
            end
            Lt = Lx + Lh - 1;
            if halfShift
                y = y((1:Lt)+floor(Lh/2));
            else
                y = y(1:Lt);
            end
            
        end

        function plotTF(S, F, logf, logp, t, fig)
            if nargin < 5
                t = 1:size(S,2);
            end
            if logp
                S = log(S);
            end
            if nargin < 6
                fig = gca;  
            end
            imagesc(fig, t, F, S);
            set(fig, "YDir", "Normal");
            if logf
                set(fig, "YScale", "Log");
            end
        end

        function rl = rle(b)
            b = b(:)';
            d = [diff([0,b,0])];            
            s = find(d == 1);
            e = find(d == -1);% end exclusive
            L = e-s;
            rl = array2table([s;e-1;L]', "VariableNames", ["Start", "End", "Length"]); %start and end inclusive
        end

        function mask = rle2idx(rl, L)
            mask = zeros(1, L);
            for i = 1:size(rl, 1)
                mask(rl.Start(i):rl.End(i)) = 1;
            end
        end

        function plotCorrMat(X)
            cm = [0 0 1; 0.1 0.1 0.5; 0 0 0; 0.5 0.1 0.1; 1 0 0];
            cmi = interp1([-1 -0.8 0 0.8 1], cm, -1:0.05:1);
            colormap(cmi);
            imagesc(corr(X));
            clim([-1 1]);
            colorbar;
        end

        function scatter(X, i, j, k)
            if nargin == 3
            x = X(:, i);
            y = X(:, j);
            scatter(x, y, 5, 'filled', 'MarkerEdgeAlpha',.1, 'MarkerFaceAlpha',.1);
            else
            x = X(:, i);
            y = X(:, j);
            z = X(:, k);
            scatter3(x, y, z, 5, 'filled', 'MarkerEdgeAlpha',.1, 'MarkerFaceAlpha',.1);
            end
        end

        function gscatter(X, t, i, j, k)
            if nargin == 4
            x = X(:, i);
            y = X(:, j);
            gscatter(x, y, t, 5, 'filled', 'MarkerEdgeAlpha',.1, 'MarkerFaceAlpha',.1);
            else
            x = X(:, i);
            y = X(:, j);
            z = X(:, k);
            gscatter3(x, y, z, t, 5, 'filled', 'MarkerEdgeAlpha',.1, 'MarkerFaceAlpha',.1);
            end
        end

        function gmmcontour(gmm, X, i, j)
            hold on
            Tools.scatter(X, i, j)
            lims1 = [min(X(:, i)) max(X(:,i))];
            lims2 = [min(X(:, j)) max(X(:,j))];
            x1 = linspace(lims1(1), lims1(2), 50);
            x2 = linspace(lims2(1), lims2(2), 50);
            p = zeros(numel(x1), numel(x2));
            for m = 1:numel(x1)
                for n = 1:numel(x2)
                    x = zeros(1, size(X, 2));
                    x(i) = x1(m);
                    x(j) = x2(n);
                    p(m,n) = pdf(gmm, x);
                end
            end
            p = log(p);
            contour(x1, x2, p');
            hold off
        end

        function plotPS(S, f)
        plot(f, S);
        xlabel("Frequency (Hz)");
        ylabel("PSD");
        end

        function plotContours(S, F, Cdata)
            
            ax1 = subplot(311);
            hold on
            Tools.plotTF(S, F, false, true);
            for c = Cdata
                plot(c.tmin:c.tmax, c.f0, 'k');
            end
            ylabel("Frequency (Hz)");
            ax2 = subplot(312);
            hold on
            for c = Cdata
                plot(c.tmin:c.tmax, c.bw, 'k');
            end
            ylabel("Bandwidth (Hz)");
            ax3 = subplot(313);
            hold on
            for c = Cdata
                plot(c.tmin:c.tmax, c.P, 'k');
            end
            ylabel("Power");
            hold off
            xlabel("Time bin");

            linkaxes([ax1, ax2], 'x');
            linkaxes([ax1, ax3], 'x');

        end

        function scatter2groups(x1, y1, x2, y2)
            c1 = "#0072BD";
            c2 = "#D95319";
            alpha = 0.4;
            size = 3;
            hold on
            scatter(x1, y1, size, 'filled' , "MarkerEdgeColor",c1, "MarkerFaceAlpha",alpha, "MarkerEdgeAlpha",alpha);
            scatter(x2, y2, size, 'filled' ,"MarkerEdgeColor",c2, "MarkerFaceAlpha",alpha, "MarkerEdgeAlpha",alpha);
            hold off
        end
    
    end

end