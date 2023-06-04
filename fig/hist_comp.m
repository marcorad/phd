function [munw, muw] = hist_comp(w, nw, comp, xl, r)
yl = "Proportion (%)";
histw = 400;
histh = 300;

Nb = 20;
x1 = nw{:, comp};
x2 = w{:, comp};
if nargin < 5
    r = [min(min(x1), min(x2)), max(max(x1), max(x2))];
end
b = linspace(r(1), r(2), Nb);

fig = figure;
hold on
histogram(x1, b, "Normalization","probability")
histogram(x2, b, "Normalization","probability")
hold off
if contains(xl, "$")
    xlabel(xl,'Interpreter','latex')
else
    xlabel(xl,'Interpreter','none')
end
ylabel(yl)
ytix = get(gca, 'YTick');
set(gca, 'YTick',ytix, 'YTickLabel',ytix*100)
Tools.saveFig(fig, histw, histh, sprintf("hist_%s", comp))
munw = mean(x1(x1 >= r(1) & x1 <= r(2)));
muw = mean(x2(x2 >= r(1) & x2 <= r(2)));
end

