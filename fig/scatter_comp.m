function scatter_comp(w, nw, compx, compy, xl, yl, px, py)
scatw = 270;
scath = 200;

x1 = nw{:, compx};
x2 = w{:, compx};
y1 = nw{:, compy};
y2 = w{:, compy};

rx(1) = min(prctile(x1, px), prctile(x2, px));
rx(2) = max(prctile(x1, 100-px), prctile(x2, 100-px));
ry(1) = min(prctile(y1, py), prctile(y2, py));
ry(2) = max(prctile(y1, 100-py), prctile(y2, 100-py));

fig = figure;
c1 = "#0072BD";
c2 = "#D95319";
alpha = 0.05;
size = 3;
hold on
scatter(x1, y1, size, 'filled' , "MarkerEdgeColor",c1, "MarkerFaceAlpha",alpha, "MarkerEdgeAlpha",alpha);
scatter(x2, y2, size, 'filled' ,"MarkerEdgeColor",c2, "MarkerFaceAlpha",alpha, "MarkerEdgeAlpha",alpha);
hold off
if contains(xl, "$")
    xlabel(xl,'Interpreter','latex')
else
    xlabel(xl,'Interpreter','tex')
end
if contains(yl, "$")
    ylabel(yl,'Interpreter','latex')
else
    ylabel(yl,'Interpreter','tex')
end
xlim(rx)
ylim(ry)
Tools.saveFig(fig, scatw, scath, sprintf("scat_%s_%s", compx, compy))
end