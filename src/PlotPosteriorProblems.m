mu_n = 4;
mu_s = 3;
std_n = 0.3;
std_s = 0.7;
p_n = 0.6;
p_s = 1 - p_n;

x = linspace(1, 6, 1000);

fs = p_s * normpdf(x, mu_s, std_s);
fn = p_n * normpdf(x, mu_n, std_n);
f = fs + fn;

post = fs ./ f;

postclamp = post;
[hmin, idx] = min(post);
postclamp(idx:end) = hmin;

figure


subplot(121)
hold on
plot(x, fs);
plot(x, fn);
plot(x, fs + fn, '--');
hold off
legend("p(H_w' | C_s)", "p(H_w' | C_n)", "p(H_w')", Interpreter="tex")
xlabel("H_w'")
subplot(122)
hold on
plot(x, post);
plot(x, postclamp, '--');
hold off
legend("p(C_s | H_w')", "p_{clamp}(C_s | H_w')")
xlabel("H_w'")


