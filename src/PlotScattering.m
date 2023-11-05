folder = "D:\Whale Data\Raw Audio Data\CaseyIslands2017";
fb = load(folder + "\scattering\filterbank.mat").fb;

s1 = fb.filterBanks(1);
s2 = fb.filterBanks(2);

figure
subplot(121)
hold on
% plot(s1.fphi, abs(s1.Phi), '--');
plot(s1.fpsi, abs(s1.Psi));
hold off
ylim([0, 2.05])
xlim([0, 140])
% title("Level 1")
xlabel("Frequency (Hz)")
ylabel("|\Psi_\lambda|", Interpreter="tex")

subplot(122)
hold on
% plot(s2.fphi, abs(s2.Phi), '--');
plot(s2.fpsi, abs(s2.Psi));
hold off
ylim([0, 2.05])
xlim([0, 12])
xlabel("Frequency (Hz)")
ylabel("|\Psi_\lambda|", Interpreter="tex")
% title("Level 2")