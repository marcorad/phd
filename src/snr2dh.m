function dh = snr2dh(snr, n)
    dh = log(n) - 1/(snr+1) * ((1-1/n)*log(n*(snr+1)) - (snr + 1/n)*log((snr+1/n)/(snr+1)));
end