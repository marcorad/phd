function snoise = noiseEstimate(s1, fs)
    s1smooth = smoothdata(s1, 1, "movmedian", 9);
s1smooth = smoothdata(s1smooth, 2, "movmedian", floor(fs*5));
snoise = smoothdata(s1smooth, 2, "movmean", floor(fs*60));
end