function snoise = noiseEstimate(s1, Nf, Nt)
    s1smooth = smoothdata(s1, 1, "movmedian", Nf);
    snoise = smoothdata(s1smooth, 2, "movmedian", Nt);
end