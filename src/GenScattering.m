function GenScattering()

Path = "D:\Whale Data\POIData";

gpuDevice(1); %reset GPU
fb = SFB(16, 0.05, 2000, 2^18, 25, 1000);
savepath = sprintf("%s/spectrograms/filterbank.mat", Path);
save(savepath, "fb", "-mat");

for ds = [1]
    dl = DataLoader(ds);
    dl.startAudioWaitbar();
    while ~dl.isAudioComplete()
    [segments, info] = dl.nextAudio();
    for n = 1:16
        try
            x = segments{n};
            s = fb.filterS(x);
            savepath = sprintf("%s/%d/%s_%d.mat", dl.spectrogrampath, ds, extractBefore(info.name, "."), n);
            save(savepath, "s", "-mat");
        catch err
            disp(getReport(err,'extended'));
        end
    end
    end
end



end