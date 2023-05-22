function GenScattering()

gpuDevice(1); %reset GPU
fb = SFB(16, 0.05, 2000, 2^18, 25);
savepath = sprintf("%s/spectrograms/filterbank.mat", POI.Path);
save(savepath, "fb", "-mat");

wb = waitbar(0);
parforWaitbar(wb, 3180);
wtq = parallel.pool.DataQueue;
afterEach(wtq,@parforWaitbar);

for ds = [1,2]
    path = sprintf("%s/%d", POI.Path, ds);
    files = dir(path);
    files = files(~[files.isdir]);
    for i = 1:numel(files)
        fname = files(i).name;
        fpath = sprintf("%s/%s", path, fname);
        xTotal = audioread(fpath);
        xTotal = 1:numel(xTotal);
        N = numel(xTotal)/16;
        for n = 1:16
            try
                x = xTotal((n-1)*N+1:(n)*N);
                s = fb.filterS(x);
                savepath = sprintf("%s/spectrograms/%d/%s_segment%d.mat", POI.Path, ds, fname(1:20), n);
                save(savepath, "s", "-mat");
                
            catch err
                disp(getReport(err,'extended'));
            end
        end
        send(wtq,[]);
    end
end
close(wb)

end