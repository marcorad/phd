function parforWaitbar(waitbarHandle,iterations)
    persistent count h N tstart
    
    if nargin == 2
        % Initialize
        
        count = 0;
        h = waitbarHandle;
        N = iterations;
        tstart = tic;
    else
        % Update the waitbar
        
        % Check whether the handle is a reference to a deleted object
        if isvalid(h)
            count = count + 1;
            dt = toc(tstart);
            left = N - count;
            tpi = dt / count;
            test = tpi*left;
            m = floor(test / 60);
            s = floor((test - m*60));
            waitbar(count / N,h, sprintf("%d of %d (%d min %d sec left)", count, N, m, s));
        end
    end
end