function [x, iter, out] = gl_ALM_dual(x0, A, b, mu, opts)
    opts.n = 512;
    opts.l = 2;
    opts.m = 256;
    opts.thres = 1e-5; 
    opts.maxiter = 2e4; 
    opts.sigma = 1; 
    
    function y = Prox(x, mu, t)
        
        p = mu * t;
        x_norm2 = vecnorm(x');
        y = max(x_norm2 - p, 0) ./ x_norm2;
        y = x .* y';
        %disp(size(y));

    end

    function [value] = value(x, A, b, mu, opts)
    
        value = 0.5 * (norm(A * x - b, 'fro') ^ 2) + mu * sum(reshape(norms(x, 2, 2), [], 1));

    end
    
    function [S] = fix(S, mu)
        S_norm2 = vecnorm(S')';
        S(S_norm2 > mu, :) = S(S_norm2 > mu, :) ./ S_norm2(S_norm2 > mu) .* mu;
    end
    
    T = b - A * x0;
    S = A' * T;
    S = fix(S, mu);
    x = x0;
    out = struct();
    out.f_hist = zeros(1, opts.maxiter);
    out.f_hist_best = zeros(1, opts.maxiter);
    %out.T_hist = zeros(1, opts.maxiter);
    %out.Tg_hist = zeros(1, opts.maxiter);
    %out.S_hist = zeros(1, opts.maxiter);
    %out.Sg_hist = zeros(1, opts.maxiter);
    f_best = inf;
    iter = 0;
    f_optim = 0.580556;
    t = 0.0001;
    t_ = 0.0001;

    while true
        iter = iter + 1;
        out.f_hist(iter) = value(x, A, b, mu, opts);
        %out.g_hist(iter) = sum(vecnorm(x_grad1'));
        f_best = min([f_best, out.f_hist(iter)]);
        out.f_hist_best(iter) = f_best;
        
        %Tg = T - b + opts.sigma * A * (A' * T - S - x ./ opts.sigma);
        %Sg = opts.sigma * (S - A' * T - x ./ opts.sigma);
        T = (opts.sigma * (A * A') + eye(opts.m)) \ (b + opts.sigma * A * S - A * x);
        S = A' * T + x / opts.sigma;
        S = fix(S, mu);
        x = x + opts.sigma * (A' * T - S);
        x(abs(x) < opts.thres) = 0;
        %if mod(iter ,1e3) == 0
            %disp(out.f_hist(iter));
            %disp(A' * T - S);
        %end
        if abs(out.f_hist(iter) - f_optim) < 1e-5
            break
        end
        if iter == 1e5
            break
        end
    end
    out.fval = f_best;
end