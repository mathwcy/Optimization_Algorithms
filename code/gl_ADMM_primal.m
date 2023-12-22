function [x, iter, out] = gl_ADMM_primal(x0, A, b, mu, opts)
    opts.n = 512;
    opts.l = 2;
    opts.m = 256;
    opts.thres = 1e-5; 
    opts.maxiter = 2e4; 
    opts.sigma = 0.01; 


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
    
    x = x0;
    S =  A * x0 - b;
    L = zeros(opts.m, opts.l);
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
    t = 0.0005;
    t_ = 0.0005;
    while true
        iter = iter + 1;
        %t = t_ / sqrt(iter);
        out.f_hist(iter) = value(x, A, b, mu, opts);
        %out.g_hist(iter) = sum(vecnorm(x_grad1'));
        f_best = min([f_best, out.f_hist(iter)]);
        out.f_hist_best(iter) = f_best;
        
        %Tg = T - b + opts.sigma * A * (A' * T - S - x ./ opts.sigma);
        %Sg = opts.sigma * (S - A' * T - x ./ opts.sigma);
        %T = (opts.sigma * (A * A') + eye(opts.m)) \ (b + opts.sigma * A * S - A * x);
        gx = vecnorm(x')';
        gx(gx < opts.thres) = inf;
        gx = x ./ gx;
        x = x - t * (gx + A' * (A * x - b - S + L / opts.sigma));
        x(abs(x) < opts.thres) = 0;
        %x(x < 1e-10 * opts.thres) = 0;
        S = opts.sigma * (A * x - b + L / opts.sigma) / 2;
        %S = fix(S, mu);
        %x = x + t * opts.sigma * A' * (A * x - b - S + L / opts.sigma);
        %x = Prox(x, mu, t);
        %mul_x = vecnorm(x')';
        %mul_x(mul_x < opts.thres) = 0;
        %mul_x(mul_x >= opts.thres) = 1 ./ mul_x(mul_x >= opts.thres);
        %g1 = diag(x(:, 1) .* mul_x);
        %g2 = diag(x(:, 2) .* mul_x);
        %g3 = A' * b + A' * S - A' * L / opts.sigma;
        %x1 = (mu * g1 + A' * A) \ g3(:, 1);
        %x2 = (mu * g2 + A' * A) \ g3(:, 2);
        %x = horzcat(x1, x2);
        L = L + t * opts.sigma * (A * x - b - S); 
        %disp(out.f_hist(iter));
        if mod(iter ,1e3) == 0
            %fprintf('%.6f\n', out.f_hist(iter) )
            %disp(out.f_hist(iter));
            if iter > 1000 && abs(out.f_hist_best(iter) - out.f_hist_best(iter-1000)) < ((out.f_hist_best(iter-1000) - f_optim) / 1e3)
                t = t / 2;
               
                %disp(t);
            end
            %disp(A' * T - S);
        end
        if abs(out.f_hist(iter) - f_optim) < 3e-5
            break
        end
        if iter == 5e5
            break
        end
    end
    out.fval = f_best;
end