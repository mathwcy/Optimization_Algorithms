function [x, iter, out] = gl_FProxGD_primal(x0, A, b, mu, opts)
    opts.n = 512;
    opts.l = 2;
    opts.m = 256;
    opts.thres = 1e-5;
    opts.maxiter = 3e4; 

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

    out = struct();
    out.f_hist = zeros(1, opts.maxiter);
    out.f_hist_best = zeros(1, opts.maxiter);
    out.g_hist = zeros(1, opts.maxiter);
    f_best = inf;
    iter = 0;
    x = x0;
    f_optim = 0.580556;
    t_ = 0.001;

    while true
        x_grad1 = A' * (A * x - b);
        if isnan(x_grad1)
            disp(iter);
            break
        end

        iter = iter + 1;
        out.f_hist(iter) = value(x, A, b, mu, opts);
        out.g_hist(iter) = sum(vecnorm(x_grad1'));
        f_best = min([f_best, out.f_hist(iter)]);
        out.f_hist_best(iter) = f_best;

        if iter <= 1
            t = t_;
        else
            s = x - x1;
            y = x_grad1 - y1;
            sy = norm(s' * y, 2);
            %disp(size(s' * y));
            if sy < opts.thres
                t = t_;
            else
                if mod(iter, 2) == 1
                    ss = norm(s' * s, 2);
                    t = ss / sy;
                else
                    yy = norm(y' * y, 2);
                    t = sy / yy;
                end

            end
        end

        %disp(t);
        x2 = x;
        y2 = x_grad1;

        if iter > 1
            x = x + (iter - 2) / (iter + 1) *(x - x1);
        end
        x = x - t * x_grad1;
        x = Prox(x, mu, t);
        x(abs(x) < opts.thres) = 0;
        if mod(iter, 1e4) == 0
            disp(iter);
            disp(out.f_hist(iter));
        end
        x1 = x2;
        y1 = y2;
        if iter == opts.maxiter
            break
        end
        if abs(out.f_hist(iter) - f_optim) < 1e-5
            break
        end
    end
    out.fval = f_best;
end