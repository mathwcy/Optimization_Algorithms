function [x, iter, out] = gl_SGD_primal(x0, A, b, mu, opts)
    opts.n = 512;
    opts.l = 2;
    opts.m = 256;
    opts.thres = 1e-5;
    opts.maxiter = 9e3;
    function [x_grad] = grad(x, A, b, mu, opts)
        x_grad = A' * (A * x - b);
        %disp(size(x_grad));
        x_norm2 = sqrt(x(:, 1) .^ 2 + x(:, 2) .^ 2);
        %disp(size(x_norm2));
        %margin = 0.1;
        for i = 1:opts.n
            %p = sqrt(x(i, 1).^2+x(i, 2).^2);
            if x_norm2(i) > 0
                %disp(i);
                x_grad(i, :) = x_grad(i, :) + mu * x(i, :) ./ x_norm2(i);
                %if x_norm2(i) ~= norm(x(i, :))
                %    disp(iter)
                %end
            end
        end
    end

    %function [x_grad1] = grad1(x, A, b, mu, opts)
        %iteration = iteration+1;
    %    x_2normalized = zeros(opts.n, opts.l);
    %    for i=1:opts.n
    %        norm2 = norm(x(i,:));
    %        if norm2==0
    %            norm2=Inf; 
    %        end
    %        x_2normalized(i,:) = x(i,:) ./ norm2;
    %    end
    %    x_grad1 = A' * (A * x - b) + mu * x_2normalized;
    %end
    
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
    %t = 1 / norm(A' * A, 2);
    %t = 0.0005;
    f_optim = 0.580556;
    %t_ = 0.0005;
    %M = 0.01;
    %value1 = inf;
    %value2 = inf;
    %nums = 0;
    while true
        x_grad = grad(x, A, b, mu, opts);
        if isnan(x_grad)
            disp(iter);
            break
        end
        %disp(norm(x_grad));
        %value1 = value2;
        
        %if value1 > value2
         %   nums = nums + 1;
        %else
        %    nums = 0;
        %end
        %if nums > 1e4
         %   nums = 0;
          %  t = t_ / iter;
        %end
        %if value1 - value2 < value1 / 1000000
         %   t = t_ / sqrt(iter);
        %end
        %if norm(x_grad) < M
        %    disp(norm(x_grad));
        %    break
        %end
        iter = iter + 1;
        out.f_hist(iter) = value(x, A, b, mu, opts);
        out.g_hist(iter) = sum(vecnorm(x_grad'));
        f_best = min([f_best, out.f_hist(iter)]);
        out.f_hist_best(iter) = f_best;
        t = (out.f_hist(iter) - f_optim) / (norm(x_grad) .^ 2);
        %disp(value2);
        %t = t_ / sqrt(iter);
        x = x - t * x_grad;
        x(abs(x) < opts.thres) = 0;
        %if mod(iter, 1e3) == 0
            %value2 = value(x, A, b, mu, opts);
        %    disp(iter);
        %    disp(t);
        %    disp(value2);
            %disp(norm(x_grad));
            %disp(norm(t * x_grad));
        %end
        if abs(out.f_hist(iter) - f_optim) < 1e-5
            break
        end
        if iter == opts.maxiter
            break
        end
    end
    out.fval = out.f_hist(iter);
    %disp('end');
    %out = NaN;
end