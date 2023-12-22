function [x, iter, out] = gl_mosek(x0, A, b, mu, opts)
    
    clear prob;
    [~,solution]=mosekopt('symbcon');

    opts.n = 512;
    opts.l = 2;
    opts.m = 256;
    
    prob.blc = zeros(opts.m * opts.l, 1);
    prob.c = [zeros((opts.n + opts.m) * opts.l, 1); mu * ones(opts.n, 1); 1; 0];

    
    Mar = (opts.n + opts.m) * opts.l + opts.n + 2;
    prob.a = zeros(opts.m * opts.l, Mar);
    

    for k = 1:opts.l
        prob.blc((k - 1) * opts.m + 1:k * opts.m) = b(:, k);
        prob.a((k - 1) * opts.m + 1:k * opts.m, opts.n * (k - 1) + 1:opts.n * k) = A;
        prob.a((k - 1) * opts.m + 1:k * opts.m, opts.m * (k - 1) + 1 + opts.n * opts.l:k * opts.m + opts.n * opts.l) = -eye(opts.m);
    end
    
    prob.cones.type = [solution.symbcon.MSK_CT_QUAD * ones(1, opts.n), solution.symbcon.MSK_CT_RQUAD];
    prob.cones.sub = [];
    
    prob.cones.subptr = 1;

    prob.buc = prob.blc;

    prob.blx = [-inf * ones((opts.n + opts.m) * opts.l, 1); zeros(opts.n + 2, 1)];
    prob.blx(Mar) = 1;

    prob.bux = inf * ones(Mar, 1);
    prob.bux(Mar) = 1;
    
    for k = 1:opts.n
        prob.cones.sub = [prob.cones.sub, (opts.n + opts.m) * opts.l + k];
        for i = 1:opts.l
            prob.cones.sub = [prob.cones.sub, (i - 1) * opts.n + k];
        end
        prob.cones.subptr = [prob.cones.subptr, 1 + k * (opts.l + 1)];
    end
    
    prob.cones.sub = [prob.cones.sub, Mar - 1, Mar, (1 + opts.l * opts.n):opts.l * (opts.m + opts.n)];

    [~, solution] = mosekopt('minimize', prob);

    x = zeros(opts.n, opts.l);
    for i = 1:opts.l
        x(:, i) = solution.sol.itr.xx((i - 1) * opts.n + (1:opts.n));
    end
    
    out = struct('fval', solution.sol.itr.pobjval);
    iter = NaN;
end
