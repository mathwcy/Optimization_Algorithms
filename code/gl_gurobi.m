function [x, iter, out] = gl_gurobi(x0, A, b, mu, opts)

    clear model;

    opts.n = 512;
    opts.l = 2;
    opts.m = 256;

    [opts.m, opts.l] = size(b);


    M = (opts.n + opts.m) * opts.l + opts.n + 2;
    
    model.modelsense = 'min';
    model.A = zeros(opts.l * opts.m + 1, M);
    model.rhs = zeros(opts.l * opts.m + 1, 1);
    model.obj = [zeros((opts.n + opts.m) * opts.l, 1); ones(opts.n, 1) * mu; 1; 0];
    

    
    
    for k = 1:opts.l
        model.A((k - 1) * opts.m + 1:k * opts.m, (k - 1) * opts.n + 1:k * opts.n) = A;
        model.A((k - 1) * opts.m + 1:k * opts.m, (k - 1) * opts.m + 1 + opts.n * opts.l:k * opts.m + opts.n * opts.l) = -eye(opts.m);
        model.rhs((k - 1) * opts.m + 1:k * opts.m) = b(:, k);
    end
    
    model.A(opts.m * opts.l + 1, M) = 1;
    model.A = sparse(model.A);
    model.rhs(opts.m * opts.l + 1) = 1;
    model.sense = '=';

    model.lb = [-inf * ones((opts.n + opts.m) * opts.l, 1); zeros(opts.n + 2, 1)];

    loc = 0:opts.n:opts.n * opts.l - 1;

    for k = 1:opts.n
        model.quadcon(k).Qrow = [k + loc, opts.n * opts.l + opts.m * opts.l + k];
        model.quadcon(k).Qcol = [k + loc, opts.n * opts.l + opts.m * opts.l + k];
        model.quadcon(k).Qval = [1, 1, -1];
        model.quadcon(k).q = sparse(zeros(M, 1));
        model.quadcon(k).rhs = 0;
        model.quadcon(k).name = 'std_cone';
    end
    
    loc = opts.n * opts.l + (1:opts.m * opts.l);
    model.quadcon(opts.n + 1).Qval = [ones(1, opts.m * opts.l), -2];
    model.quadcon(opts.n + 1).q = sparse(zeros(M, 1));
    model.quadcon(opts.n + 1).rhs = 0;
    model.quadcon(opts.n + 1).name = 'rot_cone';
    model.quadcon(opts.n + 1).Qrow = [loc, M - 1];
    model.quadcon(opts.n + 1).Qcol = [loc, M];
    

    solution = gurobi(model);

    x = zeros(opts.n, opts.l);
    for k = 0:(opts.l - 1)
        x(:, k + 1) = solution.x(k * opts.n + (1:opts.n));
    end
    
    iter = solution.baritercount;
    out = struct('fval', solution.objval);
end
