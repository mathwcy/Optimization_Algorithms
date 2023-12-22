function [x, iter, out] = gl_cvx_gurobi(x0, A, b, mu, opts)
    opts.n = 512;
    opts.l = 2;
    opts.m = 256;

    cvx_solver('gurobi');
    
    cvx_begin
        variable x(opts.n, opts.l)
        minimize (norm(A * x - b, 'fro') + mu * sum(reshape(norms(x, 2, 2), [], 1)))
    cvx_end
    
    
    if strcmp(cvx_status, 'Solved')
        disp('问题被成功解决！');
    elseif strcmp(cvx_status, 'Inaccurate/Solved')
        disp('求解器找到了近似解！');
    else
        disp('求解失败，可能达到了最大迭代次数或存在其他问题。');
    end
    
    x = x;
    iter = NaN;
    out = struct('fval', cvx_optval);
end