%% Solve
Nx = 220;
Nt = 220;
xs = linspace(0, 1, Nx);
ts = linspace(0, 1, Nt);
global params;
% params = [tau/L^2], Tref=200
params = [0.15/20];

m = 0;
u_sol_s = pdepe(m,@pdefun,@pdeic,@pdebc,xs,ts);

csvwrite('heat_220.csv', u_sol_s);

%% Plots 
surf(xs, ts, u_sol_s);
title("Non-dimesional Heat Equation")
xlabel('Dimensionless Distance (x*)')
ylabel('Dimensionless Time (t*)')

%% 
function [c, f, s] = pdefun(x, t, u, dudx) % Equation to solve
    % Rxn is : A -> B, B -> A, B -> C
    global params;
    n1 = params(1); % tau/L^2
    c = 1;
    f = n1 .* dudx;
    s = 0;  
end

% ---------------------------------------------
function u0 = pdeic(x) % Initial Conditions
    u0 = 2 * ((x - 0.5).^2 < 0.1) + 1 * ((x - 0.5).^2 >= 0.1);
end
% ---------------------------------------------
function [pl,ql,pr,qr] = pdebc(xl,ul,xr,ur,t) % Boundary Conditions
    pl = 0;
    ql = 1;
    pr = 0;
    qr = 1;
end
% ---------------------------------------------