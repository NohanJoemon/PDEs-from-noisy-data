%% Solve
Nx = 100;
Nt = 100;
xs = linspace(0, 1, Nx);
ts = linspace(0, 1, Nt);
global params;
% params = [L^2/(tau * D), Lv/D, (k1 * L^2)/D]
params = [2, 1, 3];


m = 0;
u_sol_s = pdepe(m,@pdefun,@pdeic,@pdebc,xs,ts);

Nx = 20;
Nt = 20;
x = linspace(0, 1, Nx);
t = linspace(0, 1, Nt);

m = 0;
u_sol_l = pdepe(m,@pdefun,@pdeic,@pdebc,x,t);

% snr = 10;
% noiseA = normrnd(0, mean(std(u_sol)) .* (1/sqrt(snr)), Nx, Nt);

csvwrite('pfr/U2_100.csv', u_sol_s);
%% Plots 

surf(xs, ts, u_sol_s);
title("Variation of Concentration with Time and Space in a non-ideal PFR for reaction A -> B")
xlabel('Dimensionless Distance (x*)')
ylabel('Dimensionless Time (t*)')
hold on
surf(x, t, u_sol_l);

function [c, f, s] = pdefun(x, t, u, dudx) % Equation to solve
    % Rxn is : A -> B, B -> A, B -> C
    global params;
    n1 = params(1); % L^2/(tau * D)
    n2 = params(2); % Lv/D
    n3 = params(3); % (k1 * L^2)/D
    c = n1;
    f = 1 .* dudx;
    s = -n2 .* dudx(1) -n3 .* (u(1) .^ 2);  
end

% ---------------------------------------------
function u0 = pdeic(x) % Initial Conditions
    u0 = 1;
end
% ---------------------------------------------
function [pl,ql,pr,qr] = pdebc(xl,ul,xr,ur,t) % Boundary Conditions
    global params;
    n2 = params(2); % Lv/D
    pl = ul(1) - 1;
    ql = -1/n2;
    pr = 0;
    qr = 1;
end
% ---------------------------------------------