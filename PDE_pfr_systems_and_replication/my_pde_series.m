%% Solve
Nx = 50;
Nt = 50;
x = linspace(0, 1, Nx);
t = linspace(0, 1, Nt);
global params;
% params = [L^2/(tau * D), Lv/D, (k1 * L^2)/D, (k2 * L^2)/D, (k3 * L^2)/D,
% Cb0/Ca0, Cc0/Ca0]
params = [1, 2, 1, 5, 1, 10, 0.01];
snr = 10;

m = 0;
sol = pdepe(m,@pdefun,@pdeic,@pdebc,x,t);

cA = sol(:, :, 1);
cB = sol(:, :, 2);
cC = sol(:, :, 3);

noiseA = normrnd(0, mean(std(cA)) .* (1/sqrt(snr)), Nx, Nt);
noiseB = normrnd(0, mean(std(cB)) .* (1/sqrt(snr)), Nx, Nt);
noiseC = normrnd(0, mean(std(cC)) .* (1/sqrt(snr)), Nx, Nt);

csvwrite('cpfr/A_50.csv', cA);
csvwrite('cpfr/B_50.csv', cB);
csvwrite('cpfr/C_50.csv', cC);
%% Plots 

surf(x, t, cA);
title("Variation of Concentration with Time and Space in a non-ideal PFR for reaction A <=> B -> C")
xlabel('Dimensionless Distance (x*)')
ylabel('Dimensionless Time (t*)')
hold on
surf(x, t, cB);
surf(x, t, cC);
legend('A', 'B', 'C')
%% Functions
function [c, f, s] = pdefun(x, t, u, dudx) % Equation to solve
    % Rxn is : A -> B, B -> A, B -> C
    global params;
    n1 = params(1); % L^2/(tau * D)
    n2 = params(2); % Lv/D
    n3 = params(3); % (k1 * L^2)/D
    n4 = params(4); % (k2 * L^2)/D
    n5 = params(5); % (k3 * L^2)/D
    c = [n1; n1; n1];
    f = [1; 1; 1] .* dudx;
    s = [- n2 .* dudx(1) - n3 .* u(1) + n4 .* u(2) ; 
        - n2 .* dudx(2) + n3 .* u(1) - (n4 + n5) .* u(2) ; 
        - n2 .* dudx(3) + n5 .* u(2)];  
end

% ---------------------------------------------
function u0 = pdeic(x) % Initial Conditions
    global params;
    n6 = params(6); % Cb0/Ca0
    n7 = params(7); % Cc0/Ca0
    u0 = [1; n6; n7];
end
% ---------------------------------------------
function [pl,ql,pr,qr] = pdebc(xl,ul,xr,ur,t) % Boundary Conditions
    global params;
    n2 = params(2); % Lv/D
    n6 = params(6); % Cb0/Ca0
    n7 = params(7); % Cc0/Ca0
    pl = [ul(1) - 1; ul(2) - n6; ul(3) - n7];
    ql = [-1/n2; -1/n2; -1/n2];
    pr = [0; 0; 0];
    qr = [1; 1; 1];
end
% ---------------------------------------------