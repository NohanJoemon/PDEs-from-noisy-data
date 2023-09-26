%% Solve
clc; clear;
Nx = 200;
Nt = 200;
x = linspace(0, 1, Nx);
t = linspace(0, 1, Nt);
global params;
% params = [L^2/(tau * D), Lv/D, (k1 * L^2)/D, (k2 * L^2)/D, (k3 * L^2)/D,
% Cb0/Ca0, Cc0/Ca0]
params = [1, 2, 1, 5, 1, 10, 0.01];
catrue = csvread('A_200.csv');
cbtrue = csvread('B_200.csv');
cctrue = csvread('C_200.csv');

m = 0;
cpred = pdepe(m,@pdefun,@pdeic,@pdebc,x,t);

%% Plots 
s1 = mesh(x, t, cctrue); 
s1.EdgeColor = [0.65, 0.32, 0.87];
s1.FaceColor = [0.65, 0.32, 0.87];
hold on
s2 = mesh(x, t, cpred(:, :, 3));   % change color map
s2.FaceColor = 'y';
legend('True Conc. C', 'Pred. Conc. C', 'Interpreter','latex', 'FontSize', 12)
title('Variation of Conc. of C across space and time in a PFR', 'Interpreter','latex', 'FontSize', 20)
xlabel('Dimensionless position ($x^{\ast}$)', 'Interpreter','latex', 'FontSize',14)
ylabel('Dimensionless time ($t^{\ast}$)', 'Interpreter','latex', 'FontSize',14)
zlabel('Dimensionless concentration ($C_{C}^{\ast}$)', 'Interpreter','latex', 'FontSize',14)
ax = gca;
ax.YAxis.FontSize = 13;
ax.YLabel.FontSize = 17;
ax.XAxis.FontSize = 13;
ax.XLabel.FontSize = 17;
ax.ZAxis.FontSize = 13;
ax.ZLabel.FontSize = 17;
%% Functions
function [c, f, s] = pdefun(x, t, u, dudx) % Equation to solve
    % Rxn is : A -> B, B -> A, B -> C
    c = [1; 1; 1];
    f = [1e-5; 1e-5; 1e-5] .* dudx;
    s = [8.1379 - 5.1205 * u(3) + 1.8517 * u(2) + 0.364 * u(1) * dudx(2);
         -10.3482 + 6.3597 * u(3) - 2.3546 * u(2) - 0.4609 * u(1)* dudx(2);
         1.82537 - 2.16701 * dudx(3) - 1.05824 * u(3) + 0.5023 * u(2)];  
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
    ql = [-1e5/n2; -1e5/n2; -1e5/n2];
    pr = [0; 0; 0];
    qr = [1; 1; 1];
end
% ---------------------------------------------