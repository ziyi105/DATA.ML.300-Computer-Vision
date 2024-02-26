x0 = -2;
y0 = 1;
x1 = 2;
y1 = 5;

theta_values = linspace(-pi/2, pi/2, 100);

rho0 = x0 * cos(theta_values) + y0 * sin(theta_values);
rho1 = x1 * cos(theta_values) + y1 * sin(theta_values);

plot(theta_values, rho0, 'r', 'LineWidth', 2);
hold on;
plot(theta_values, rho1, 'b', 'LineWidth', 2);

theta_intersection = atan(-1);
rho_intersection = -2*cos(theta_intersection) + sin(theta_intersection);

plot(theta_intersection, rho_intersection, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');

xlabel('\theta');
ylabel('\rho');
title('Sinusoids in Polar Coordinate Parameter Space');
legend('Sinusoid 1', 'Sinusoid 2', 'Intersection Point');

hold off;
