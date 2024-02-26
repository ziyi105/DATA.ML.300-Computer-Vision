x0 = -2;
y0 = 1;
x1 = 2;
y1 = 5;

x_values = linspace(-5, 5, 100);

y_values1 = -x0 * x_values + y0;
plot(x_values, y_values1, 'r', 'LineWidth', 2);
hold on;

y_values2 = -x1 * x_values + y1;
plot(x_values, y_values2, 'b', 'LineWidth', 2);

intersection_x = 1;
intersection_y = 3;
scatter(intersection_x, intersection_y, 100, 'g', 'filled');

text(intersection_x, intersection_y, sprintf('(%.2f, %.2f)', intersection_x, intersection_y), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 10);

xlabel('m');
ylabel('b');
title('Intersection of Two Lines');
legend('Line 1', 'Line 2', 'Intersection Point');

axis([-5 5 -5 10]);

hold off;

