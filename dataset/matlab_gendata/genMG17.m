[x,t] = MackeyGlass(210000,17,1.0);
plot(x(10001:10:end));
hold on
[x1,t] = MackeyGlass(210000,17,1.01);
plot(x1(10001:10:end));


csvwrite('MackeyGlass.csv', x(10001:10:end));