clear;
close all;
Z=[];  

d0=1e-7; 
ntransient = 1000;
nseg = 20000;
tstep = 0.001;
tseg = 0.1;

lsum=0;
x = -1;
y = 0;
z = 0.5;
x1 = -1;
y1 = 0+d0;
z1 = -0.5;
for t=1: ntransient+ nseg
    tspan = 0:tstep:tseg;
    [T1,Y1]=ode45(@(t,X) scroll4(X), tspan, [x;y;z]); 
    [T2,Y2]=ode45(@(t,X) scroll4(X), tspan, [x1;y1;z1]); 
    n1=length(Y1);n2=length(Y2);
    x=Y1(n1,1);y=Y1(n1,2);z=Y1(n1,3);
    x1=Y2(n2,1);y1=Y2(n2,2);z1=Y2(n2,3);
    d1=sqrt((x-x1)^2+(y-y1)^2+(z-z1)^2);

    % 新的偏离点在上一次计算的两轨迹末端的连线上，且距离仍等于d0
    x1=x + (d0/d1)*(x1-x);   
    y1=y + (d0/d1)*(y1-y);    
    z1=z + (d0/d1)*(z1-z);
    
    if t>ntransient
        lsum=lsum+log(d1/d0)/tseg;
        Z=[Z lsum/(t-ntransient)];
    end
end
     


close all;
plot(1:nseg,Z,'-k');  
title('Lorenz System''s LLE v.s. parameter b')  
xlabel('parameter b'),ylabel('Largest Lyapunov Exponents'); 
grid on;

figure,
plot(Y1), hold on
plot(Y2)


function dX = Lorenz(X) 
a = 10;
b = 28;
c = 8/3;
x=X(1); 
y=X(2); 
z=X(3);
dX = zeros(3,1);
dX(1)=a*(y-x);
dX(2)=x*(b-z)-y;
dX(3)=x*y-c*z;
end 

function dX = Rossler(X)
a = 0.2;
b = 0.2;
c = 5.7; 
x=X(1); 
y=X(2); 
z=X(3);
dX = zeros(3,1);
dX(1) = -y-z;
dX(2) = x+a*y;
dX(3) = b+z*(x-c);
end 


function dX = scroll3(X)
a = 0.977;
b = 10;
c = 4;
d = 0.1;
x=X(1); 
y=X(2); 
z=X(3);
dX = zeros(3,1);
dX(1)= a * (x - y) - y*z;
dX(2)= -b*y + x*z;
dX(3)= -c*z + d*x + x*y;
end 

function dX = scroll4(X) 
a = 1.46;
b = 9;
c = 5;
d = 0.06;
x=X(1); 
y=X(2); 
z=X(3);
dX = zeros(3,1);
dX(1)= a * (x - y) - y*z;
dX(2)= -b*y + x*z;
dX(3)= -c*z + d*x + x*y;
end 

function dX = RabinovichFabrikant(X) 
a = 1.1;
b = 0.87;
x=X(1); 
y=X(2); 
z=X(3);
dX = zeros(3,1);
dX(1)= y * (z - 1 + x * x) + b * x;
dX(2)= x * (3 * z + 1 - x * x) + b * y;
dX(3)= -2 * z * (a + x * y);
end