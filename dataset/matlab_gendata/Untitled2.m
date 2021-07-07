clear;
yinit = [0.1,0.1,0.1];
orthmatrix = [1 0 0;
              0 1 0;
              0 0 1];

y = zeros(12,1);
% ��ʼ������
y(1:3) = yinit;
y(4:12) = orthmatrix;
tstart = 0; % ʱ���ʼֵ
tstep = 1e-3; % ʱ�䲽��
wholetimes = 1e5; % �ܵ�ѭ������
steps = 10; % ÿ���ݻ��Ĳ���
iteratetimes = wholetimes/steps; % �ݻ��Ĵ���
mod = zeros(3,1);
lp = zeros(3,1);
% ��ʼ������Lyapunovָ��
Lyapunov1 = zeros(iteratetimes,1);
Lyapunov2 = zeros(iteratetimes,1);
Lyapunov3 = zeros(iteratetimes,1);
for i=1:iteratetimes
    tspan = tstart:tstep:(tstart + tstep*steps);  
    [T,Y] = ode45(@(t,y) Lorenz_ly(t,y), tspan, y);
    % ȡ���ֵõ������һ��ʱ�̵�ֵ
    y = Y(size(Y,1),:);
    % ���¶�����ʼʱ��
    tstart = tstart + tstep*steps;
    y0 = [y(4) y(7) y(10);
          y(5) y(8) y(11);
          y(6) y(9) y(12)];
    %������
    y0 = ThreeGS(y0);
    % ȡ����������ģ
    mod(1) = sqrt(y0(:,1)'*y0(:,1));
    mod(2) = sqrt(y0(:,2)'*y0(:,2));
    mod(3) = sqrt(y0(:,3)'*y0(:,3));
    y0(:,1) = y0(:,1)/mod(1);
    y0(:,2) = y0(:,2)/mod(2);
    y0(:,3) = y0(:,3)/mod(3);
    lp = lp+log(abs(mod));
    %����Lyapunovָ��
    Lyapunov1(i) = lp(1)/(tstart);
    Lyapunov2(i) = lp(2)/(tstart);
    Lyapunov3(i) = lp(3)/(tstart);
        y(4:12) = y0';
end
% ��Lyapunovָ����ͼ
figure,
i = 1:iteratetimes;
plot(i,Lyapunov1,i,Lyapunov2,i,Lyapunov3)


%G-S������
function A = ThreeGS(V) % V Ϊ3*3����
v1 = V(:,1);
v2 = V(:,2);
v3 = V(:,3);
a1 = zeros(3,1);
a2 = zeros(3,1);
a3 = zeros(3,1);
a1 = v1;
a2 = v2-((a1'*v2)/(a1'*a1))*a1;
a3 = v3-((a1'*v3)/(a1'*a1))*a1-((a2'*v3)/(a2'*a2))*a2;
A = [a1,a2,a3];
end

function dX = Rossler_ly(t,X)
% Rossler�����ӣ���������Lyapunovָ��
%        a=0.15,b=0.20,c=10.0
%        dx/dt = -y-z,
%        dy/dt = x+ay,
%        dz/dt = b+z(x-c),
    a = 0.20;
    b = 0.20;
    c = 5.7;
x=X(1); y=X(2); z=X(3); 
% Y������������Ϊ�໥�����ĵ�λ����
Y = [X(4), X(7), X(10);
    X(5), X(8), X(11);
    X(6), X(9), X(12)];
% ��������ĳ�ʼ�����ز�����
dX = zeros(12,1);
% Rossler������
dX(1) = -y-z;
dX(2) = x+a*y;
dX(3) = b+z*(x-c);
% Rossler�����ӵ�Jacobi����
Jaco = [0 -1 -1;
        1 a 0;
        z 0 x-c];
dX(4:12) = Jaco*Y;
end

function dX = Lorenz_ly(t,X)
% Lorenz �����ӣ���������Lyapunovָ��
a = 10;
b = 28;
c = 8/3;
x=X(1); y=X(2); z=X(3);
% Y������������Ϊ�໥�����ĵ�λ����
Y = [X(4), X(7), X(10);
    X(5), X(8), X(11);
    X(6), X(9), X(12)];
% ��������ĳ�ʼ�����ز�����
dX = zeros(12,1);
% Lorenz ������
dX(1)=a*(y-x);
dX(2)=x*(b-z)-y;
dX(3)=x*y-c*z;
% Lorenz �����ӵ�Jacobi����
Jaco = [-a a 0;
        b-z -1 -x;
        y x -c];
dX(4:12) = Jaco*Y;
end