function dX = Rossler(t,X,params) 

a = params(1);
b = params(2);
c = params(3);

x=X(1); 
y=X(2); 
z=X(3);

dX = zeros(3,1);
dX(1)= -y-z;
dX(2)=x + a*y;
dX(3)=b + z * (x - c);

end 