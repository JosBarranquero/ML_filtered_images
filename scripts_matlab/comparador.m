clc
ruta = '../prediccion/';

for i = 0:9
    [A_real] = imread(strcat(ruta, 'pred-',num2str(i),'-actual.bmp'));
    [A_pred] = imread(strcat(ruta, 'pred-',num2str(i),'.bmp'));
    
    disp(['A_real = ', num2str(A_real)])
    disp(['A_pred = ', num2str(A_pred)])
    disp('----------------')
end