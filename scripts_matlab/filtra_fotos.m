filtra_bajo = true;
filtra_alto = true;
filtra_mediana = true;

ruta = '../';

for i = 1:100
    entrada = strcat(ruta, 'imagen',num2str(i),'.bmp');

    if filtra_bajo
        salida_bajo = strcat(ruta, 'imagen',num2str(i),'-bajo.bmp');
        FiltroPasoBajo(entrada, salida_bajo);
    end
    
    if filtra_alto
        salida_alto = strcat(ruta, 'imagen',num2str(i),'-alto.bmp');
        FiltroPasoAlto(entrada, salida_alto);
    end
    
    if filtra_mediana
        salida_mediana = strcat(ruta, 'imagen',num2str(i),'-mediana.bmp');
        FiltroMediana(entrada, salida_mediana);
    end
end