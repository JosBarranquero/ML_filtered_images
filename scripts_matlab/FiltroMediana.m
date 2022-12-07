function FiltroMediana(archivo, salida)

[A,pal] = imread(archivo);

if ndims(A) == 3    % comprobamos que la imagen sea en escala de gris
    error(strcat(archivo,' es imagen en color real'))
end

% Se aplica filtrado de mediana y guardamos en disco
C = medfilt2(A);
imwrite(C, pal, salida)