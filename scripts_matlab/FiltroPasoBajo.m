function FiltroPasoBajo(archivo, salida)

[A,pal] = imread(archivo);

if ndims(A) == 3    % comprobamos que la imagen sea en escala de gris
    error(strcat(archivo,' es imagen en color real'))
end

% Creamos el filtro paso bajo 3x3
tam = 3;    % matriz 3x3
H = 1/(tam^2) * ones(tam, tam);

% Se aplica filtrado paso bajo 3x3 y guardamos en disco
C = filter2(H, A);
imwrite(C, pal, salida)


