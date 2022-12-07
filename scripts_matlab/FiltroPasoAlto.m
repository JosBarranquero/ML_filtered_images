function FiltroPasoAlto(archivo, salida)

[A,pal] = imread(archivo);

if ndims(A) == 3    % comprobamos que la imagen sea en escala de gris
    error(strcat(archivo,' es imagen en color real'))
end

% Creamos el filtro paso alto 3x3
tam = 3;    % matriz 3x3
H = -1/(tam^2) * ones(tam, tam);
H(2,2) = H(2,2) + 1;

% Se aplica filtrado paso alto 3x3 y guardamos en disco
B = filter2(H, A);
imwrite(B, pal, salida)
