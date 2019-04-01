clear all
close all

noiseSigma = 0.001; %noise sigma
filterSigma = 2; %gaussian Kernel sigma

%read the image 
img = imread('cameraman.tif');
f = double(img);

%-----select a filter and comment the other one------

%gaussian blur filter
h = fspecial('gaussian', 256, filterSigma);

%motion blur filter
%h = zeros(256,256);
%h(128,128:140) = 1/13;

%Fourier transforms
F = (fft2(f));

%transfer function
H = (fft2(fftshift(h)));

%blurred image
fh = ifft2((F.*H));

%add noise (gaussian or poisson)
g = imnoise(uint8(fh),'gaussian',noiseSigma,noiseSigma);
%g = poissrnd(fh);

G = (fft2(g));

%Compute the de-convolution in the wrong way
Fest = G./(H + eps);
fest = (ifft2(Fest));

%---------------------------------
%
%--------- Plot images ----------
%
%---------------------------------

figure()
subplot(1,3,1)
imshow(f,[])
title('Original f');

subplot(1,3,2)
imshow(fh,[])
title('Blurred fh')

subplot(1,3,3)
imshow(g,[])
str = strcat('Blurred + noise g, \sigma = ',num2str(noiseSigma));
title(str)

figure()
mesh(log(abs(F)));
title('F')

figure()
imagesc(log(abs(H)));
title('H')

figure()
mesh(log(abs(G)));
title('G')

figure()
imagesc(fest);
colormap(gray)
title('De-convolution in the wrong way')

%---------------------------------
%
%--------- EM algorithm ----------
%
%---------------------------------


nrip = 100;
err = zeros(1,nrip);
fk = double(g);
normf = norm(double(f));
bestRec = zeros(size(f));
bestK = 1;
errMin = 1;

%initial error
normDif = norm(double(g)-double(f));
startErr = normDif/normf;

%Iterations
for i = 1:nrip
    
    %denominator
    FK = fft2(fk);
    D = FK.*H;   
    d = ifft2(D);
    
    %numerator
    n = double(g)./(d + eps);
    N = fft2(n).*conj(H);
    
    a = ifft2(N);    
    fk1 = fk.*a;
     
    %update
    fk = fk1;
    
    %error
    normDif = norm(double(fk1)-double(f));
    err(i) = normDif/normf;
    
    %get the best reconstruction
    if err(i) < errMin
        
        bestRec = fk1;
        errMin = err(i);
        bestK = i;
        
    end
end

%get the percentage of image recovered
perc = (startErr - errMin)*100;

%plot the reconstruction error
figure();
plot(linspace(1,nrip,nrip),err);
grid on;
title('Error of the reconstruction');
legend('Reconstruction error');

%original image
figure()
imagesc(abs(f));
colormap(gray)
title('Original Image')

%blur + noise
figure()
imagesc(abs(g));
colormap(gray)
title('Blurred & noised image')

%best reconstruction
figure()
imagesc(abs(bestRec));
colormap(gray)
str = strcat('Best reconstruction, k=',num2str(bestK),', % image recovered=',num2str(perc),'%'); 
title(str)
