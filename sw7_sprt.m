% sw7.m - 2D Elastic Wave Simulation using Pseudo-Spectral Method

% Clean workspace
clear; clc; close all;

% ========================
%% Simulation Parameters
% ========================
nx = 2000;          % Grid points in x-direction
nz = 2000;          % Grid points in z-direction
dx = 10;            % Spatial step in x (m)
dz = 10;            % Spatial step in z (m)
dt = 0.001;         % Time step (s)
nt = 2500;          % Number of time steps
isnap = 50;         % Snapshot interval steps
% ========================
%% Medium Parameters
% ========================
vp = 2500;          % P-wave velocity (m/s)
vs = 1500;          % S-wave velocity (m/s)
rho = 2000;         % Density (kg/m³)

% ========================
%% Spectral Parameters
% ========================
kx = 2*pi/(nx*dx) * [0:nx/2-1, -nx/2:-1];  % Wavenumber in x
kz = 2*pi/(nz*dz) * [0:nz/2-1, -nz/2:-1];  % Wavenumber in z
[KX, KZ] = meshgrid(kx, kz);               % 2D wavenumber grid

% ========================
%% Stability Check
% ========================
CFL = 0.5 * min(dx, dz)/sqrt(vp^2 + vs^2);
if dt >= CFL
    error('Time step too large! CFL condition requires dt < %.4f', CFL);
end

% ========================
%% Wavefield Initialization
% ========================
ux = zeros(nz, nx);  % Horizontal displacement
uy = zeros(nz, nx);  % Vertical displacement

% ========================
%% Source Parameters
% ========================
src_x = floor(nx/2);        % X coordinate (grid point)
src_z = floor(nz/2);        % Z coordinate (grid point)
src_f0 = 20;                % Central frequency (Hz)
src_t0 = 1/src_f0;          % Time offset (s)

% Ricker wavelet source function
t = (0:nt-1)*dt;
src = (1 - 2*(pi*src_f0*(t - src_t0)).^2) .* exp(-(pi*src_f0*(t - src_t0)).^2);

% ========================
%% GIF Initialization
% ========================
gif_filename = 'wave_propagation_ps.gif';
h = figure('Position', [100, 100, 1200, 600]);
colormap(jet);
frame_count = 0;

% ========================
%% Wavefield Separation Init
% ========================
sep_gif = 'wave_separation_ps.gif';
sep_fig = figure('Position', [200, 200, 1200, 600]);

% ========================
%% Main Simulation Loop
% ========================
for it = 1:nt
    % Compute Fourier transforms
    Ux = fft2(ux);
    Uy = fft2(uy);
    
    % Compute spatial derivatives in Fourier domain
    Ux_xx = -(KX.^2) .* Ux;
    Ux_zz = -(KZ.^2) .* Ux;
    Ux_xz = -KX.*KZ .* Ux;
    
    Uy_xx = -(KX.^2) .* Uy;
    Uy_zz = -(KZ.^2) .* Uy;
    Uy_xz = -KX.*KZ .* Uy;
    
    % Inverse Fourier transform to get spatial derivatives
    ux_xx = real(ifft2(Ux_xx));
    ux_zz = real(ifft2(Ux_zz));
    ux_xz = real(ifft2(Ux_xz));
    
    uy_xx = real(ifft2(Uy_xx));
    uy_zz = real(ifft2(Uy_zz));
    uy_xz = real(ifft2(Uy_xz));
    
    % Update wave equations
    rhs_x = vp^2*(ux_xx + uy_xz) + vs^2*(ux_zz - uy_xz);
    rhs_y = vp^2*(uy_zz + ux_xz) + vs^2*(uy_xx - ux_xz);
    
    % Time integration using leap-frog scheme
    if it == 1
        % Initial condition
        ux_new = dt^2 * rhs_x;
        uy_new = dt^2 * rhs_y;
    else
        ux_new = 2*ux - ux_prev + dt^2 * rhs_x;
        uy_new = 2*uy - uy_prev + dt^2 * rhs_y;
    end
    
    % Add source term
    ux_new(src_z, src_x) = ux_new(src_z, src_x) + dt^2 * src(it);
    
    % Update wavefields
    ux_prev = ux;
    ux = ux_new;
    uy_prev = uy;
    uy = uy_new;
    
    % Visualization
    if mod(it, isnap) == 0
        figure(h)
        subplot(1,2,1);
        imagesc(ux);
        axis equal tight;
        caxis(1e-7*[-1 1]);
        colorbar;
        title(sprintf('UX @ t=%.3fs', it*dt));
        
        subplot(1,2,2);
        imagesc(uy);
        axis equal tight;
        caxis(1e-7*[-1 1]);
        colorbar; 
        title(sprintf('UY @ t=%.3fs', it*dt));
        
        % Export to GIF
        frame = getframe(h);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        
        if frame_count == 0
            imwrite(imind, cm, gif_filename, 'gif',...
                    'Loopcount', inf,...
                    'DelayTime', 0.1);
        else
            imwrite(imind, cm, gif_filename, 'gif',...
                    'WriteMode', 'append',...
                    'DelayTime', 0.1);
        end
        frame_count = frame_count + 1;
        
        %% wave field separation
        % Fourier method
        P = real(ifft2(1i*KX.*Ux + 1i*KZ.*Uy));  % 散度(P波)
        S = real(ifft2(1i*KZ.*Ux - 1i*KX.*Uy));  % 旋度(S波)
        
        figure(sep_fig)
        subplot(1,2,1)
        imagesc(P)
        axis equal tight
        caxis(1e-9*[-1 1])
        title(sprintf('P-Wave @ %.2fs',it*dt))
        colorbar
        
        subplot(1,2,2)
        imagesc(S)
        axis equal tight
        caxis(1e-9*[-1 1])
        title(sprintf('S-Wave @ %.2fs',it*dt)) 
        colorbar
        
        % save results
        frame = getframe(sep_fig);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        if it == isnap
            imwrite(imind, cm, sep_gif, 'gif',...
                    'Loopcount', inf,...
                    'DelayTime', 0.1);
        else
            imwrite(imind, cm, sep_gif, 'gif',...
                    'WriteMode', 'append',...
                    'DelayTime', 0.1);
        end
    end
end
