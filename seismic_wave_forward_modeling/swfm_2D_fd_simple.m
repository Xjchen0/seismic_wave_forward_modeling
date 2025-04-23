% 2D Elastic Wave Finite Difference Simulation (Velocity-Stress Formulation)

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

% ========================
%% Stability Check
% ========================
CFL = 0.5 * min([dx, dz]) / sqrt(vp^2 + vs^2);
if dt >= CFL
    error('Time step too large! CFL condition requires dt < %.4f', CFL);
end

% ========================
%% Wavefield Initialization
% ========================
ux_old = zeros(nz, nx);     % Current time step
ux_older = ux_old;          % Previous time step
uy_old = zeros(nz, nx);
uy_older = uy_old;

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
% GIF Initialization
% ========================
gif_filename = 'wave_propagation.gif';
h = figure('Position', [100, 100, 2500, 1500]);
colormap(jet);  % Set colormap for better visibility
frame_count = 0;

% ========================
%% Wavefield Separation Init
% ========================
sep_gif = 'wave_separation.gif';
sep_fig = figure('Position', [200, 200, 2500, 1500]);

% ========================
%% Main Simulation Loop
% ========================
for it = 1:nt
    % Calculate spatial derivatives
    [ux_xx, ux_zz, ux_xz] = computeDerivatives(ux_old, dx, dz);
    [uy_xx, uy_zz, uy_xz] = computeDerivatives(uy_old, dx, dz);
    
    % Initialize new wavefield
    ux_new = zeros(nz, nx);
    uy_new = zeros(nz, nx);
    
    % Core calculation region (excluding boundaries)
    core_rows = 2:nz-1;
    core_cols = 2:nx-1;
    
    % Update wave equations
    ux_new(core_rows, core_cols) = 2*ux_old(core_rows, core_cols) - ux_older(core_rows, core_cols) + ...
                                   dt^2*(vp^2*(ux_xx + uy_xz) + vs^2*(ux_zz - uy_xz));
    
    uy_new(core_rows, core_cols) = 2*uy_old(core_rows, core_cols) - uy_older(core_rows, core_cols) + ...
                                   dt^2*(vp^2*(uy_zz + ux_xz) + vs^2*(uy_xx - ux_xz));
    
    % Add source term
    ux_new(src_z, src_x) = ux_new(src_z, src_x) + dt^2 * src(it);
    
    % Update wavefields
    ux_older = ux_old;
    ux_old = ux_new;
    uy_older = uy_old;
    uy_old = uy_new;
    
    % Visualization and output
    if mod(it, isnap) == 0
        figure(h)
        subplot(1,2,1);
        imagesc(ux_new);
        axis equal tight;
        caxis([-1e-7 1e-7]);
        colorbar;
        title(sprintf('Horizontal Component (ux) @ step %d (%.2f s)', it, it*dt));
        
        subplot(1,2,2);
        imagesc(uy_new);
        axis equal tight;
        caxis([-1e-7 1e-7]);
        colorbar;
        title(sprintf('Vertical Component (uz) @ step %d (%.2f s)', it, it*dt));
        
        % Capture frame for GIF
        drawnow;
        frame = getframe(h);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        
        % Write to GIF
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
        
        %% wave field separating
        % finite diff method
        [dux_dx, dux_dz] = gradient(ux_old, dx, dz);
        [duy_dx, duy_dz] = gradient(uy_old, dx, dz);
        
        div = dux_dx + duy_dz;   % P-wave component
        curl_z = duy_dx - dux_dz; % S-wave component
        
        figure(sep_fig)
        subplot(1,2,1)
        imagesc(div)
        axis equal tight
        caxis([-1e-9 1e-9])
        title(sprintf('P-Wave (Divergence) @ %.2fs',it*dt))
        colorbar
        
        subplot(1,2,2)
        imagesc(curl_z)
        axis equal tight 
        caxis([-1e-9 1e-9])
        title(sprintf('S-Wave (Curl) @ %.2fs',it*dt))
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

% ========================
%% Derivative Computation
% ========================
function [dxx, dzz, dxz] = computeDerivatives(u, dx, dz)
    % Second derivative in x-direction
    dxx = (u(2:end-1, 3:end) - 2*u(2:end-1, 2:end-1) + u(2:end-1, 1:end-2)) / dx^2;
    
    % Second derivative in z-direction
    dzz = (u(3:end, 2:end-1) - 2*u(2:end-1, 2:end-1) + u(1:end-2, 2:end-1)) / dz^2;
    
    % Cross derivative term
    dxz = (u(3:end, 3:end) - u(1:end-2, 3:end) - u(3:end, 1:end-2) + u(1:end-2, 1:end-2)) / (4*dx*dz);
end
