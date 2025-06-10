% Copyright [2025] [Xiangjun Chen] directed by Prof. Enjiang Wang at OUC
%
% Licensed under the Apache License, Version 2.0 (the "License"): you may
% not use this file except in compliance with the License. You may obtain
% a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
% WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
% License for the specific language governing permissions and limitations
% under the License.

%% 2D Acoustic Wave Finite Difference Simulation
%  Uses a 2nd/4th/6th/8th-order central difference and Pseudo-spectral method for time derivative(Explicit time-marching);
%   2nd/4th/6th-order central differences for spatial derivatives;
%   Point source injection at grid center
%   Simple fixed boundaries (no absorption)
%   The prescibed model can be adjusted by choosing the code chunk in
%   paragragh 'Medium Parameter'
%   This project is still under construction(can be simplified and modularized)

% Clean workspace
clear; clc; close all;

% ========================
%% Simulation Parameters
% ========================
nx = 200;          % Grid points in x-direction
nz = 200;          % Grid points in z-direction
dx = 10;            % Spatial step in x (m)
dz = 10;            % Spatial step in z (m)
dt = 0.001;         % Time step (s)
nt = 350;          % Number of time steps
isnap = 10;         % Snapshot interval steps
order = input('Select spatial order (2/4/6/8/i): ');
if ~ismember(order, [2,4,6,8,i]) && ~strcmpi(order, 'i')
    error('Invalid order selection! Use 2/4/6/8/i');
end


% ========================
%% Medium Parameters
% ========================
vp = zeros(nz, nx);

% homogeneous
vp(:,:) = 3000;

%layered
% vp(1:150,:) = 2500;
% vp(151:200,:) = 4500;

%flaw
% vp(:,:) = 3000;
% vp(125,100) = 400;

%interlayer
% vp(:,:) = 2000;
% vp(120:150,:) = 5600;

% ========================
%% Stability Check
% ========================
CFL = max(vp) * dt * sqrt(1/dx^2 + 1/dz^2);
if CFL >= 1
    error('CFL condition violated!');
end

if strcmpi(order, 'i')
    CFL_spectral = max(vp) * dt * (sqrt(1/dx^2 + 1/dz^2)) * max([nx, nz])/pi;
    if CFL_spectral > 0.3
        error('Spectral method CFL > 0.3: dt=%.4f too large!', dt);
    end
end
% ========================
%% Wavefield Initialization
% ========================
p_old = zeros(nz, nx);     % Current time step
p_older = p_old;          % Previous time step

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
gif_filename = 'wave_t2_x246i.gif';
h = figure();
colormap("gray");  % Set colormap for better visibility
frame_count = 0;

% ========================
%% Main Simulation Loop
% ========================
tic
for it = 1:nt
    % Calculate spatial derivatives
    if order==2
        [p_xx, p_zz] = computeDerivatives2(p_old, dx, dz);
    elseif order==4
        [p_xx, p_zz] = computeDerivatives4(p_old, dx, dz);
    elseif order==6
        [p_xx, p_zz] = computeDerivatives6(p_old, dx, dz);
    elseif order==8
        [p_xx, p_zz] = computeDerivatives8(p_old, dx, dz);
    else
        [p_xx, p_zz] = computeDerivativesi(p_old, dx, dz);
    end
    
    % Initialize new wavefield
    p_new = zeros(nz, nx);
    
    % Core calculation region (excluding boundaries)
    if order==2 || order==4 || order==6 || order==8
        core_rows = 1+order/2:nz-order/2;
        core_cols = 1+order/2:nx-order/2;
    
        % Update wave equations
        p_new(core_rows, core_cols) = 2*p_old(core_rows, core_cols) - p_older(core_rows, core_cols) + ...
                                       dt^2*vp(core_rows, core_cols).^2.*(p_xx+p_zz);
    else
        p_new = 2*p_old - p_older + dt^2*vp.^2.*(p_xx+p_zz);
    end
    % Add source term
    p_new(src_z, src_x) = p_new(src_z, src_x) + dt^2 * src(it);
    
    % Update wavefields
    p_older = p_old;
    p_old = p_new;
    
    % Visualization and output
    if mod(it, isnap) == 0
        figure(h)
        imagesc(0:dx:2000,0:dz:2000,p_old);
        xlabel('x(m)')
        ylabel('y(m)')
        axis equal tight;
        caxis([-1e-7 1e-7]);
        colorbar;
        title(sprintf('Pressure @ step %d (%.2f s)', it, it*dt));
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
    end
end
toc

% ========================
%% Derivative Computation
% ========================
function [dxx, dzz] = computeDerivatives2(u, dx, dz)
    % Second derivative in x-direction
    dxx = (u(2:end-1, 3:end) - 2*u(2:end-1, 2:end-1) + u(2:end-1, 1:end-2)) / dx^2;
    
    % Second derivative in z-direction
    dzz = (u(3:end, 2:end-1) - 2*u(2:end-1, 2:end-1) + u(1:end-2, 2:end-1)) / dz^2;
end

function [dxx, dzz] = computeDerivatives4(u, dx, dz)
    % Second derivative in x-direction
    dxx = (-5/2*u(3:end-2,3:end-2) + 4/3*(u(3:end-2,4:end-1) + u(3:end-2,2:end-3)) - 1/12*(u(3:end-2,5:end) + u(3:end-2,1:end-4))) / dx^2;

    % Second derivative in z-direction
    dzz = (-5/2*u(3:end-2,3:end-2) + 4/3*(u(4:end-1,3:end-2) + u(2:end-3,3:end-2)) - 1/12*(u(5:end,3:end-2) + u(1:end-4,3:end-2))) / dz^2;
end

function [dxx, dzz] = computeDerivatives6(u, dx, dz)
    % Second derivative in x-direction
    dxx = (-49/18*u(4:end-3,4:end-3) +...
        3/2*(u(4:end-3,5:end-2) + u(4:end-3,3:end-4)) -...
        3/20*(u(4:end-3,6:end-1) + u(4:end-3,2:end-5)) +...
        1/90*(u(4:end-3,7:end) + u(4:end-3,1:end-6))) / dx^2;

    % Second derivative in z-direction
    dzz = (-49/18*u(4:end-3,4:end-3) +...
        3/2*(u(5:end-2,4:end-3) + u(3:end-4,4:end-3)) -...
        3/20*(u(6:end-1,4:end-3) + u(2:end-5,4:end-3)) +...
        1/90*(u(7:end,4:end-3) + u(1:end-6,4:end-3))) / dx^2;
end

function [dxx, dzz] = computeDerivatives8(u, dx, dz)
    dxx = (-2.8472*u(5:end-4,5:end-4) +...
        1.6*(u(5:end-4,6:end-3) + u(5:end-4,4:end-5)) -...
        0.2*(u(5:end-4,7:end-2) + u(5:end-4,3:end-6)) +...
        0.0254*(u(5:end-4,8:end-1) + u(5:end-4,2:end-7)) -...
        0.0018*(u(5:end-4,9:end) + u(5:end-4,1:end-8))) / dx^2;
    dzz = (-2.8472*u(5:end-4,5:end-4) +...
        1.6*(u(6:end-3,5:end-4) + u(4:end-5,5:end-4)) -...
        0.2*(u(7:end-2,5:end-4) + u(3:end-6,5:end-4)) +...
        0.0254*(u(8:end-1,5:end-4) + u(2:end-7,5:end-4)) -...
        0.0018*(u(9:end,5:end-4) + u(1:end-8,5:end-4))) / dz^2;
end

function [dxx, dzz] = computeDerivativesi(u, dx, dz)
    [nz, nx] = size(u);
    kx = 2*pi/(nx*dx) * [0:nx/2-1, -nx/2:-1];
    kz = 2*pi/(nz*dz) * [0:nz/2-1, -nz/2:-1];
    [KX, KZ] = meshgrid(kx, kz);
    U = fft2(u);
    fdxx = (-1i*KX).^2 .* U;
    fdzz = (-1i*KZ).^2 .* U;
    dxx = real(ifft2(fdxx));
    dzz = real(ifft2(fdzz));
end
