%% Parameter Settings
N = 256;        % Grid resolution
L = 15;         % Domain half-width
h = 2*L/(N-1);  % Grid spacing
max_iter = 1e4; % Maximum iterations
tolerance = 1e-6; % Convergence tolerance
src_radius = 0.1; % Source radius

%% Field Initialization
[x, y] = meshgrid(linspace(-L, L, N));
u = inf(N, N);  % Initialize time to infinity

% Set source region (propagation from center)
r = sqrt(x.^2 + y.^2);
u(r <= src_radius) = 0; % Source region initial time

% Generate slowness field
s = 0.05 + 0.45*rand(N,N); 
s(r <= src_radius) = 1; % Normalize source velocity

%% Eikonal Equation Solver Iteration
for iter = 1:max_iter
    u_old = u;
    u_new = u;
    
    % Traverse all grid points
    for i = 1:N
        for j = 1:N
            % Skip fixed source points
            if r(i,j) <= src_radius
                continue
            end
            
            % Get neighbor minima (handle boundary conditions)
            [ux_min, uy_min] = get_neighbor_mins(u, i, j);
            
            % Construct quadratic equation
            a = 2;
            b = -2*(ux_min + uy_min);
            c = (ux_min^2 + uy_min^2) - (s(i,j)*h)^2;
            
            discriminant = b^2 - 4*a*c;
            
            % Solution cases
            if discriminant >= 0
                root1 = (-b + sqrt(discriminant))/(2*a);
                root2 = (-b - sqrt(discriminant))/(2*a);
                u_candidate = max(root1, root2);
            else
                u_candidate = min(ux_min, uy_min) + s(i,j)*h;
            end
            
            % Update rule: take smaller time value
            u_new(i,j) = min(u(i,j), u_candidate);
        end
    end
    
    % Check convergence
    delta = max(abs(u_new(:) - u_old(:)));
    if delta < tolerance
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
    u = u_new;
    
    % Display progress every 100 iterations
    if mod(iter,100) == 0
        fprintf('Iteration %d, Max change: %.4e\n', iter, delta);
    end
end

%% Neighbor Minimum Calculation Function
function [ux_min, uy_min] = get_neighbor_mins(u, i, j)
    % x-direction neighbors
    if i == 1
        ux_min = u(i+1,j);
    elseif i == size(u,1)
        ux_min = u(i-1,j);
    else
        ux_min = min(u(i-1,j), u(i+1,j));
    end
    
    % y-direction neighbors
    if j == 1
        uy_min = u(i,j+1);
    elseif j == size(u,2)
        uy_min = u(i,j-1);
    else
        uy_min = min(u(i,j-1), u(i,j+1));
    end
end

%% Visualize Wave Propagation and Save as GIF
t_steps = linspace(0, max(u(:)), 50); % Generate time sequence
gif_name = 'wave_propagation.gif';   % GIF filename

figure('Position', [100 100 800 600]);
colormap('jet');
h_colorbar = colorbar('southoutside');
h_colorbar.Label.String = 'Slowness Field (s)';

for idx = 1:length(t_steps)
    t = t_steps(idx);
    
    % Plot graphics
    imagesc(x(1,:), y(:,1), s);
    caxis([0, 0.5]);
    set(gca, 'YDir', 'normal');
    
    hold on;
    contour(x, y, u, [t t], ...
        'LineWidth', 2, ...
        'LineColor', [1 1 1]);
    hold off;
    
    title(sprintf('Wavefront Propagation Time t = %.2f', t), ...
        'FontSize', 12, 'Color', 'k');
    axis equal tight;
    
    % Capture frame and save to GIF
    frame = getframe(gcf);
    im = frame2im(frame);
    [A, map] = rgb2ind(im, 256);
    
    % Set GIF parameters
    if idx == 1
        imwrite(A, map, gif_name, 'gif', ...
            'LoopCount', Inf, ...   % Loop indefinitely
            'DelayTime', 0.2);      % Frame interval
    else
        imwrite(A, map, gif_name, 'gif', ...
            'WriteMode', 'append', ...
            'DelayTime', 0.2);
    end
    
    drawnow;
end
