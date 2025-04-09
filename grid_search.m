%% Parameter Settings
nx = 20; ny = 20; nz = 20;      % Grid resolution
xrange = [0 1000];               % X-axis range (meters)
yrange = [0 1000];               % Y-axis range (meters)
zrange = [0 1000];               % Z-axis range (meters)
n_stations = 5;                 % Number of stations
t0_search = [-5 5];             % t0 search range (seconds)
dt = 0.2;                       % Time step (seconds)

%% Generate Velocity Model
x = linspace(xrange(1), xrange(2), nx);
y = linspace(yrange(1), yrange(2), ny);
z = linspace(zrange(1), zrange(2), nz);
[XX, YY, ZZ] = ndgrid(x, y, z);  % 3D grid coordinates
velocity = 2000 + 3000*rand(nx, ny, nz); % Velocity range 2000-5000 m/s

%% Generate Stations (surface deployment)
station_pos = [rand(n_stations,2)*1000, 1000*ones(n_stations,1)];

%% True Source Parameters
src_idx = [randi(nx), randi(ny), randi(nz)]; % True source grid index
true_pos = [x(src_idx(1)), y(src_idx(2)), z(src_idx(3))]; % True position
t0_real = 0;                                   % True origin time
v_src = velocity(src_idx(1), src_idx(2), src_idx(3)); % Velocity at source

%% Calculate Theoretical Arrivals (with noise)
T_obs = zeros(n_stations, 1);
for s = 1:n_stations
    d = norm(true_pos - station_pos(s,:));
    T_obs(s) = t0_real + d/v_src;% + 0.02*rand()-0.02*rand();
end

%% Precompute Travel Time Field (vectorized)
all_pos = [XX(:), YY(:), ZZ(:)];               % All grid coordinates
n_grid = size(all_pos, 1);
travel_time = zeros(n_grid, n_stations);       % Travel time matrix

for s = 1:n_stations
    % Calculate distances from all grid points to current station
    delta = all_pos - station_pos(s,:);
    distances = sqrt(sum(delta.^2, 2));
    
    % Compute travel time: distance / velocity at grid point
    travel_time(:,s) = distances ./ velocity(:); % Vectorized calculation
end

%% Grid Search Algorithm
t0_values = t0_search(1):dt:t0_search(2);     % t0 candidates
min_energy = inf;                              % Initialize minimum energy
best_t0 = 0;                                   % Best origin time
best_pos = [0 0 0];                            % Best source position

% Progress bar
h = waitbar(0,'Searching for source...');

for t0 = t0_values
    % Compute time difference energy
    time_diff = T_obs' - t0;                  % Theoretical time differences
    energy = sum(abs(travel_time - time_diff), 2); % L1 norm
    
    % Find current minimum
    [current_min, idx] = min(energy);
    
    % Update global minimum
    if current_min < min_energy
        min_energy = current_min;
        best_t0 = t0;
        best_pos = all_pos(idx,:);
    end
    
    % Update progress bar
    waitbar((t0 - t0_search(1))/(t0_search(2)-t0_search(1)), h);
end
close(h);

%% Visualization
figure('Position', [200 200 1200 500])

% 3D localization results
subplot(1,2,1)
scatter3(station_pos(:,1), station_pos(:,2), station_pos(:,3),...
        'filled', 'MarkerFaceColor',[1 0 0], 'DisplayName','Stations');
hold on
scatter3(true_pos(1), true_pos(2), true_pos(3), 150,...
        'green','pentagram','filled','DisplayName','True Source');
scatter3(best_pos(1), best_pos(2), best_pos(3), 150,...
        'blue','^','filled','DisplayName','Estimated Source');
title(sprintf('Localization Result\nError: %.1f m', norm(true_pos - best_pos)));
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
legend show
grid on
view(40,30)

% Velocity model slices
subplot(1,2,2)
% Permute dimensions for slice visualization
velocity_perm = permute(velocity, [2,1,3]);  % Swap X-Y dimensions
% Regenerate grid coordinates with correct ordering
[YY_slice, XX_slice, ZZ_slice] = ndgrid(y, x, z); 

slice(XX_slice, YY_slice, ZZ_slice, velocity_perm, 500, 500, 500);
shading interp
colormap jet
colorbar
title('3D Velocity Model Slice');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');

%% Results Output
fprintf('\n=== Localization Results ===\n');
fprintf('True Position: (%.1f, %.1f, %.1f)\n', true_pos);
fprintf('Estimated Position: (%.1f, %.1f, %.1f)\n', best_pos);
fprintf('Position Error: %.2f m\n', norm(true_pos - best_pos));
fprintf('True Origin Time: %.2f s\n', t0_real);
fprintf('Estimated Origin Time: %.2f s\n', best_t0);
fprintf('Time Error: %.2f s\n', abs(t0_real - best_t0));
