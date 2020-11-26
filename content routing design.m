%%
 
%   Written by "Kais Suleiman" (ksuleiman.weebly.com)

%   Notes:
%
%   - The contents of this script have been explained in details 
%   in Chapter 6 of the thesis:
%       Kais Suleiman, "Popular Content Distribution in Public 
%       Transportation Using Artificial Intelligence Techniques.", 
%       Ph.D. thesis, University of Waterloo, Ontario, Canada, 2019.
%   - Notice that the very beginning and the very end of this script
%   address the cluster extraction function and operations explained 
%   in details at the end of Chapter 4 of the aforementioned thesis.
%   - Similar to the thesis, the functions to be used throughout 
%   content distribution tasks are introduced first.
%   - Simpler but still similar variable names have been used throughout 
%   this script instead of the mathematical notations used in the thesis.
%   - The assumptions used in the script are the same as those used in 
%   the thesis including those related to the case study considered 
%   representing the Grand River Transit bus service offered throughout 
%   the Region of Waterloo, Ontario, Canada.
%   - The following external MATLAB functions have been used throughout 
%   the script:
%       - Patrick Mineault (2020). Unique elements in cell array 
%       (https://www.mathworks.com/matlabcentral/fileexchange/
%       31718-unique-elements-in-cell-array), 
%       MATLAB Central File Exchange. Retrieved November 16, 2020.
%       - M Sohrabinia (2020). LatLon distance 
%       (https://www.mathworks.com/matlabcentral/fileexchange/
%       38812-latlon-distance), 
%       MATLAB Central File Exchange. Retrieved November 16, 2020.
%   - Figures and animations are created throughout this script to aid 
%   in thesis visualizations and other forms of results sharing.

%%

%   The extractCluster function:

function [biggest_cluster, cluster_members] = ...
    extractCluster(period_start_time, period_end_time, ...
    connectivities, time_delta, minimum_contact_duration, ...
    maximum_number_of_hops)

    for i = 1:size(connectivities{1},1)
    
        for t = period_start_time * 6:period_end_time * 6
        
            if t == period_start_time * 6
          
                period_connectivities{i} = ...
                    connectivities{t}(i,:); %#ok<*SAGROW>
            
            else
            
                period_connectivities{i} = ...
                    period_connectivities{i} + ...
                    connectivities{t}(i,:);
            
            end
        
        end
    
        previous_cluster_members{i} = ...
            find(period_connectivities{i} .* time_delta >= ...
            minimum_contact_duration);
    
    end

    next_cluster_members = previous_cluster_members;

    for n = 1:maximum_number_of_hops / 2 - 1
    
        for i = 1:size(connectivities{1},1)
        
            for j = 1:size(previous_cluster_members{i},2);
            
                next_cluster_members{i} = ...
                    union(next_cluster_members{i}, ...
                    previous_cluster_members{ ...
                    previous_cluster_members{i}(1,j)});
            
            end
        
        end
    
        previous_cluster_members = next_cluster_members;
    
    end

    [cluster_members,cluster_indices,~] = ...
        uniquecell(next_cluster_members);
 
    cluster_sizes = ...
    zeros(size(cluster_indices,1),1);

    for i = 1:size(cluster_indices,1)
    
        cluster_sizes(i) = ...
            size(cluster_members{i},2);
    
    end

    biggest_cluster = ...
        find(cluster_sizes == max(cluster_sizes));

    biggest_cluster = biggest_cluster(1);

end

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('connectivities');
load('modified_data');
load('synthetic_lats');
load('synthetic_lons');

%   Extracting the first biggest cluster connectivities:
%
%   The first biggest cluster is chosen within the specified
%   period start and end times given the first minimum discontinuous
%   contact duration. The next biggest clusters are chosen afterwards
%   according to the minimum contact durations schedule shown at the
%   end of this script.

period_start_time = 16 * 60;
period_end_time = 18 * 60;
time_delta = 10;
minimum_contact_duration = 20 * 60;
maximum_number_of_hops = 20;

[biggest_cluster, cluster_members] = ...
    extractCluster(period_start_time, period_end_time, ...
    connectivities, time_delta, ...
    minimum_contact_duration, maximum_number_of_hops);

for t = 1:(period_end_time * 6 - period_start_time * 6 + 1)
    
    biggest_cluster_connectivities{t} = ...
        connectivities{period_start_time * 6 + t - 1} ...
        (cluster_members{biggest_cluster}', ...
        cluster_members{biggest_cluster}'); ...
         
end

biggest_cluster_members = ...
    cluster_members{biggest_cluster};
    
save('biggest_cluster_data.mat', ...
    'biggest_cluster_connectivities', ...
    'biggest_cluster_members', ...
    'period_start_time', ...
    'period_end_time');

%   Visualizing the first biggest cluster at the busiest time:

busiest_time = ...
    (period_start_time + period_end_time) / 2 * 6;

title('First biggest cluster @ 5:00 PM','FontSize',18);
hold on

map = imread('map.jpg');
image('CData',map, ...
    'XData', ...
    [min(modified_data(:,6)) max(modified_data(:,6))], ...
    'YData', ...
    [min(modified_data(:,5)) max(modified_data(:,5))])
hold on

plot(synthetic_lons(:,busiest_time), ...
    synthetic_lats(:,busiest_time), ...
    'ko','MarkerFaceColor','y');
hold on

plot(synthetic_lons( ...
    biggest_cluster_members,busiest_time), ...
    synthetic_lats( ...
    biggest_cluster_members,busiest_time), ...
    'ko','MarkerFaceColor','r');
hold on

axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
    min(modified_data(:,5)) max(modified_data(:,5))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on

time = ...
    sprintf('%02d:%02d',floor(busiest_time / (60 * 6)), ...
    floor((busiest_time / (60 * 6) - floor(busiest_time / (60 * 6))) * 60));
text(-80.35,43.575, ...
    time,'Color','black','FontSize',20,'FontWeight','bold')
hold on

grid on   

saveas(figure(1),'first biggest cluster at 5 pm.fig');
saveas(figure(1),'first biggest cluster at 5 pm.bmp');

%%

%   The divideData function:

function initial_node_data_sizes = divideData(number_of_nodes, ...
    period_start_time, period_end_time, biggest_cluster_connectivities, ...
    numbers_of_data_segments, n)

    total_time = period_end_time * 6 - period_start_time * 6 + 1;
    
    for i = 1:number_of_nodes
        
        for t = 1:total_time
            
            if t == 1
                
                initial_node_data_sizes(i) = ...
                    sum(biggest_cluster_connectivities{t}(i,:));
                
            else
                
                initial_node_data_sizes(i) = initial_node_data_sizes(i) + ...
                    sum(biggest_cluster_connectivities{t}(i,:));
                
            end
            
        end
        
    end
    
    initial_node_data_sizes = ...
        floor(numbers_of_data_segments(n) .* ...
        initial_node_data_sizes ./ sum(initial_node_data_sizes));
    
end

%%

%   The allocateData function:

function segment_allocations = ...
    allocateData(number_of_nodes, initial_node_data_sizes)

    segment_allocations = ...
        zeros(number_of_nodes,sum(initial_node_data_sizes));
    
    for i = 1:number_of_nodes
        
        if i == 1
            
            segment_allocations(i,1:initial_node_data_sizes(i)) = ones;
            last_segment_allocation = initial_node_data_sizes(i);
            
        else
            
            segment_allocations(i, ...
                last_segment_allocation + 1: ...
                last_segment_allocation + initial_node_data_sizes(i)) = ones;
            
            last_segment_allocation = ...
                last_segment_allocation + initial_node_data_sizes(i);
            
        end
        
    end
    
end

%%

%   The computeFeatures function:

function node_features = computeFeatures(number_of_nodes, ...
    biggest_cluster_connectivities, time_index, segment_allocations) 

    node_features = zeros(number_of_nodes,4);
    
    for i = 1:number_of_nodes
            
        node_features(i,1) = ...
        	sum(biggest_cluster_connectivities{time_index}(i,:) == 1) ...
        	/ number_of_nodes;
            
        if node_features(i,1) ~= 0
                
        	node_segments = ...
            	find(segment_allocations(i,:) == 1);
                
            node_neighbors = ...
            	find(biggest_cluster_connectivities{time_index}(i,:) == 1);
                
            neighbors_neighbors = [];
                
            for j = 1:length(node_neighbors)
                    
            	neighbor_missing_segments = ...
                	find(segment_allocations(node_neighbors(j),:) == 0);
                    
                if ~ isempty(neighbor_missing_segments)
                        
                	node_features(i,2) = (node_features(i,2) * (j - 1) + ...
                    	length(node_segments(ismembc(node_segments, ...
                    	sort(neighbor_missing_segments)))) ...
                    	/ length(neighbor_missing_segments)) / j;
                        
                end
                    
                if j == 1
                        
                	shared_missing_segments = ...
                    	node_segments(ismembc(node_segments, ...
                    	sort(neighbor_missing_segments)));
                        
                else
                        
                    shared_missing_segments = ...
                    	shared_missing_segments(ismembc(shared_missing_segments, ...
                    	sort(node_segments(ismembc(node_segments, ...
                    	sort(neighbor_missing_segments))))));
                        
                end
                    
                neighbors_neighbors = [neighbors_neighbors,...
                	find(biggest_cluster_connectivities{time_index} ...
                	(node_neighbors(j),:) == 1)]; %#ok<AGROW>
                    
            end
                
            node_features(i,3) = ...
            	length(shared_missing_segments) / size(segment_allocations,2);
                
            node_features(i,4) = 1 - ...
            	length(setdiff(neighbors_neighbors,[node_neighbors,i])) ...
            	/ number_of_nodes;
                
        end
        
    end
        
end

%%

%   The controlRange function:

function [transmitter_neighbors, distances] = controlRange( ...
    transmitter_neighbors, synthetic_lats, synthetic_lons, ...
    biggest_cluster_members, transmitter, time_index, period_start_time, ...
    node_statuses)

    distances = zeros(size(transmitter_neighbors,1),1);
                    
    for j = 1:length(transmitter_neighbors)
                        
    	distances(j) = lldistkm( ...
        	[synthetic_lats( ...
        	biggest_cluster_members(transmitter),time_index + ...
        	period_start_time * 6 - 1) ...
        	synthetic_lons( ...
        	biggest_cluster_members(transmitter),time_index + ...
        	period_start_time * 6 - 1)], ...
        	[synthetic_lats( ...
        	biggest_cluster_members(transmitter_neighbors(j)),time_index + ...
        	period_start_time * 6 - 1) ...
        	synthetic_lons( ...
        	biggest_cluster_members(transmitter_neighbors(j)),time_index + ...
        	period_start_time * 6 - 1)]);
                        
    end
                    
    eliminated_neighbors = [];
                    
    for j = 1:length(transmitter_neighbors)
                        
    	if (node_statuses(transmitter_neighbors(j),1) == 1)|| ...
        	(node_statuses(transmitter_neighbors(j),3) == 1)
                            
        	eliminated_neighbors = ...
            	[eliminated_neighbors, ...
            	find(distances >= distances(j))]; %#ok<AGROW>
        end
                    
        transmitter_neighbors(eliminated_neighbors) = [];
        distances(eliminated_neighbors) = [];

    end
        
end

%%

%   The targetSegments function:

function targeted_segments = targetSegments(segment_allocations, ...
    transmitter, transmitter_neighbors, node_statuses)

    targeted_segments = [];
                        
	transmitter_segments = ...
    	find(segment_allocations(transmitter,:) == 1);
                                
    for j = 1:length(transmitter_neighbors)
                            
    	if node_statuses(transmitter_neighbors(j),2) == 0
                                
        	neighbor_missing_segments = ...
            	find(segment_allocations(transmitter_neighbors(j),:) == 0);
                                
        	targeted_segments = ...
            	[targeted_segments, ...
            	transmitter_segments(ismembc(transmitter_segments, ...
            	sort(neighbor_missing_segments)))]; %#ok<AGROW>
                                
        end
                            
    end
                        
    targeted_segments = unique(targeted_segments,'stable');

end

%%

%   The transmitData function:

function [segment_allocations, actions_history, node_statuses] = ...
    transmitData(node_statuses, transmitter, transmitter_neighbors, ...
    segment_allocations, targeted_segments, actions_history, time_index, ...
    period_start_time, distances, number_of_nodes)

    node_statuses(transmitter,1) = 1;
                            
    segment_popularities = ...
    	sum(segment_allocations(transmitter_neighbors( ...
    	node_statuses(transmitter_neighbors,2) == 0),:),1);
                            
	chosen_segment = ...
    	targeted_segments(segment_popularities(targeted_segments) == ...
    	min(segment_popularities(targeted_segments)));
                            
    chosen_segment = chosen_segment(1);
                            
    for k = 1:length(transmitter_neighbors)
                                
    	if (node_statuses(transmitter_neighbors(k),2) == 0) && ...
        	(segment_allocations ...
        	(transmitter_neighbors(k),chosen_segment) == 0)
                                    
        	node_statuses(transmitter_neighbors(k),2:3) = ones;
                                    
            segment_allocations ...
            	(transmitter_neighbors(k),chosen_segment) = 1;
                                    
            actions_history = vertcat(actions_history, ...
            	[time_index + period_start_time * 6 - 1,transmitter, ...
            	max(distances) * 1000,chosen_segment, ...
            	zeros(1,number_of_nodes)]); %#ok<AGROW>
                                    
            actions_history(end,4 + transmitter_neighbors(k)) = 1;
                                    
        end
                                
	end
                            
    node_statuses(transmitter_neighbors,2) = ones;

end

%%

%   The followPolicy function:

function [segment_allocations, actions_history, node_statuses] = ...
    followPolicy(period_start_time, period_end_time, number_of_nodes, ...
    biggest_cluster_connectivities, segment_allocations, omegas, ...
    synthetic_lats, synthetic_lons, biggest_cluster_members)

    actions_history = [];
    
    total_time = period_end_time * 6 - period_start_time * 6 + 1;
    
    for t = 1:total_time
        
        node_statuses = zeros(number_of_nodes,3);
        
        %   node_statuses(:,1) ==> indicate whether nodes are transmitting or not
        %   node_statuses(:,2) ==> indicate whether nodes are covered or not
        %   node_statuses(:,3) ==> indicate whether nodes are receiving or not
        
        %   Computing node features:
        
        node_features = computeFeatures(number_of_nodes, ...
            biggest_cluster_connectivities, t, segment_allocations);
        
        utility_function = ...
            node_features * omegas;
        
        [~,transmissions_order] = sortrows(utility_function,-1);
        
        for i = 1:number_of_nodes
 
            transmitter = transmissions_order(i);
            
            if node_statuses(transmitter,2) == 1
                
                continue;
                
            else
                
                transmitter_neighbors = ...
                    find(biggest_cluster_connectivities{t} ...
                    (transmitter,:) == 1);
                
                if (all(node_statuses(transmitter_neighbors,2)) == 1) || ...
                        isempty(transmitter_neighbors)
                    
                    continue;
                    
                else
                    
                    %   Controlling range:
                    
                    [transmitter_neighbors, distances] = controlRange( ...
                        transmitter_neighbors, synthetic_lats, synthetic_lons, ...
                        biggest_cluster_members, transmitter, t, ...
                        period_start_time, node_statuses);
                    
                    if isempty(transmitter_neighbors)
                        
                        continue;
                        
                    else
                        
                        %   Targeting segments:
                        
                        targeted_segments = targetSegments(segment_allocations, ...
                            transmitter, transmitter_neighbors, node_statuses);
                        
                        if isempty(targeted_segments)
                            
                            continue;
                            
                        else
                            
                            %   Transmitting data:
                            
                            [segment_allocations, actions_history, node_statuses] = ...
                                transmitData(node_statuses, transmitter, ...
                                transmitter_neighbors, segment_allocations, ...
                                targeted_segments, actions_history, t, ...
                                period_start_time, distances, number_of_nodes);
                            
                        end
                        
                    end
                    
                end
                
            end
            
        end
        
    end
    
end

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('biggest_cluster_data');
load('synthetic_lats');
load('synthetic_lons');

number_of_nodes = size(biggest_cluster_connectivities{1},1);
numbers_of_data_segments = 50:50:2000;
segments_number_variation = [];
minimum_segments_exchanged_over_distributed_ratio = 18;

%   Estimating the maximum number of data segments to be exchanged
%   using the naive-policy under a minimum 
%   "segments_exchanged_over_distributed_ratio":

for n = 1:length(numbers_of_data_segments)
    
    %   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, ...
        period_start_time, period_end_time, biggest_cluster_connectivities, ...
        numbers_of_data_segments, n));
    
    %   Allocating data:
    
    segment_allocations = ...
        allocateData(number_of_nodes, initial_node_data_sizes);
    
    %   Specifying the naive policy weights:
    
    omegas = [10;0;0;0];
    
    %   Notice that the naive policy is choosing the node with the highest
    %   number of neighbors normalized by the total number of nodes to start
    %   transmission first while ignoring the other features.
    
    %   Following the specified policy:
    
    [segment_allocations, actions_history, node_statuses] = ...
        followPolicy(period_start_time, period_end_time, number_of_nodes, ...
        biggest_cluster_connectivities, segment_allocations, omegas, ...
        synthetic_lats, synthetic_lons, biggest_cluster_members);
    
    segments_number_variation = vertcat(segments_number_variation, ...
        [numbers_of_data_segments(n),sum(initial_node_data_sizes), ...
        sum(sum(segment_allocations)) - sum(initial_node_data_sizes)]); %#ok<AGROW>
    
    progress = ...
        sprintf('%.0f percent', ...
        (n / length(numbers_of_data_segments)) * 100);
    
    clc;
    disp(progress);
    
end

figure(1);

figure_title = ...
    sprintf('Number of data segments\ninitially-distributed effect');
title(figure_title,'FontSize',20);
hold on
plot(segments_number_variation(:,1), ...
    segments_number_variation(:,3) ./ segments_number_variation(:,2), ...
    '-*k','LineWidth',2);
hold on
plot([0;segments_number_variation(:,1)], ...
    ones(1,1 + length(numbers_of_data_segments)) .* ...
    minimum_segments_exchanged_over_distributed_ratio, ...
    '--r','LineWidth',2);
hold on
label = ...
    sprintf('Number of data segments\ninitially-distributed');
xlabel(label,'FontSize',20);
hold on
label = ...
    sprintf('Segments exchanged\nover initially-distributed ratio');
ylabel(label,'FontSize',20);
hold on
grid on

saveas(figure(1),'segments number variation.fig');
saveas(figure(1),'segments number variation.bmp');

save('segments_number_variation.mat','segments_number_variation');

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('biggest_cluster_data');
load('synthetic_lats');
load('synthetic_lons');

number_of_visualizations = 6;
number_of_iterations = 100;
number_of_nodes = size(biggest_cluster_connectivities{1},1);
number_of_data_segments = 150;
visualizations_data = ...
    zeros(number_of_iterations * number_of_visualizations,4 + 1);

%   Building search space 3D-visualizations data:

for v = 1:number_of_visualizations
    
    for iteration = 1:number_of_iterations
        
        %   Dividing data:
    
        initial_node_data_sizes = divideData(number_of_nodes, ...
            period_start_time, period_end_time, biggest_cluster_connectivities, ...
            [number_of_data_segments], 1));
    
        %   Allocating data:
    
        segment_allocations = ...
            allocateData(number_of_nodes, initial_node_data_sizes);
        
        %   Specifying visualization slice policy weights:
        
        if v == 1
            
            omegas = randi([-10,10],4,1);
            omegas([1,2],1) = 0;
            
        end
        
        if v == 2
            
            omegas = randi([-10,10],4,1);
            omegas([1,3],1) = 0;

        end

        if v == 3
            
            omegas = randi([-10,10],4,1);
            omegas([1,4],1) = 0;

        end
        
        if v == 4
            
            omegas = randi([-10,10],4,1);
            omegas([2,3],1) = 0;

        end
        
        if v == 5
            
            omegas = randi([-10,10],4,1);
            omegas([2,4],1) = 0;

        end
        
        if v == 6
            
            omegas = randi([-10,10],4,1);
            omegas([3,4],1) = 0;

        end
        
        %   Following the specified policy:
        
        [segment_allocations, actions_history, node_statuses] = ...
            followPolicy(period_start_time, period_end_time, number_of_nodes, ...
            biggest_cluster_connectivities, segment_allocations, omegas, ...
            synthetic_lats, synthetic_lons, biggest_cluster_members);
        
        visualizations_data((v - 1) * number_of_iterations + iteration,:) = ...
            [omegas',length(actions_history)];
        
        progress = ...
            sprintf('%.0f percent', ...
            ((v - 1) * number_of_iterations + iteration) / ...
            (number_of_visualizations * number_of_iterations) * 100);
        clc;
        disp(progress);
        
    end

end

save('visualizations_data.mat','visualizations_data');

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('visualizations_data');

%   Visualizing search space 3D-data:

figure(1);

title('Search space with \omega_{1} & \omega_{2} set to 0', ...
    'FontSize',18);
hold on;
xlabel('\omega_{3}','FontSize',18,'FontWeight','Bold');
hold on;
ylabel('\omega_{4}','FontSize',18,'FontWeight','Bold');
hold on;

v = 1;
x = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,3);
y = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,4);
z = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,5);

tri = delaunay(x,y);
trisurf(tri,x,y,z,'linestyle','none')
shading interp
view(2)
colorbar

saveas(figure(1),'search space with zero omega 1 and 2.fig');
saveas(figure(1),'search space with zero omega 1 and 2.bmp');

figure(2);

title('Search space with \omega_{1} & \omega_{3} set to 0', ...
    'FontSize',18);
hold on;
xlabel('\omega_{2}','FontSize',18,'FontWeight','Bold');
hold on;
ylabel('\omega_{4}','FontSize',18,'FontWeight','Bold');
hold on;

v = 2;
x = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,2);
y = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,4);
z = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,5);

tri = delaunay(x,y);
trisurf(tri,x,y,z,'linestyle','none')
shading interp
view(2)
colorbar

saveas(figure(2),'search space with zero omega 1 and 3.fig');
saveas(figure(2),'search space with zero omega 1 and 3.bmp');

figure(3);

title('Search space with \omega_{1} & \omega_{4} set to 0', ...
    'FontSize',18);
hold on;
xlabel('\omega_{2}','FontSize',18,'FontWeight','Bold');
hold on;
ylabel('\omega_{3}','FontSize',18,'FontWeight','Bold');
hold on;

v = 3;
x = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,2);
y = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,3);
z = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,5);

tri = delaunay(x,y);
trisurf(tri,x,y,z,'linestyle','none')
shading interp
view(2)
colorbar

saveas(figure(3),'search space with zero omega 1 and 4.fig');
saveas(figure(3),'search space with zero omega 1 and 4.bmp');

figure(4);

title('Search space with \omega_{2} & \omega_{3} set to 0', ...
    'FontSize',18);
hold on;
xlabel('\omega_{1}','FontSize',18,'FontWeight','Bold');
hold on;
ylabel('\omega_{4}','FontSize',18,'FontWeight','Bold');
hold on;

v = 4;
x = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,1);
y = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,4);
z = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,5);

tri = delaunay(x,y);
trisurf(tri,x,y,z,'linestyle','none')
shading interp
view(2)
colorbar

saveas(figure(4),'search space with zero omega 2 and 3.fig');
saveas(figure(4),'search space with zero omega 2 and 3.bmp');

figure(5);

title('Search space with \omega_{2} & \omega_{4} set to 0', ...
    'FontSize',18);
hold on;
xlabel('\omega_{1}','FontSize',18,'FontWeight','Bold');
hold on;
ylabel('\omega_{3}','FontSize',18,'FontWeight','Bold');
hold on;

v = 5;
x = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,1);
y = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,3);
z = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,5);

tri = delaunay(x,y);
trisurf(tri,x,y,z,'linestyle','none')
shading interp
view(2)
colorbar

saveas(figure(5),'search space with zero omega 2 and 4.fig');
saveas(figure(5),'search space with zero omega 2 and 4.bmp');

figure(6);

title('Search space with \omega_{3} & \omega_{4} set to 0', ...
    'FontSize',18);
hold on;
xlabel('\omega_{1}','FontSize',18,'FontWeight','Bold');
hold on;
ylabel('\omega_{2}','FontSize',18,'FontWeight','Bold');
hold on;

v = 6;
x = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,1);
y = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,2);
z = ...
    visualizations_data((v - 1) * 100 + 1:v * 100,5);

tri = delaunay(x,y);
trisurf(tri,x,y,z,'linestyle','none')
shading interp
view(2)
colorbar

saveas(figure(6),'search space with zero omega 3 and 4.fig');
saveas(figure(6),'search space with zero omega 3 and 4.bmp');

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('biggest_cluster_data');
load('synthetic_lats');
load('synthetic_lons');

number_of_iterations = 100;
number_of_nodes = size(biggest_cluster_connectivities{1},1);
number_of_data_segments = 150;
regression_data = zeros(number_of_iterations,4 + 1);

%   Generating initial regression data:

for iteration = 1:number_of_iterations
    
    %   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, ...
        period_start_time, period_end_time, biggest_cluster_connectivities, ...
        [number_of_data_segments], 1));
    
    %   Allocating data:
    
    segment_allocations = ...
        allocateData(number_of_nodes, initial_node_data_sizes);

    %   Specifying the random policy weights:
    
    omegas = randi([-10,10],4,1);
    
    %   Following the specified policy:

    [segment_allocations, actions_history, node_statuses] = ...
        followPolicy(period_start_time, period_end_time, number_of_nodes, ...
        biggest_cluster_connectivities, segment_allocations, omegas, ...
        synthetic_lats, synthetic_lons, biggest_cluster_members);               
    
    regression_data(iteration,:) = ...
        [omegas',length(actions_history)];
    
    progress = ...
        sprintf('%.0f percent',iteration / number_of_iterations * 100);
    clc;
    disp(progress);
    
end

save('regression_data.mat','regression_data');

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('biggest_cluster_data');
load('regression_data');
load('synthetic_lats');
load('synthetic_lons');

number_of_iterations = 900;
number_of_random_points = 1000;
number_of_nodes = size(biggest_cluster_connectivities{1},1);
number_of_data_segments = 150;

%   Bayesian optimization using GP regression:

optimized_gp_regression_data = regression_data;
optimized_gp_execution_times = zeros(number_of_iterations,1);

for iteration = 1:number_of_iterations
    
    for i = 1:10
        
        tic;
        
        gp_model = ...
            fitrgp(optimized_gp_regression_data(:,1:4), ...
            optimized_gp_regression_data(:,4 + 1), ...
            'FitMethod','exact');
        
        optimized_gp_execution_times(iteration) = ...
            ((i - 1) * optimized_gp_execution_times(iteration) + toc) / i;
        
    end

    %   Using UCB:
    
    random_data = ...
        zeros(number_of_random_points,4 + 1);

    for i = 1:number_of_random_points
            
        omegas = randi([-10,10],4,1);

        random_data(i,1:4) = omegas;
        
        [prediction,standard_deviation] = ...
            predict(gp_model,random_data(i,1:4));
        
        random_data(i,4 + 1) = ...
            round(prediction + standard_deviation);
        
    end
    
    best_random_point = random_data(random_data(:,4 + 1) == ...
        max(random_data(:,4 + 1)),:);
              
    %   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, ...
        period_start_time, period_end_time, biggest_cluster_connectivities, ...
        [number_of_data_segments], 1));
    
    %   Allocating data:
    
    segment_allocations = ...
        allocateData(number_of_nodes, initial_node_data_sizes);
    
    %   Specifying the best policy weights found so far:
    
    omegas = best_random_point(1,1:4)';
    
    %   Following the specified policy weights:

    [segment_allocations, actions_history, node_statuses] = ...
        followPolicy(period_start_time, period_end_time, number_of_nodes, ...
        biggest_cluster_connectivities, segment_allocations, omegas, ...
        synthetic_lats, synthetic_lons, biggest_cluster_members);               
    
    optimized_gp_regression_data = vertcat(optimized_gp_regression_data, ...
        [omegas',length(actions_history)]); %#ok<AGROW>

    progress = ...
        sprintf('%.0f percent',iteration / number_of_iterations * 100);
    clc;
    disp(progress);
        
end

save('optimized_gp_data.mat','optimized_gp_regression_data', ...
    'optimized_gp_execution_times');

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('biggest_cluster_data');
load('regression_data');
load('synthetic_lats');
load('synthetic_lons');

number_of_iterations = 900;
number_of_random_points = 1000;
number_of_nodes = size(biggest_cluster_connectivities{1},1);
number_of_data_segments = 150;

%   Bayesian optimization using RF regression:

optimized_rf_regression_data = regression_data;
optimized_rf_execution_times = zeros(number_of_iterations,1);

for iteration = 1:number_of_iterations
    
    for i = 1:10
        
        tic;
        
        for j = 1:10
            
            tree_model{j} = ...
                fitrtree(optimized_rf_regression_data(:,1:4), ...
                optimized_rf_regression_data(:,4 + 1), ...
                'MinLeafSize',j);
            
        end
                
        optimized_rf_execution_times(iteration) = ...
            ((i - 1) * optimized_rf_execution_times(iteration) + toc) / i;
        
    end

    %   Using UCB:
    
    random_data = ...
        zeros(number_of_random_points,4 + 1);

    for i = 1:number_of_random_points
        
        omegas = randi([-10,10],4,1);

        random_data(i,1:4) = omegas;
        
        for j = 1:10
            
            tree_predictions(j) = ...
                predict(tree_model{j},random_data(i,1:4));
            
        end
        
        prediction = mean(tree_predictions);
        
        standard_deviation = ...
            sqrt(1 / 10 * sum(tree_predictions .^ 2) - prediction ^ 2);
        
        random_data(i,4 + 1) = ...
            round(prediction + standard_deviation);

    end
    
    best_random_point = random_data(random_data(:,4 + 1) == ...
        max(random_data(:,4 + 1)),:);

    %   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, ...
        period_start_time, period_end_time, biggest_cluster_connectivities, ...
        [number_of_data_segments], 1));
    
    %   Allocating data:
    
    segment_allocations = ...
        allocateData(number_of_nodes, initial_node_data_sizes);
    
    %   Specifying the best policy weights found so far:
    
    omegas = best_random_point(1,1:4)';
    
    %   Following the specified policy weight:

    [segment_allocations, actions_history, node_statuses] = ...
        followPolicy(period_start_time, period_end_time, number_of_nodes, ...
        biggest_cluster_connectivities, segment_allocations, omegas, ...
        synthetic_lats, synthetic_lons, biggest_cluster_members);               
    
    optimized_rf_regression_data = vertcat(optimized_rf_regression_data, ...
        [omegas',length(actions_history)]); %#ok<AGROW>
        
    progress = ...
        sprintf('%.0f percent',iteration / number_of_iterations * 100);
    clc;
    disp(progress);
        
end

save('optimized_rf_data.mat','optimized_rf_regression_data', ...
    'optimized_rf_execution_times');

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('biggest_cluster_data');
load('regression_data');
load('synthetic_lats');
load('synthetic_lons');

number_of_iterations = 900;
number_of_random_points = 1000;
number_of_nodes = size(biggest_cluster_connectivities{1},1);
number_of_data_segments = 150;

%   Bayesian optimization using NN regression:

optimized_nn_regression_data = regression_data;
optimized_nn_execution_times = zeros(number_of_iterations,1);

for iteration = 1:number_of_iterations
    
    for i = 1:10
        
        tic;
        
        nn_model = fitnet([10,10,10]);
        
        nn_model.layers{1}.transferFcn = 'tansig';
        nn_model.layers{2}.transferFcn = 'tansig';
        nn_model.layers{3}.transferFcn = 'tansig';
        nn_model.layers{4}.transferFcn = 'purelin';
        
        nn_model.trainParam.showWindow = 0;
        
        nn_model = ...
            train(nn_model, ...
            optimized_nn_regression_data(:,1:4)', ...
            optimized_nn_regression_data(:,4 + 1)');
        
        %   Replacing the last hidden layer:
        
        nn_model.layerConnect(4,3) = 0;
        nn_model.outputConnect(1,4) = 0;
        nn_model.outputConnect(1,3) = 1;
        
        nn_model_predictions = sim(nn_model, ...
            optimized_nn_regression_data(:,1:4)');
        
        l_model = ...
            fitlm(nn_model_predictions', ...
            optimized_nn_regression_data(:,4 + 1));
        
        optimized_nn_execution_times(iteration) = ...
            ((i - 1) * optimized_nn_execution_times(iteration) + toc) / i;
        
    end
    
    %   Using UCB:
    
    random_data = ...
        zeros(number_of_random_points,4 + 1);

    for i = 1:number_of_random_points
        
        omegas = randi([-10,10],4,1);

        random_data(i,1:4) = omegas;
        
        prediction = ...
            predict(l_model, ...
            sim(nn_model,random_data(i,1:4)')');
        
        random_data(i,4 + 1) = ...
            round(prediction + l_model.RMSE);

    end
    
    best_random_point = random_data(random_data(:,4 + 1) == ...
        max(random_data(:,4 + 1)),:);

    %   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, ...
        period_start_time, period_end_time, biggest_cluster_connectivities, ...
        [number_of_data_segments], 1));
    
    %   Allocating data:
    
    segment_allocations = ...
        allocateData(number_of_nodes, initial_node_data_sizes);
    
    %   Specifying the best policy weights found so far:
    
    omegas = best_random_point(1,1:4)';
    
    %   Following the specified policy weights:
   
    [segment_allocations, actions_history, node_statuses] = ...
        followPolicy(period_start_time, period_end_time, number_of_nodes, ...
        biggest_cluster_connectivities, segment_allocations, omegas, ...
        synthetic_lats, synthetic_lons, biggest_cluster_members);               
        
    optimized_nn_regression_data = vertcat(optimized_nn_regression_data, ...
        [omegas',length(actions_history)]); %#ok<AGROW>
    
    progress = ...
        sprintf('%.0f percent',iteration / number_of_iterations * 100);
    clc;
    disp(progress);
        
end

save('optimized_nn_data.mat','optimized_nn_regression_data', ...
    'optimized_nn_execution_times');

%%

clear all;
clc;

load('optimized_gp_data');
load('optimized_rf_data');
load('optimized_nn_data');

%   Cleaning the execution-time results of the GP regression,
%   the RF regression and the NN regression used for Bayesian optimization:

%   Removing outliers:

outlier_removing_window = 10;

for i = outlier_removing_window + 1: ...
        length(optimized_gp_execution_times) - outlier_removing_window
    
    if (optimized_gp_execution_times(i,1) > ...
            prctile(optimized_gp_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) + ...
            1.5 * (prctile(optimized_gp_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) - ...
            prctile(optimized_gp_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25))) || ...
            (optimized_gp_execution_times(i,1) < ...
            prctile(optimized_gp_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25) - ...
            1.5 * (prctile(optimized_gp_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) - ...
            prctile(optimized_gp_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25)))
        
        optimized_gp_execution_times(i,1) = ...
            mean(optimized_gp_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1));
        
    end
    
end

for i = outlier_removing_window + 1: ...
        length(optimized_rf_execution_times) - outlier_removing_window
    
    if (optimized_rf_execution_times(i,1) > ...
            prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) + ...
            1.5 * (prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) - ...
            prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25))) || ...
            (optimized_rf_execution_times(i,1) < ...
            prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25) - ...
            1.5 * (prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) - ...
            prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25)))
        
        optimized_rf_execution_times(i,1) = ...
            mean(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1));
        
    end
    
end

for i = outlier_removing_window + 1: ...
        length(optimized_nn_execution_times) - outlier_removing_window

    if (optimized_nn_execution_times(i,1) > ...
            prctile(optimized_nn_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) + ...
            1.5 * (prctile(optimized_nn_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) - ...
            prctile(optimized_nn_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25))) || ...
            (optimized_nn_execution_times(i,1) < ...
            prctile(optimized_nn_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25) - ...
            1.5 * (prctile(optimized_nn_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) - ...
            prctile(optimized_nn_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25)))
        
        optimized_nn_execution_times(i,1) = ...
            mean(optimized_nn_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1));
        
    end
    
end

%   Smoothing data:

moving_average_window = 10;

for i = moving_average_window + 1: ...
        length(optimized_gp_execution_times) - moving_average_window
    
    optimized_gp_execution_times(i,1) = ...
        mean(optimized_gp_execution_times ...
        (i - moving_average_window:i + moving_average_window,1));
    
end

for i = moving_average_window + 1: ...
        length(optimized_rf_execution_times) - moving_average_window
    
    optimized_rf_execution_times(i,1) = ...
        mean(optimized_rf_execution_times ...
        (i - moving_average_window:i + moving_average_window,1));
    
end

for i = moving_average_window + 1: ...
        length(optimized_nn_execution_times) - moving_average_window
    
    optimized_nn_execution_times(i,1) = ...
        mean(optimized_nn_execution_times ...
        (i - moving_average_window:i + moving_average_window,1));
    
end

optimized_gp_smoothed_execution_times = ...
    optimized_gp_execution_times;

optimized_rf_smoothed_execution_times = ...
    optimized_rf_execution_times;

optimized_nn_smoothed_execution_times = ...
    optimized_nn_execution_times;

save('optimized_gp_smoothed_execution_times.mat', ...
    'optimized_gp_smoothed_execution_times');

save('optimized_rf_smoothed_execution_times.mat', ...
    'optimized_rf_smoothed_execution_times');

save('optimized_nn_smoothed_execution_times.mat', ...
    'optimized_nn_smoothed_execution_times');

%%

clear all;
clc;

load('optimized_gp_data');
load('optimized_rf_data');
load('optimized_nn_data');
load('optimized_gp_smoothed_execution_times');
load('optimized_rf_smoothed_execution_times');
load('optimized_nn_smoothed_execution_times');

%   Comparing the GP regression, the RF regression
%   and the NN regression used for Bayesian optimization:

figure(1);

title('Bayesian optimization performances', ...
    'FontSize',18,'FontWeight','bold');
hold on;
xlabel('Iteration index', ...
    'FontSize',18,'FontWeight','bold');
hold on;
ylabel('Number of data segments exchanged', ...
    'FontSize',18,'FontWeight','bold');
hold on;
plot(1:size(optimized_gp_regression_data,1), ...
    optimized_nn_regression_data(:,end),'-ob','LineWidth',1.5);
hold on;
plot(1:size(optimized_gp_regression_data,1), ...
    optimized_rf_regression_data(:,end),'-*k','LineWidth',1.5);
hold on;
plot(1:size(optimized_gp_regression_data,1), ...
    optimized_gp_regression_data(:,end),'--r','LineWidth',1.5);
hold on;
xlim([0,size(optimized_gp_regression_data,1)]);
ylim([min( ...
    [min(optimized_gp_regression_data(:,end)), ...
    min(optimized_rf_regression_data(:,end)), ...
    min(optimized_nn_regression_data(:,end))]) * 0.9, ...
    max( ...
    [max(optimized_gp_regression_data(:,end)), ...
    max(optimized_rf_regression_data(:,end)), ...
    max(optimized_nn_regression_data(:,end))]) * 1.1]);
hold on;
legend({'w/ Neural-network-regression', ...
    'w/ Random-forest-regression', ...
    'w/ Gaussian-processes-regression'}, ...
    'Location','Best','FontSize',16);
hold on;
grid on;

saveas(figure(1),'gp vs. rf vs. nn - data segments exchanged.fig');
saveas(figure(1),'gp vs. rf vs. nn - data segments exchanged.bmp');

figure(2);

title('Bayesian optimization performances', ...
    'FontSize',18,'FontWeight','bold');
hold on;
xlabel('Number of observations', ...
    'FontSize',18,'FontWeight','bold');
hold on;
ylabel('Execution time (msec)', ...
    'FontSize',18,'FontWeight','bold');
hold on;
plot(1:size(optimized_gp_execution_times,1), ...
    mean(optimized_nn_smoothed_execution_times,2) .* 1000,'-ob','LineWidth',1.5);
hold on;
plot(1:size(optimized_gp_execution_times,1), ...
    mean(optimized_rf_smoothed_execution_times,2) .* 1000,'-*k','LineWidth',1.5);
hold on;
plot(1:size(optimized_gp_execution_times,1), ...
    mean(optimized_gp_smoothed_execution_times,2) .* 1000,'--r','LineWidth',1.5);
hold on;
xlim([0,size(optimized_gp_execution_times,1)]);
ylim([min( ...
    [min(mean(optimized_gp_smoothed_execution_times,2)), ...
    min(mean(optimized_rf_smoothed_execution_times,2)), ...
    min(mean(optimized_nn_smoothed_execution_times,2))]) * 1000 * 0.9, ...
    max( ...
    [max(mean(optimized_gp_smoothed_execution_times,2)), ...
    max(mean(optimized_rf_smoothed_execution_times,2)), ...
    max(mean(optimized_nn_smoothed_execution_times,2))]) * 1000 * 1.1]);
hold on;
legend({'w/ Neural-network-regression', ...
    'w/ Random-forest-regression', ...
    'w/ Gaussian-processes-regression'}, ...
    'Location','Best','FontSize',16);
hold on;
grid on;

saveas(figure(2),'gp vs. rf vs. nn - execution times.fig');
saveas(figure(2),'gp vs. rf vs. nn - execution times.bmp');

figure(3);

title('Policy weights under GP regression', ...
    'FontSize',18,'FontWeight','bold');
hold on;
xlabel('Iteration index','FontSize',18,'FontWeight','bold');
hold on;
ylabel('Weight value','FontSize',18,'FontWeight','bold');
hold on;
xlim([1,size(optimized_gp_regression_data,1)]);
ylim([-10,10]);
hold on;

for i = 1:size(optimized_gp_regression_data,2) - 1

    plot(1:size(optimized_gp_regression_data,1), ...
        optimized_gp_regression_data(:,i),'color',rand(1,3));
    hold on;
    
end

grid on;

saveas(figure(3),'policy weights under GP regression.fig');
saveas(figure(3),'policy weights under GP regression.bmp');

figure(4);

title('Policy weights under RF regression', ...
    'FontSize',18,'FontWeight','bold');
hold on;
xlabel('Iteration index','FontSize',18,'FontWeight','bold');
hold on;
ylabel('Weight value','FontSize',18,'FontWeight','bold');
hold on;
xlim([1,size(optimized_rf_regression_data,1)]);
ylim([-10,10]);
hold on;

for i = 1:size(optimized_rf_regression_data,2) - 1

    plot(1:size(optimized_rf_regression_data,1), ...
        optimized_rf_regression_data(:,i),'color',rand(1,3));
    hold on;
    
end

grid on;

saveas(figure(4),'policy weights under RF regression.fig');
saveas(figure(4),'policy weights under RF regression.bmp');

figure(5);

title('Policy weights under NN regression', ...
    'FontSize',18,'FontWeight','bold');
hold on;
xlabel('Iteration index','FontSize',18,'FontWeight','bold');
hold on;
ylabel('Weight value','FontSize',18,'FontWeight','bold');
hold on;
xlim([1,size(optimized_nn_regression_data,1)]);
ylim([-10,10]);
hold on;

for i = 1:size(optimized_nn_regression_data,2) - 1

    plot(1:size(optimized_nn_regression_data,1), ...
        optimized_nn_regression_data(:,i),'color',rand(1,3));
    hold on;
    
end

grid on;

saveas(figure(5),'policy weights under NN regression.fig');
saveas(figure(5),'policy weights under NN regression.bmp');

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('biggest_cluster_data');
load('regression_data');
load('synthetic_lats');
load('synthetic_lons');

number_of_iterations = 900;
number_of_random_points = 1000;
number_of_nodes = size(biggest_cluster_connectivities{1},1);
number_of_data_segments = 150;

%   Bayesian optimization using batch-based RF regression:

%   Notice that the data batch used includes 100 samples. If a bigger
%   data batch is needed, then we need the "regression_data" found 
%   previously to have the larger number of samples needed beforehand.

optimized_brf_regression_data = regression_data;
optimized_brf_execution_times = zeros(number_of_iterations,1);

for iteration = 1:number_of_iterations
    
    for i = 1:10
        
        tic;
        
        for j = 1:10
            
            tree_model{j} = ...
                fitrtree(optimized_brf_regression_data ...
                (iteration:99 + iteration,1:4), ...
                optimized_brf_regression_data ...
                (iteration:99 + iteration,4 + 1), ...
                'MinLeafSize',j);
            
        end
        
        optimized_brf_execution_times(iteration) = ...
            ((i - 1) * optimized_brf_execution_times(iteration) + toc) / i;

    end

    %   Using UCB:
    
    random_data = ...
        zeros(number_of_random_points,4 + 1);

    for i = 1:number_of_random_points
        
        for j = 1:4
            
            average = ...
                mean(optimized_brf_regression_data ...
                (iteration:99 + iteration,j));
            
            standard_deviation = ...
                std(optimized_brf_regression_data ...
                (iteration:99 + iteration,j));

            min_batch_omega = ...
                round(max(average - standard_deviation,-10));
            
            max_batch_omega = ...
                round(min(average + standard_deviation,10));
            
            omegas(j,1) = ...
                randi([min_batch_omega,max_batch_omega]);
                        
        end

        random_data(i,1:4) = omegas;
        
        for j = 1:10
            
            tree_predictions(j) = ...
                predict(tree_model{j},random_data(i,1:4));
            
        end
        
        prediction = mean(tree_predictions);
        
        standard_deviation = ...
            sqrt(1 / 10 * sum(tree_predictions .^ 2) - prediction ^ 2);
        
        random_data(i,4 + 1) = ...
            round(prediction + standard_deviation);

    end
    
    best_random_point = random_data(random_data(:,4 + 1) == ...
        max(random_data(:,4 + 1)),:);

    %   Dividing data:
    
    initial_node_data_sizes = divideData(number_of_nodes, ...
        period_start_time, period_end_time, biggest_cluster_connectivities, ...
        [number_of_data_segments], 1));
    
    %   Allocating data:
    
    segment_allocations = ...
        allocateData(number_of_nodes, initial_node_data_sizes);
    
    %   Specifying the best policy weights found so far:
    
    omegas = best_random_point(1,1:4)';
    
    %   Following the specified policy weights:
   
    [segment_allocations, actions_history, node_statuses] = ...
        followPolicy(period_start_time, period_end_time, number_of_nodes, ...
        biggest_cluster_connectivities, segment_allocations, omegas, ...
        synthetic_lats, synthetic_lons, biggest_cluster_members);      

    optimized_brf_regression_data = vertcat(optimized_brf_regression_data, ...
        [omegas',length(actions_history)]); %#ok<AGROW>
        
    progress = ...
        sprintf('%.0f percent',iteration / number_of_iterations * 100);
    clc;
    disp(progress);
        
end

save('optimized_brf_data.mat','optimized_brf_regression_data', ...
    'optimized_brf_execution_times');

%%

clear all;
clc;

load('optimized_rf_data');
load('optimized_brf_data');

%   Cleaning the execution-time results of the RF regression,
%   and the batch-based RF regression used for Bayesian optimization:

%   Removing outliers:

outlier_removing_window = 10;

for i = outlier_removing_window + 1: ...
        length(optimized_rf_execution_times) - outlier_removing_window
    
    if (optimized_rf_execution_times(i,1) > ...
            prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) + ...
            1.5 * (prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) - ...
            prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25))) || ...
            (optimized_rf_execution_times(i,1) < ...
            prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25) - ...
            1.5 * (prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) - ...
            prctile(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25)))
        
        optimized_rf_execution_times(i,1) = ...
            mean(optimized_rf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1));
        
    end
    
end

for i = outlier_removing_window + 1: ...
        length(optimized_brf_execution_times) - outlier_removing_window

    if (optimized_brf_execution_times(i,1) > ...
            prctile(optimized_brf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) + ...
            1.5 * (prctile(optimized_brf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) - ...
            prctile(optimized_brf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25))) || ...
            (optimized_brf_execution_times(i,1) < ...
            prctile(optimized_brf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25) - ...
            1.5 * (prctile(optimized_brf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),75) - ...
            prctile(optimized_brf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1),25)))
        
        optimized_brf_execution_times(i,1) = ...
            mean(optimized_brf_execution_times ...
            (i - outlier_removing_window:i + outlier_removing_window,1));
        
    end
    
end

%   Smoothing data:

moving_average_window = 10;

for i = moving_average_window + 1: ...
        length(optimized_rf_execution_times) - moving_average_window
    
    optimized_rf_execution_times(i,1) = ...
        mean(optimized_rf_execution_times ...
        (i - moving_average_window:i + moving_average_window,1));
    
end

for i = moving_average_window + 1: ...
        length(optimized_brf_execution_times) - moving_average_window
    
    optimized_brf_execution_times(i,1) = ...
        mean(optimized_brf_execution_times ...
        (i - moving_average_window:i + moving_average_window,1));
    
end

optimized_rf_smoothed_execution_times = ...
    optimized_rf_execution_times;

optimized_brf_smoothed_execution_times = ...
    optimized_brf_execution_times;

save('optimized_rf_smoothed_execution_times.mat', ...
    'optimized_rf_smoothed_execution_times');

save('optimized_brf_smoothed_execution_times.mat', ...
    'optimized_brf_smoothed_execution_times');

%%

clear all;
clc;

load('optimized_rf_data');
load('optimized_brf_data');
load('optimized_rf_smoothed_execution_times');
load('optimized_brf_smoothed_execution_times');

%   Comparing the all-data RF regression and
%   the batch-based RF regression used for Bayesian optimization:

figure(1);

title('Bayesian optimization performances', ...
    'FontSize',18,'FontWeight','bold');
hold on;
xlabel('Iteration index', ...
    'FontSize',18,'FontWeight','bold');
hold on;
ylabel('Number of data segments exchanged', ...
    'FontSize',18,'FontWeight','bold');
hold on;
plot(1:size(optimized_rf_regression_data,1), ...
    optimized_rf_regression_data ...
    (:,end),'-ob','LineWidth',1.5);
hold on;
plot(1:size(optimized_rf_regression_data,1), ...
    optimized_brf_regression_data ...
    (1:size(optimized_rf_regression_data,1),end),'-*k','LineWidth',1.5);
hold on;
xlim([0,size(optimized_rf_regression_data,1)]);
ylim([min( ...
    [min(optimized_rf_regression_data(:,end)), ...
    min(optimized_brf_regression_data(:,end))]) * 0.9, ...
    max( ...
    [max(optimized_rf_regression_data(:,end)), ...
    max(optimized_brf_regression_data(:,end))]) * 1.1]);
hold on;
legend({'w/ Random-forest-regression', ...
    'w/ Batch-based random-forest-regression'}, ...
    'Location','Best','FontSize',16);
hold on;
grid on;

saveas(figure(1),'rf vs. brf - data segments exchanged.fig');
saveas(figure(1),'rf vs. brf - data segments exchanged.bmp');

figure(2);

title('Bayesian optimization performances', ...
    'FontSize',18,'FontWeight','bold');
hold on;
xlabel('Number of observations', ...
    'FontSize',18,'FontWeight','bold');
hold on;
ylabel('Execution time (msec)', ...
    'FontSize',18,'FontWeight','bold');
hold on;
plot(1:size(optimized_rf_execution_times,1), ...
    mean(optimized_rf_smoothed_execution_times,2) .* 1000,'-*k','LineWidth',1.5);
hold on;
plot(1:size(optimized_brf_execution_times,1), ...
    mean(optimized_brf_smoothed_execution_times,2) .* 1000,'--r','LineWidth',1.5);
hold on;
xlim([0,size(optimized_rf_execution_times,1)]);
ylim([min( ...
    [min(mean(optimized_rf_smoothed_execution_times,2)), ...
    min(mean(optimized_brf_smoothed_execution_times,2))]) * 1000 * 0.9, ...
    max( ...
    [max(mean(optimized_rf_smoothed_execution_times,2)), ...
    max(mean(optimized_brf_smoothed_execution_times,2))]) * 1000 * 1.1]);
hold on;
legend({'w/ Random-forest-regression', ...
    'w/ Batch-based random-forest-regression'}, ...
    'Location','Best','FontSize',16);
hold on;
grid on;

saveas(figure(2),'rf vs. brf - execution times.fig');
saveas(figure(2),'rf vs. brf - execution times.bmp');

figure(3);

title('Policy weights under RF regression', ...
    'FontSize',18,'FontWeight','bold');
hold on;
xlabel('Iteration index','FontSize',18,'FontWeight','bold');
hold on;
ylabel('Weight value','FontSize',18,'FontWeight','bold');
hold on;
xlim([1,size(optimized_rf_regression_data,1)]);
ylim([-10,10]);
hold on;

for i = 1:size(optimized_rf_regression_data,2) - 1

    plot(1:size(optimized_rf_regression_data,1), ...
        optimized_rf_regression_data(:,i),'color',rand(1,3));
    hold on;
    
end

grid on;

saveas(figure(3),'policy weights under RF regression.fig');
saveas(figure(3),'policy weights under RF regression.bmp');

figure(4);

title('Policy weights under batch-based RF regression', ...
    'FontSize',18,'FontWeight','bold');
hold on;
xlabel('Iteration index','FontSize',18,'FontWeight','bold');
hold on;
ylabel('Weight value','FontSize',18,'FontWeight','bold');
hold on;
xlim([1,size(optimized_rf_regression_data,1)]);
ylim([-10,10]);
hold on;

for i = 1:size(optimized_brf_regression_data,2) - 1

    plot(1:size(optimized_brf_regression_data,1), ...
        optimized_brf_regression_data(:,i),'color',rand(1,3));
    hold on;
    
end

grid on;

saveas(figure(4),'policy weights under batch-based RF regression.fig');
saveas(figure(4),'policy weights under batch-based RF regression.bmp');

%%

clear all;
clc;

load('biggest_cluster_data');

load('optimized_gp_data');
load('optimized_rf_data');
load('optimized_nn_data');
load('optimized_brf_data');

load('modified_data');
load('synthetic_lats');
load('synthetic_lons');

number_of_nodes = size(biggest_cluster_connectivities{1},1);
number_of_data_segments = 150;

%   Comparing worst-policy, naive-policy and best-policy performances: 

worst_gp_performance = ...
    optimized_gp_regression_data( ...
    optimized_gp_regression_data(:,end) == ...
    min(optimized_gp_regression_data(:,end)),:);
worst_gp_performance = worst_gp_performance(1,:);

worst_rf_performance = ...
    optimized_rf_regression_data( ...
    optimized_rf_regression_data(:,end) == ...
    min(optimized_rf_regression_data(:,end)),:);
worst_rf_performance = worst_rf_performance(1,:);

worst_nn_performance = ...
    optimized_nn_regression_data( ...
    optimized_nn_regression_data(:,end) == ...
    min(optimized_nn_regression_data(:,end)),:);
worst_nn_performance = worst_nn_performance(1,:);

worst_brf_performance = ...
    optimized_brf_regression_data( ...
    optimized_brf_regression_data(:,end) == ...
    min(optimized_brf_regression_data(:,end)),:);
worst_brf_performance = worst_brf_performance(1,:);

worst_performances = ...
    vertcat(worst_gp_performance, ...
    worst_rf_performance, ...
    worst_nn_performance, ...
    worst_brf_performance);

worst_policy_omegas = ...
    worst_performances( ...
    worst_performances(:,end) == min(worst_performances(:,end)),1:end - 1);

naive_policy_omegas = [10,0,0,0];

best_gp_performance = ...
    optimized_gp_regression_data( ...
    optimized_gp_regression_data(:,end) == ...
    max(optimized_gp_regression_data(:,end)),:);
best_gp_performance = best_gp_performance(1,:);

best_rf_performance = ...
    optimized_rf_regression_data( ...
    optimized_rf_regression_data(:,end) == ...
    max(optimized_rf_regression_data(:,end)),:);
best_rf_performance = best_rf_performance(1,:);

best_nn_performance = ...
    optimized_nn_regression_data( ...
    optimized_nn_regression_data(:,end) == ...
    max(optimized_nn_regression_data(:,end)),:);
best_nn_performance = best_nn_performance(1,:);

best_brf_performance = ...
    optimized_brf_regression_data( ...
    optimized_brf_regression_data(:,end) == ...
    max(optimized_brf_regression_data(:,end)),:);
best_brf_performance = best_brf_performance(1,:);

best_performances = ...
    vertcat(best_gp_performance, ...
    best_rf_performance, ...
    best_nn_performance, ...
    best_brf_performance);

best_policy_omegas = ...
    best_performances( ...
    best_performances(:,end) == max(best_performances(:,end)),1:end - 1);

%   Dividing data:
    
initial_node_data_sizes = divideData(number_of_nodes, ...
	period_start_time, period_end_time, biggest_cluster_connectivities, ...
	[number_of_data_segments], 1));

%   Allocating data:
    
initial_segment_allocations = ...
	allocateData(number_of_nodes, initial_node_data_sizes);

%   Worst-policy simulation:

%   Specifying the worst policy weights found so far:
    
omegas = worst_policy_omegas';
    
%   Following the specified policy weights:

final_segment_allocations = initial_segment_allocations;

[final_segment_allocations, worst_policy_actions_history, node_statuses] = ...
	followPolicy(period_start_time, period_end_time, number_of_nodes, ...
	biggest_cluster_connectivities, final_segment_allocations, omegas, ...
	synthetic_lats, synthetic_lons, biggest_cluster_members); 

worst_policy_segment_allocations = final_segment_allocations;

%   Naive-policy simulation:

%   Specifying the naive policy weights:
    
omegas = naive_policy_omegas';
    
%   Following the specified policy weights:

final_segment_allocations = initial_segment_allocations;

[final_segment_allocations, naive_policy_actions_history, node_statuses] = ...
	followPolicy(period_start_time, period_end_time, number_of_nodes, ...
	biggest_cluster_connectivities, final_segment_allocations, omegas, ...
	synthetic_lats, synthetic_lons, biggest_cluster_members); 

naive_policy_segment_allocations = final_segment_allocations;

%   Best-policy simulation:

%   Specifying the best policy weights found so far:
    
omegas = best_policy_omegas';
    
%   Following the specified policy weights:

final_segment_allocations = initial_segment_allocations;

[final_segment_allocations, best_policy_actions_history, node_statuses] = ...
	followPolicy(period_start_time, period_end_time, number_of_nodes, ...
	biggest_cluster_connectivities, final_segment_allocations, omegas, ...
	synthetic_lats, synthetic_lons, biggest_cluster_members); 

best_policy_segment_allocations = final_segment_allocations;

%   Visualizing the different policy performances:

worst_policy_animation = ...
    VideoWriter('worst_policy_animation.avi');
worst_policy_animation.FrameRate = 10;
worst_policy_animation.Quality = 75;
open(worst_policy_animation);

for t = period_start_time * 6:period_end_time * 6

    figure(1);
    
    clf;

    title('Data exchages under worst policy','FontSize',20);
    hold on
    
    map = imread('policies_map.jpg');
    image('CData',map, ...
        'XData', ...
        [min(min(synthetic_lons( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6))) ...
        max(max(synthetic_lons( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6)))], ...
        'YData', ...
        [min(min(synthetic_lats( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6))) ...
        max(max(synthetic_lats( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6)))])
    hold on
    
    plot(synthetic_lons(biggest_cluster_members,t), ...
        synthetic_lats(biggest_cluster_members,t), ...
        'k.','MarkerSize',16);
    hold on
    
    for i = 1:size(biggest_cluster_members,2)
    
        relevant_actions_history = ...
            worst_policy_actions_history ...
            (worst_policy_actions_history(:,1) == t,:);

        relevant_actions_history = ...
            relevant_actions_history( ...
            relevant_actions_history(:,2) == i,:);
    
        if ~ isempty(relevant_actions_history)
          
            transmitter = i;
            
            plot( ...
                synthetic_lons( ...
                biggest_cluster_members(transmitter),t), ...
                synthetic_lats( ...
                biggest_cluster_members(transmitter),t), ...
                'r.','MarkerSize',16);
            hold on
            
            %   Displaying coverage areas:
            
            broadcasting_range = ...
                relevant_actions_history(1,3) / 1000;
            earth_radius = 6371;

            transmitter_lon = synthetic_lons( ...
                biggest_cluster_members(transmitter),t);
            transmitter_lat = synthetic_lats( ...
                biggest_cluster_members(transmitter),t);
            
            transmitter_x = earth_radius * ...
                cos(transmitter_lat * pi / 180) * ...
                cos(transmitter_lon * pi / 180);
            transmitter_y = earth_radius * ...
                cos(transmitter_lat * pi / 180) * ...
                sin(transmitter_lon * pi / 180);
            
            transmitter_xs = transmitter_x + ...
                broadcasting_range .* cos(0:pi / 50:2 * pi) ;
            transmitter_ys = transmitter_y + ...
                broadcasting_range .* sin(0:pi / 50:2 * pi);
            
            transmitter_lons = ...
                atand(transmitter_ys ./ transmitter_xs);
            transmitter_lats = acosd(transmitter_ys ./ ...
                (earth_radius .* sind(transmitter_lons)));

            %   Correcting coverage areas:
            
            for j = 1:length(transmitter_lons)
                
                distance = ...
                    lldistkm([transmitter_lat,transmitter_lon], ...
                    [transmitter_lats(j),transmitter_lons(j)]);
                
                if distance > broadcasting_range
                                        
                    theta = ...
                        atan2((transmitter_ys(j) - transmitter_y), ...
                        (transmitter_xs(j) - transmitter_x));
                    
                    transmitter_xs(j) = ...
                        transmitter_x + ...
                        broadcasting_range / distance * broadcasting_range * cos(theta);                    
                    transmitter_ys(j) = ...
                        transmitter_y + ...
                        broadcasting_range / distance * broadcasting_range * sin(theta);
                    
                    transmitter_lons(j) = ...
                        atand(transmitter_ys(j) ./ transmitter_xs(j));
                    transmitter_lats(j) = acosd(transmitter_ys(j) ./ ...
                        (earth_radius .* sind(transmitter_lons(j))));
                    
                end
                
            end
            
            plot(transmitter_lons,transmitter_lats,'-r');
            hold on

            for j = 1:size(relevant_actions_history,1)
                
                receiver = find(relevant_actions_history ...
                    (j,4 + 1:end) == 1);
                
                line( ...
                    [synthetic_lons( ...
                    biggest_cluster_members(transmitter),t), ...
                    synthetic_lons( ...
                    biggest_cluster_members(receiver),t)], ...
                    [synthetic_lats( ...
                    biggest_cluster_members(transmitter),t), ...
                    synthetic_lats( ...
                    biggest_cluster_members(receiver),t)], ...
                    'Color','r','LineWidth',2);
                hold on
                
                plot( ...
                    synthetic_lons( ...
                    biggest_cluster_members(receiver),t), ...
                    synthetic_lats( ...
                    biggest_cluster_members(receiver),t), ...
                    'g.','MarkerSize',16);
                hold on
                
            end
            
        end
        
    end
    
    axis([min(min(synthetic_lons( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6))) ...
        max(max(synthetic_lons( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6))) ...
        min(min(synthetic_lats( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6))) ...
        max(max(synthetic_lats( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6)))]);
    ylabel('Latitude','FontSize',20);
    xlabel('Longitude','FontSize',20);
    hold on
    
    time = ...
        sprintf('%02d:%02d',floor(t / (60 * 6)), ...
        floor((t / (60 * 6) - floor(t / (60 * 6))) * 60));
    text(-80.45,43.505, ...
        time,'Color','black','FontSize',20,'FontWeight','bold')
    hold on;
    
    grid on;
        
    writeVideo(worst_policy_animation, ...
        getframe(figure(1)));

end

close(worst_policy_animation);

best_policy_animation = ...
    VideoWriter('best_policy_animation.avi');
best_policy_animation.FrameRate = 10;
best_policy_animation.Quality = 75;
open(best_policy_animation);

for t = period_start_time * 6:period_end_time * 6
    
    figure(2);
    
    clf;
    
    title('Data exchages under best policy','FontSize',20);
    hold on
    
    map = imread('policies_map.jpg');
    image('CData',map, ...
        'XData', ...
        [min(min(synthetic_lons( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6))) ...
        max(max(synthetic_lons( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6)))], ...
        'YData', ...
        [min(min(synthetic_lats( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6))) ...
        max(max(synthetic_lats( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6)))])
    hold on
    
    plot(synthetic_lons(biggest_cluster_members,t), ...
        synthetic_lats(biggest_cluster_members,t), ...
        'k.','MarkerSize',16);
    hold on
    
    for i = 1:size(biggest_cluster_members,2)

        relevant_actions_history = ...
            best_policy_actions_history ...
            (best_policy_actions_history(:,1) == t,:);

        relevant_actions_history = ...
            relevant_actions_history( ...
            relevant_actions_history(:,2) == i,:);
    
        if ~ isempty(relevant_actions_history)
          
            transmitter = i;
            
            plot( ...
                synthetic_lons( ...
                biggest_cluster_members(transmitter),t), ...
                synthetic_lats( ...
                biggest_cluster_members(transmitter),t), ...
                'r.','MarkerSize',16);
            hold on
            
            %   Displaying coverage areas:
            
            broadcasting_range = ...
                relevant_actions_history(1,3) / 1000;
            earth_radius = 6371;

            transmitter_lon = synthetic_lons( ...
                biggest_cluster_members(transmitter),t);
            transmitter_lat = synthetic_lats( ...
                biggest_cluster_members(transmitter),t);
            
            transmitter_x = earth_radius * ...
                cos(transmitter_lat * pi / 180) * ...
                cos(transmitter_lon * pi / 180);
            transmitter_y = earth_radius * ...
                cos(transmitter_lat * pi / 180) * ...
                sin(transmitter_lon * pi / 180);
            
            transmitter_xs = transmitter_x + ...
                broadcasting_range .* cos(0:pi / 50:2 * pi) ;
            transmitter_ys = transmitter_y + ...
                broadcasting_range .* sin(0:pi / 50:2 * pi);
            
            transmitter_lons = ...
                atand(transmitter_ys ./ transmitter_xs);
            transmitter_lats = acosd(transmitter_ys ./ ...
                (earth_radius .* sind(transmitter_lons)));

            %   Correcting coverage areas:
            
            for j = 1:length(transmitter_lons)
                
                distance = ...
                    lldistkm([transmitter_lat,transmitter_lon], ...
                    [transmitter_lats(j),transmitter_lons(j)]);
                
                if distance > broadcasting_range
                                        
                    theta = ...
                        atan2((transmitter_ys(j) - transmitter_y), ...
                        (transmitter_xs(j) - transmitter_x));
                    
                    transmitter_xs(j) = ...
                        transmitter_x + ...
                        broadcasting_range / distance * broadcasting_range * cos(theta);                    
                    transmitter_ys(j) = ...
                        transmitter_y + ...
                        broadcasting_range / distance * broadcasting_range * sin(theta);
                    
                    transmitter_lons(j) = ...
                        atand(transmitter_ys(j) ./ transmitter_xs(j));
                    transmitter_lats(j) = acosd(transmitter_ys(j) ./ ...
                        (earth_radius .* sind(transmitter_lons(j))));
                    
                end
                
            end
            
            plot(transmitter_lons,transmitter_lats,'-r');
            hold on
            
            for j = 1:size(relevant_actions_history,1)
                
                receiver = find(relevant_actions_history ...
                    (j,4 + 1:end) == 1);
                
                line( ...
                    [synthetic_lons( ...
                    biggest_cluster_members(transmitter),t), ...
                    synthetic_lons( ...
                    biggest_cluster_members(receiver),t)], ...
                    [synthetic_lats( ...
                    biggest_cluster_members(transmitter),t), ...
                    synthetic_lats( ...
                    biggest_cluster_members(receiver),t)], ...
                    'Color','r','LineWidth',2);
                hold on
                
                plot( ...
                    synthetic_lons( ...
                    biggest_cluster_members(receiver),t), ...
                    synthetic_lats( ...
                    biggest_cluster_members(receiver),t), ...
                    'g.','MarkerSize',16);
                hold on
                
            end
            
        end
        
    end
    
    axis([min(min(synthetic_lons( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6))) ...
        max(max(synthetic_lons( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6))) ...
        min(min(synthetic_lats( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6))) ...
        max(max(synthetic_lats( ...
        biggest_cluster_members,period_start_time * 6:period_end_time * 6)))]);
    ylabel('Latitude','FontSize',20);
    xlabel('Longitude','FontSize',20);
    hold on
    
    time = ...
        sprintf('%02d:%02d',floor(t / (60 * 6)), ...
        floor((t / (60 * 6) - floor(t / (60 * 6))) * 60));
    text(-80.45,43.505, ...
        time,'Color','black','FontSize',20,'FontWeight','bold')
    hold on;
    
    grid on;

    drawnow;
    
    writeVideo(best_policy_animation, ...
        getframe(figure(2)));

end

close(best_policy_animation);

zoomed_in_best_policy_animation = ...
    VideoWriter('zoomed_in_best_policy_animation.avi');
zoomed_in_best_policy_animation.FrameRate = 10;
zoomed_in_best_policy_animation.Quality = 75;
open(zoomed_in_best_policy_animation);

for t = period_start_time * 6:period_end_time * 6
    
    figure(3);
    
    clf;
    
    title('Zoomed-in data exchages under best policy','FontSize',20);
    hold on
    
    map = imread('zoomed_in_policies_map.jpg');
    image('CData',map, ...
        'XData',[-80.5 -80.48], ...
        'YData',[43.44 43.46])
    hold on
    
    plot(synthetic_lons(biggest_cluster_members,t), ...
        synthetic_lats(biggest_cluster_members,t), ...
        'k.','MarkerSize',16);
    hold on
    
    for i = 1:size(biggest_cluster_members,2)

        relevant_actions_history = ...
            best_policy_actions_history ...
            (best_policy_actions_history(:,1) == t,:);

        relevant_actions_history = ...
            relevant_actions_history( ...
            relevant_actions_history(:,2) == i,:);
    
        if ~ isempty(relevant_actions_history)
          
            transmitter = i;
            
            plot( ...
                synthetic_lons( ...
                biggest_cluster_members(transmitter),t), ...
                synthetic_lats( ...
                biggest_cluster_members(transmitter),t), ...
                'r.','MarkerSize',16);
            hold on
            
            %   Displaying coverage areas:
            
            broadcasting_range = ...
                relevant_actions_history(1,3) / 1000;
            earth_radius = 6371;

            transmitter_lon = synthetic_lons( ...
                biggest_cluster_members(transmitter),t);
            transmitter_lat = synthetic_lats( ...
                biggest_cluster_members(transmitter),t);
            
            transmitter_x = earth_radius * ...
                cos(transmitter_lat * pi / 180) * ...
                cos(transmitter_lon * pi / 180);
            transmitter_y = earth_radius * ...
                cos(transmitter_lat * pi / 180) * ...
                sin(transmitter_lon * pi / 180);
            
            transmitter_xs = transmitter_x + ...
                broadcasting_range .* cos(0:pi / 50:2 * pi) ;
            transmitter_ys = transmitter_y + ...
                broadcasting_range .* sin(0:pi / 50:2 * pi);
            
            transmitter_lons = ...
                atand(transmitter_ys ./ transmitter_xs);
            transmitter_lats = acosd(transmitter_ys ./ ...
                (earth_radius .* sind(transmitter_lons)));

            %   Correcting coverage areas:
            
            for j = 1:length(transmitter_lons)
                
                distance = ...
                    lldistkm([transmitter_lat,transmitter_lon], ...
                    [transmitter_lats(j),transmitter_lons(j)]);
                
                if distance > broadcasting_range
                                        
                    theta = ...
                        atan2((transmitter_ys(j) - transmitter_y), ...
                        (transmitter_xs(j) - transmitter_x));
                    
                    transmitter_xs(j) = ...
                        transmitter_x + ...
                        broadcasting_range / distance * broadcasting_range * cos(theta);                    
                    transmitter_ys(j) = ...
                        transmitter_y + ...
                        broadcasting_range / distance * broadcasting_range * sin(theta);
                    
                    transmitter_lons(j) = ...
                        atand(transmitter_ys(j) ./ transmitter_xs(j));
                    transmitter_lats(j) = acosd(transmitter_ys(j) ./ ...
                        (earth_radius .* sind(transmitter_lons(j))));
                    
                end
                
            end
            
            plot(transmitter_lons,transmitter_lats,'-r');
            hold on
            
            for j = 1:size(relevant_actions_history,1)
                
                receiver = find(relevant_actions_history ...
                    (j,4 + 1:end) == 1);
                
                line( ...
                    [synthetic_lons( ...
                    biggest_cluster_members(transmitter),t), ...
                    synthetic_lons( ...
                    biggest_cluster_members(receiver),t)], ...
                    [synthetic_lats( ...
                    biggest_cluster_members(transmitter),t), ...
                    synthetic_lats( ...
                    biggest_cluster_members(receiver),t)], ...
                    'Color','r','LineWidth',2);
                hold on
                
                plot( ...
                    synthetic_lons( ...
                    biggest_cluster_members(receiver),t), ...
                    synthetic_lats( ...
                    biggest_cluster_members(receiver),t), ...
                    'g.','MarkerSize',16);
                hold on
                
            end
            
        end
         
    end
    
    axis([-80.5 -80.48 43.44 43.46]);
    ylabel('Latitude','FontSize',20);
    xlabel('Longitude','FontSize',20);
    hold on
    
    time = ...
        sprintf('%02d:%02d',floor(t / (60 * 6)), ...
        floor((t / (60 * 6) - floor(t / (60 * 6))) * 60));
    text(-80.483,43.4585, ...
        time,'Color','black','FontSize',20,'FontWeight','bold')
    hold on;
    
    grid on;

    drawnow;
        
    writeVideo(zoomed_in_best_policy_animation, ...
        getframe(figure(3)));

end

close(zoomed_in_best_policy_animation);

figure(4);

title('Data exchages under best policy @ 5:00 PM','FontSize',20);
hold on

map = imread('policies_map.jpg');
image('CData',map, ...
    'XData', ...
    [min(synthetic_lons( ...
    biggest_cluster_members,17 * 60 * 6 + 4)) ...
    max(synthetic_lons( ...
    biggest_cluster_members,17 * 60 * 6 + 4))], ...
    'YData', ...
    [min(synthetic_lats( ...
    biggest_cluster_members,17 * 60 * 6 + 4)) ...
    max(synthetic_lats( ...
    biggest_cluster_members,17 * 60 * 6 + 4))])
hold on

plot(synthetic_lons(biggest_cluster_members,17 * 60 * 6 + 4), ...
    synthetic_lats(biggest_cluster_members,17 * 60 * 6 + 4), ...
    'k.','MarkerSize',16);
hold on

for i = 1:size(biggest_cluster_members,2)
    
    relevant_actions_history = ...
        best_policy_actions_history ...
        (best_policy_actions_history(:,1) == 17 * 60 * 6 + 4,:);
    
    relevant_actions_history = ...
        relevant_actions_history( ...
        relevant_actions_history(:,2) == i,:);
    
    if ~ isempty(relevant_actions_history)
        
        transmitter = i;
        
        plot( ...
            synthetic_lons( ...
            biggest_cluster_members(transmitter),17 * 60 * 6 + 4), ...
            synthetic_lats( ...
            biggest_cluster_members(transmitter),17 * 60 * 6 + 4), ...
            'r.','MarkerSize',16);
        hold on
        
        %   Displaying coverage areas:
        
        broadcasting_range = ...
            relevant_actions_history(1,3) / 1000;
        earth_radius = 6371;
        
        transmitter_lon = synthetic_lons( ...
            biggest_cluster_members(transmitter),17 * 60 * 6 + 4);
        transmitter_lat = synthetic_lats( ...
            biggest_cluster_members(transmitter),17 * 60 * 6 + 4);
        
        transmitter_x = earth_radius * ...
            cos(transmitter_lat * pi / 180) * ...
            cos(transmitter_lon * pi / 180);
        transmitter_y = earth_radius * ...
            cos(transmitter_lat * pi / 180) * ...
            sin(transmitter_lon * pi / 180);
        
        transmitter_xs = transmitter_x + ...
            broadcasting_range .* cos(0:pi / 50:2 * pi) ;
        transmitter_ys = transmitter_y + ...
            broadcasting_range .* sin(0:pi / 50:2 * pi);
        
        transmitter_lons = ...
            atand(transmitter_ys ./ transmitter_xs);
        transmitter_lats = acosd(transmitter_ys ./ ...
            (earth_radius .* sind(transmitter_lons)));
        
        %   Correcting coverage areas:
        
        for j = 1:length(transmitter_lons)
            
            distance = ...
                lldistkm([transmitter_lat,transmitter_lon], ...
                [transmitter_lats(j),transmitter_lons(j)]);
            
            if distance > broadcasting_range
                
                theta = ...
                    atan2((transmitter_ys(j) - transmitter_y), ...
                    (transmitter_xs(j) - transmitter_x));
                
                transmitter_xs(j) = ...
                    transmitter_x + ...
                    broadcasting_range / distance * broadcasting_range * cos(theta);
                transmitter_ys(j) = ...
                    transmitter_y + ...
                    broadcasting_range / distance * broadcasting_range * sin(theta);
                
                transmitter_lons(j) = ...
                    atand(transmitter_ys(j) ./ transmitter_xs(j));
                transmitter_lats(j) = acosd(transmitter_ys(j) ./ ...
                    (earth_radius .* sind(transmitter_lons(j))));
                
            end
            
        end
        
        plot(transmitter_lons,transmitter_lats,'-r','LineWidth',2);
        hold on
        
        for j = 1:size(relevant_actions_history,1)
            
            receiver = find(relevant_actions_history ...
                (j,4 + 1:end) == 1);
            
            line( ...
                [synthetic_lons( ...
                biggest_cluster_members(transmitter),17 * 60 * 6 + 4), ...
                synthetic_lons( ...
                biggest_cluster_members(receiver),17 * 60 * 6 + 4)], ...
                [synthetic_lats( ...
                biggest_cluster_members(transmitter),17 * 60 * 6 + 4), ...
                synthetic_lats( ...
                biggest_cluster_members(receiver),17 * 60 * 6 + 4)], ...
                'Color','r','LineWidth',2);
            hold on
            
            plot( ...
                synthetic_lons( ...
                biggest_cluster_members(receiver),17 * 60 * 6 + 4), ...
                synthetic_lats( ...
                biggest_cluster_members(receiver),17 * 60 * 6 + 4), ...
                'g.','MarkerSize',16);
            hold on
            
        end
        
    end
    
end

axis([min(synthetic_lons( ...
    biggest_cluster_members,17 * 60 * 6 + 4)) ...
    max(synthetic_lons( ...
    biggest_cluster_members,17 * 60 * 6 + 4)) ...
    min(synthetic_lats( ...
    biggest_cluster_members,17 * 60 * 6 + 4)) ...
    max(synthetic_lats( ...
    biggest_cluster_members,17 * 60 * 6 + 4))]);
ylabel('Latitude','FontSize',20);
xlabel('Longitude','FontSize',20);
hold on

text(-80.45,43.505, ...
    '17:00','Color','black','FontSize',20,'FontWeight','bold')
hold on;

grid on;

saveas(figure(4),'data exchanges at 5.fig');
saveas(figure(4),'data exchanges at 5.bmp');

figure(5);

figure_title = ...
    sprintf('Zoomed-in data exchages\nunder best policy @ 5:00 PM');
title(figure_title,'FontSize',20);
hold on

map = imread('zoomed_in_policies_map.jpg');
image('CData',map, ...
    'XData',[-80.5 -80.48], ...
    'YData',[43.44 43.46])
hold on

plot(synthetic_lons(biggest_cluster_members,17 * 60 * 6 + 4), ...
    synthetic_lats(biggest_cluster_members,17 * 60 * 6 + 4), ...
    'k.','MarkerSize',16);
hold on

for i = 1:size(biggest_cluster_members,2)
    
    relevant_actions_history = ...
        best_policy_actions_history ...
        (best_policy_actions_history(:,1) == 17 * 60 * 6 + 4,:);
    
    relevant_actions_history = ...
        relevant_actions_history( ...
        relevant_actions_history(:,2) == i,:);
    
    if ~ isempty(relevant_actions_history)
        
        transmitter = i;
        
        plot( ...
            synthetic_lons( ...
            biggest_cluster_members(transmitter),17 * 60 * 6 + 4), ...
            synthetic_lats( ...
            biggest_cluster_members(transmitter),17 * 60 * 6 + 4), ...
            'r.','MarkerSize',16);
        hold on
        
        %   Displaying coverage areas:
        
        broadcasting_range = ...
            relevant_actions_history(1,3) / 1000;
        earth_radius = 6371;
        
        transmitter_lon = synthetic_lons( ...
            biggest_cluster_members(transmitter),17 * 60 * 6 + 4);
        transmitter_lat = synthetic_lats( ...
            biggest_cluster_members(transmitter),17 * 60 * 6 + 4);
        
        transmitter_x = earth_radius * ...
            cos(transmitter_lat * pi / 180) * ...
            cos(transmitter_lon * pi / 180);
        transmitter_y = earth_radius * ...
            cos(transmitter_lat * pi / 180) * ...
            sin(transmitter_lon * pi / 180);
        
        transmitter_xs = transmitter_x + ...
            broadcasting_range .* cos(0:pi / 50:2 * pi) ;
        transmitter_ys = transmitter_y + ...
            broadcasting_range .* sin(0:pi / 50:2 * pi);
        
        transmitter_lons = ...
            atand(transmitter_ys ./ transmitter_xs);
        transmitter_lats = acosd(transmitter_ys ./ ...
            (earth_radius .* sind(transmitter_lons)));
        
        %   Correcting coverage areas:
        
        for j = 1:length(transmitter_lons)
            
            distance = ...
                lldistkm([transmitter_lat,transmitter_lon], ...
                [transmitter_lats(j),transmitter_lons(j)]);
            
            if distance > broadcasting_range
                
                theta = ...
                    atan2((transmitter_ys(j) - transmitter_y), ...
                    (transmitter_xs(j) - transmitter_x));
                
                transmitter_xs(j) = ...
                    transmitter_x + ...
                    broadcasting_range / distance * broadcasting_range * cos(theta);
                transmitter_ys(j) = ...
                    transmitter_y + ...
                    broadcasting_range / distance * broadcasting_range * sin(theta);
                
                transmitter_lons(j) = ...
                    atand(transmitter_ys(j) ./ transmitter_xs(j));
                transmitter_lats(j) = acosd(transmitter_ys(j) ./ ...
                    (earth_radius .* sind(transmitter_lons(j))));
                
            end
            
        end
        
        plot(transmitter_lons,transmitter_lats,'-r','LineWidth',2);
        hold on
        
        for j = 1:size(relevant_actions_history,1)
            
            receiver = find(relevant_actions_history ...
                (j,4 + 1:end) == 1);
            
            line( ...
                [synthetic_lons( ...
                biggest_cluster_members(transmitter),17 * 60 * 6 + 4), ...
                synthetic_lons( ...
                biggest_cluster_members(receiver),17 * 60 * 6 + 4)], ...
                [synthetic_lats( ...
                biggest_cluster_members(transmitter),17 * 60 * 6 + 4), ...
                synthetic_lats( ...
                biggest_cluster_members(receiver),17 * 60 * 6 + 4)], ...
                'Color','r','LineWidth',2);
            hold on
            
            plot( ...
                synthetic_lons( ...
                biggest_cluster_members(receiver),17 * 60 * 6 + 4), ...
                synthetic_lats( ...
                biggest_cluster_members(receiver),17 * 60 * 6 + 4), ...
                'g.','MarkerSize',16);
            hold on
            
        end
        
    end
    
end

axis([-80.5 -80.48 43.44 43.46]);
ylabel('Latitude','FontSize',20);
xlabel('Longitude','FontSize',20);
hold on

text(-80.483,43.4585, ...
    '17:00','Color','black','FontSize',20,'FontWeight','bold')
hold on;

grid on;

saveas(figure(5),'zoomed-in data exchanges at 5.fig');
saveas(figure(5),'zoomed-in data exchanges at 5.bmp');

figure(6);

imagesc(initial_segment_allocations);
hold on;
colorbar;
caxis([0 1]);
title('Initial data segment allocations','FontSize',20);
hold on;
xlabel('Segment index','FontSize',20);
ylabel('Node index','FontSize',20);
hold on

saveas(figure(6),'initial segment allocations.fig');
saveas(figure(6),'initial segment allocations.bmp');

figure(7);

imagesc(worst_policy_segment_allocations);
hold on;
colorbar;
caxis([0 1]);
title('Worst-policy data segment allocations','FontSize',20);
hold on;
xlabel('Segment index','FontSize',20);
ylabel('Node index','FontSize',20);
hold on

saveas(figure(7),'worst policy segment allocations.fig');
saveas(figure(7),'worst policy segment allocations.bmp');

figure(8);

imagesc(naive_policy_segment_allocations);
hold on;
colorbar;
caxis([0 1]);
title('Naive-policy data segment allocations','FontSize',20);
hold on;
xlabel('Segment index','FontSize',20);
ylabel('Node index','FontSize',20);
hold on

saveas(figure(8),'naive policy segment allocations.fig');
saveas(figure(8),'naive policy segment allocations.bmp');

figure(9);

imagesc(best_policy_segment_allocations);
hold on;
colorbar;
caxis([0 1]);
title('Best-policy data segment allocations','FontSize',20);
hold on;
xlabel('Segment index','FontSize',20);
ylabel('Node index','FontSize',20);
hold on

saveas(figure(9),'best policy segment allocations.fig');
saveas(figure(9),'best policy segment allocations.bmp');

worst_policy_transmission_times = ...
    unique(worst_policy_actions_history(:,1),'stable');
worst_policy_data_collected = ...
    zeros(size(worst_policy_transmission_times,1),1);
worst_policy_data_accumulated = ...
    zeros(size(worst_policy_transmission_times,1),1);

for i = 1:size(worst_policy_transmission_times,1)
    
    worst_policy_data_collected(i) = ...
        size(worst_policy_actions_history( ...
        worst_policy_actions_history(:,1) == worst_policy_transmission_times(i),:),1);

    worst_policy_data_accumulated(i) = ...
        sum(worst_policy_data_collected);
    
end

naive_policy_transmission_times = ...
    unique(naive_policy_actions_history(:,1),'stable');
naive_policy_data_collected = ...
    zeros(size(naive_policy_transmission_times,1),1);
naive_policy_data_accumulated = ...
    zeros(size(naive_policy_transmission_times,1),1);

for i = 1:size(naive_policy_transmission_times,1)
    
    naive_policy_data_collected(i) = ...
        size(naive_policy_actions_history( ...
        naive_policy_actions_history(:,1) == naive_policy_transmission_times(i),:),1);

    naive_policy_data_accumulated(i) = ...
        sum(naive_policy_data_collected);
    
end

best_policy_transmission_times = ...
    unique(best_policy_actions_history(:,1),'stable');
best_policy_data_collected = ...
    zeros(size(best_policy_transmission_times,1),1);
best_policy_data_accumulated = ...
    zeros(size(best_policy_transmission_times,1),1);

for i = 1:size(best_policy_transmission_times,1)
    
    best_policy_data_collected(i) = ...
        size(best_policy_actions_history( ...
        best_policy_actions_history(:,1) == best_policy_transmission_times(i),:),1);

    best_policy_data_accumulated(i) = ...
        sum(best_policy_data_collected);
    
end

figure(10);

figure_title = ...
    sprintf('Instantaneous data segments\nexchanged vs. time');
title(figure_title,'FontSize',20);
hold on

plot(worst_policy_transmission_times ./ (60 * 6), ...
    worst_policy_data_collected,'--r','LineWidth',1);
hold on;
plot(naive_policy_transmission_times ./ (60 * 6), ...
    naive_policy_data_collected,'-*b','LineWidth',1);
hold on;
plot(best_policy_transmission_times ./ (60 * 6), ...
    best_policy_data_collected,'-og','LineWidth',1);
hold on;

ylim([0 max(max(max(worst_policy_data_collected), ...
    max(naive_policy_data_collected)), ...
    max(best_policy_data_collected)) * 1.1]);
xlabel('Time (Hour)','FontSize',20);
hold on
label = ...
    sprintf('Instantaneous data\nsegments exchanged');
ylabel(label,'FontSize',20);
hold on
legend({'Worst-policy','Naive-policy','Best-policy'}, ...
    'Location','best','FontSize',20);
hold on
grid on

saveas(figure(10),'instantaneous data segments exchanged vs. time.fig');
saveas(figure(10),'instantaneous data segments exchanged vs. time.bmp');

figure(11);

figure_title = ...
    sprintf('Cumulative data segments\nexchanged vs. time');
title(figure_title,'FontSize',20);
hold on

plot(worst_policy_transmission_times ./ (60 * 6), ...
    worst_policy_data_accumulated,'--r','LineWidth',1);
hold on;
plot(naive_policy_transmission_times ./ (60 * 6), ...
    naive_policy_data_accumulated,'-*b','LineWidth',1);
hold on;
plot(best_policy_transmission_times ./ (60 * 6), ...
    best_policy_data_accumulated,'-og','LineWidth',1);
hold on;

xlabel('Time (Hour)','FontSize',20);
hold on
label = ...
    sprintf('Cumulative data\nsegments exchanged');
ylabel(label,'FontSize',20);
hold on
legend({'Worst-policy','Naive-policy','Best-policy'}, ...
    'Location','best','FontSize',20);
hold on
grid on

saveas(figure(11),'cumulative data segments exchanged vs. time.fig');
saveas(figure(11),'cumulative data segments exchanged vs. time.bmp');

data_initially_distributed_percentages = ...
    initial_node_data_sizes' ./ ...
    sum(initial_node_data_sizes) .* 100;

worst_policy_data_exchanged_percentages = ...
    sum(worst_policy_segment_allocations,2) ./ ...
    sum(initial_node_data_sizes) .* 100 - ...
    data_initially_distributed_percentages;

worst_policy_data_finally_distributed_percentages = ...
    ones(number_of_nodes,1) .* 100 - ...
    (data_initially_distributed_percentages + ...
    worst_policy_data_exchanged_percentages);

naive_policy_data_exchanged_percentages = ...
    sum(naive_policy_segment_allocations,2) ./ ...
    sum(initial_node_data_sizes) .* 100 - ...
    data_initially_distributed_percentages;

naive_policy_data_finally_distributed_percentages = ...
    ones(number_of_nodes,1) .* 100 - ...
    (data_initially_distributed_percentages + ...
    naive_policy_data_exchanged_percentages);

best_policy_data_exchanged_percentages = ...
    sum(best_policy_segment_allocations,2) ./ ...
    sum(initial_node_data_sizes) .* 100 - ...
    data_initially_distributed_percentages;

best_policy_data_finally_distributed_percentages = ...
    ones(number_of_nodes,1) .* 100 - ...
    (data_initially_distributed_percentages + ...
    best_policy_data_exchanged_percentages);

figure(12);

figure_title = ...
    sprintf('Node data segment category\npercentages under worst-policy');
title(figure_title,'FontSize',20);
hold on;
handle = bar([data_initially_distributed_percentages, ...
    worst_policy_data_exchanged_percentages, ...
    worst_policy_data_finally_distributed_percentages],'stacked');
hold on
legend_labels = cell(1,3);
legend_labels{1} = 'Initially-distributed';
legend_labels{2} = 'Exchanged';
legend_labels{3} = 'Finally-distributed';
legend(handle,legend_labels,'FontSize',20, ...
    'Location','southoutside');
hold on;
xlim([0 (number_of_nodes + 1)]);
xlabel('Node index','FontSize',20);
label = ...
    sprintf('Data segment\ncategory percentage');
ylabel(label,'FontSize',20);
hold on
grid on

saveas(figure(12),'segment category percentages under worst policy.fig');
saveas(figure(12),'segment category percentages under worst policy.bmp');

figure(13);

figure_title = ...
    sprintf('Node data segment category\npercentages under naive-policy');
title(figure_title,'FontSize',20);
hold on;
handle = bar([data_initially_distributed_percentages, ...
    naive_policy_data_exchanged_percentages, ...
    naive_policy_data_finally_distributed_percentages],'stacked');
hold on
legend_labels = cell(1,3);
legend_labels{1} = 'Initially-distributed';
legend_labels{2} = 'Exchanged';
legend_labels{3} = 'Finally-distributed';
legend(handle,legend_labels,'FontSize',20, ...
    'Location','southoutside');
hold on;
xlim([0 (number_of_nodes + 1)]);
xlabel('Node index','FontSize',20);
label = ...
    sprintf('Data segment\ncategory percentage');
ylabel(label,'FontSize',20);
hold on
grid on

saveas(figure(13),'segment category percentages under naive policy.fig');
saveas(figure(13),'segment category percentages under naive policy.bmp');

figure(14);

figure_title = ...
    sprintf('Node data segment category\npercentages under best-policy');
title(figure_title,'FontSize',20);
hold on;
handle = bar([data_initially_distributed_percentages, ...
    best_policy_data_exchanged_percentages, ...
    best_policy_data_finally_distributed_percentages],'stacked');
hold on
legend_labels = cell(1,3);
legend_labels{1} = 'Initially-distributed';
legend_labels{2} = 'Exchanged';
legend_labels{3} = 'Finally-distributed';
legend(handle,legend_labels,'FontSize',20, ...
    'Location','southoutside');
hold on;
xlim([0 (number_of_nodes + 1)]);
xlabel('Node index','FontSize',20);
label = ...
    sprintf('Data segment\ncategory percentage');
ylabel(label,'FontSize',20);
hold on
grid on

saveas(figure(14),'segment category percentages under best policy.fig');
saveas(figure(14),'segment category percentages under best policy.bmp');

data_rates = 3:3:27;

data_initially_distributed = ...
    sum(sum(initial_segment_allocations)) ...
    .* 10 .* data_rates .* 1024 ./ 8 ./ 1024 ./ 1024;

data_after_exchanges = ...
    sum(sum(best_policy_segment_allocations)) ...
    .* 10 .* data_rates .* 1024 ./ 8 ./ 1024 ./ 1024;

data_finally_distributed = ...
    number_of_nodes .* sum(sum(initial_segment_allocations)) ...
    .* 10 .* data_rates .* 1024 ./ 8 ./ 1024 ./ 1024;

figure(15);

figure_title = ...
    sprintf('Data category sizes\nunder best-policy');
title(figure_title,'FontSize',20);
hold on;

plot(data_rates, ...
    data_initially_distributed,'--k','LineWidth',3);
hold on
plot(data_rates, ...
    data_after_exchanges,'-+b','LineWidth',3);
hold on
plot(data_rates, ...
    data_finally_distributed,'-xr','LineWidth',3);
hold on

xlim([3 27]);
xlabel('Data rate (Mbps)','FontSize',20);
set(gca,'Xtick',3:3:27);
ylabel('Data size (GB)','FontSize',20);
hold on;
legend({'Initially-distributed','After-exchanges', ...
    'After-final-distributions'}, ...
    'Location','best','FontSize',20);
hold on;
grid on

saveas(figure(15),'data category sizes under best policy.fig');
saveas(figure(15),'data category sizes under best policy.bmp');

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('connectivities');
load('biggest_cluster_data');
load('modified_data');
load('synthetic_lats');
load('synthetic_lons');

%   Extracting next biggest clusters according
%   to the minimum contact durations schedule:

%   Notice that the first biggest cluster is chosen within 
%   the specified period start and end times given the first 
%   minimum discontinuous contact duration as shown at 
%   the beginning of this script (i.e. 20 min). Also notice 
%   that the last minimum discontinuous contact duration of 
%   the schedule should guarantee that all nodes are chosen 
%   eventually to be part of a cluster. Consider that 
%   the schedule given here does not necessarily lead to this 
%   eventual inclusion of all nodes; it is just presented as 
%   a sample for explanation purposes.

period_start_time = 16 * 60;
period_end_time = 18 * 60;
time_delta = 10;
minimum_contact_durations_schedule = [15,15,15] .* 60;
maximum_number_of_hops = 20;

for k = 1:length(minimum_contact_durations_schedule)
    
    %   Removing connectivities of previous biggest clusters:
    
    for t = 1:27 * 60 * 6
        
        connectivities{t}(biggest_cluster_members,:) = zeros;
        connectivities{t}(:,biggest_cluster_members) = zeros;
        
    end
    
    %   Extracting the next biggest cluster:
    
    [biggest_cluster, cluster_members] = ...
        extractCluster(period_start_time, period_end_time, ...
        connectivities, time_delta, ...
        minimum_contact_durations_schedule(k), maximum_number_of_hops);
    
    biggest_cluster_members = ...
        cluster_members{biggest_cluster};
    
    %   Visualizing the biggest cluster at the busiest time:
    
    busiest_time = ...
        (period_start_time + period_end_time) / 2 * 6;
    
    figure(k);
    
    if k == 1
        
        title('Second biggest cluster @ 5:00 PM','FontSize',18);
        hold on
    
    end
    
    if k == 2
        
        title('Third biggest cluster @ 5:00 PM','FontSize',18);
        hold on
        
    end
    
    if k == 3
        
        title('Fourth biggest cluster @ 5:00 PM','FontSize',18);
        hold on
        
    end

    map = imread('map.jpg');
    image('CData',map, ...
        'XData', ...
        [min(modified_data(:,6)) max(modified_data(:,6))], ...
        'YData', ...
        [min(modified_data(:,5)) max(modified_data(:,5))])
    hold on
    
    plot(synthetic_lons(:,busiest_time), ...
        synthetic_lats(:,busiest_time), ...
        'ko','MarkerFaceColor','y');
    hold on
    
    plot(synthetic_lons( ...
        biggest_cluster_members,busiest_time), ...
        synthetic_lats( ...
        biggest_cluster_members,busiest_time), ...
        'ko','MarkerFaceColor','r');
    hold on
    
    axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
        min(modified_data(:,5)) max(modified_data(:,5))]);
    ylabel('Latitude','FontSize',18);
    xlabel('Longitude','FontSize',18);
    hold on
    
    time = ...
        sprintf('%02d:%02d',floor(busiest_time / (60 * 6)), ...
        floor((busiest_time / (60 * 6) - floor(busiest_time / (60 * 6))) * 60));
    text(-80.35,43.575, ...
        time,'Color','black','FontSize',20,'FontWeight','bold')
    hold on
    
    grid on
    
    if k == 1
        
        saveas(figure(k),'second biggest cluster at 5 pm.fig');
        saveas(figure(k),'second biggest cluster at 5 pm.bmp');
        
    end
    
    if k == 2
        
        saveas(figure(k),'third biggest cluster at 5 pm.fig');
        saveas(figure(k),'third biggest cluster at 5 pm.bmp');
        
    end
     
    if k == 3
        
        saveas(figure(k),'fourth biggest cluster at 5 pm.fig');
        saveas(figure(k),'fourth biggest cluster at 5 pm.bmp');
        
    end

end

%%

%   Written by "Kais Suleiman" (ksuleiman.weebly.com)
