function index1= DBSCAN2(data,eps,minPts)
        %%
        core_object = [];
        count = 1;
        k = 1; % label index
        vmask = ones(size(data, 1), 1); % variable used to record whether one observation has been visited (1: not visited, 0: visited)
        label = zeros(size(data, 1), 1); % pre-allocate result
        
        % Find core objects
        for i = 1:size(data, 1)
            x = data(i, :); % current observation of interests
            [~, neighbors] = Find_Neighbor(data, x, eps);
            
            % Check if the number of observations in neighbors of x is larger than minPts 
            if((size(neighbors,1)-1) >= minPts)
                
                % Add the index of x to the set of core object
                core_object(count, 1) = i;
                count = count + 1;
            end
        end
        
        % fprintf('---- DBSCAN finds a total number of %i core objects ----\n', size(core_object, 1));
        
        while(~isempty(core_object))
            
            % Construct one queue for further use 
            queue = [];
            qcount = 1;
            
            % Update core_object, observations have been visited should be deleted
            tcount = 1;
            list = [];
            for i = 1:size(core_object, 1)
                if(vmask(core_object(i, 1),1) == 0)
                    list(tcount ,1) = i;
                    tcount = tcount + 1;
                end
            end
            core_object(list,:) = [];
            
            if(isempty(core_object))
                break;
            end
            
            % Randomly choose one core object
            index = core_object(randi(size(core_object, 1)), 1);
            queue(qcount, 1) = index; % add the selected core object to the queue
            qcount = qcount + 1;
            vmask(index, 1) = 0; % update vmask
            label(index, 1) = k; % assign result 
            core_object(core_object == index, :) = []; % remove the selected core object from the set
            
            while(~isempty(queue))
                pivot = queue(1, 1); 
                queue(1, :) = []; % remove the first element in queue
                qcount = qcount - 1;
                
                % Find neighbors for current observation of interests 
                [tindex, tneighbors] = Find_Neighbor(data, data(pivot, :), eps);
                if(size(tneighbors, 1) >= minPts)
                    for i = 1 : size(tindex, 1)
                        % If one neighbor has not been visited, add it to the queue
                        if(vmask(tindex(i, 1),1) == 1)
                            queue(qcount, 1) = tindex(i, 1);
                            qcount = qcount + 1;
                            vmask(tindex(i, 1),1) = 0; % update vmask
                            label(tindex(i, 1),1) = k; % assign result
                        end
                    end
                end   
            end
          
        %     fprintf('---- Finish finding observations for class %i ----\n', k);
            k = k + 1; % next class
        end
              a=unique(label); %找出分类出的个数
          b=length(a);
           C=cell(1,length(a));
           for i=1:length(a)
               C(1,i)={find(label==a(i,1))};
           end
           for j=1:length(C)
           n1=length(C{1,j});
           index1(C{1,j})=j;
           end
    end