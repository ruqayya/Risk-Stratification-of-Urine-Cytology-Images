function ROI = detect_cellular_region( wsi_path,  BottomResolution, isShow)
    % this code will generate a mask for the cellular region at 0.3125x
    % resolution (equivalent to level 7 if the highest resolution is 40x)
    
    num_thresholds = 3;
    se = strel('line', 12, 90);     % 50 and 12 for the level 5 and 7 respectively
       
    %% read the image and apply thresholding to get a map for the marker marks and curvy lines
    rgb = imread(wsi_path, 'ReductionLevel', BottomResolution);
    gray = rgb2gray(rgb);  
    level = multithresh(gray, num_thresholds);
    mask = imquantize(gray, level);        
    mask(mask == 1) = 0;          
    mask(mask > 1) = 1;        
    mask = imcomplement(mask);

    if sum(mask(:)) < 12500             % 50000 and 12500 for level 5 and 7 respectively
        mask = imquantize(gray, level);        
        mask(mask == 1 | mask == 2) = 0;          
        mask(mask > 2) = 1;        
        mask = imcomplement(mask);
    end   

    %% Remove the marker marks to get a mask for two curvy lines only
    curvy_mask = zeros(size(mask));
    CC = bwconncomp(mask);
    stats = regionprops(CC, 'Area');
    area = [stats.Area];  
    sorted_area = sort(area, 'descend');
    indx = find(area == sorted_area(1) | area == sorted_area(2));
    curvy_mask(cat(1,CC.PixelIdxList{indx})) = 1; 
    curvy_mask = imfill(curvy_mask, 'holes');
    curvy_mask  = imerode(curvy_mask, se);
    curvy_mask = double(bwareaopen(curvy_mask, 2500));         % 10000 and 2500 for level 5 and 7 respectively
    
    %% get the inside region of the curvy lines
    curvy_mask(:,1:20) = 0;
    curvy_mask(:,end-20:end) = 0;
    L = bwlabel(curvy_mask);
    bar_col = []; 
    bar_row = [];

    temp = curvy_mask;
    for iBar = 1:2
        bar = (L==iBar);
        [rows, cols] = find(bar);
        bar_col(iBar, :) = [min(cols), max(cols)]; 
        bar_row(iBar, :) = [min(rows), max(rows)];
    end 
    if sum(bar_row(1,:) < bar_row(2,:)) == 2
        temp(bar_row(1,1):bar_row(2,2), max(bar_col(:,1)):max(bar_col(:,1))+1) = 1;
        temp(bar_row(1,1):bar_row(2,2), min(bar_col(:,2)):min(bar_col(:,2))+1) = 1;
    else
        temp(bar_row(2,1):bar_row(1,2), max(bar_col(:,1)):max(bar_col(:,1))+1) = 1;
        temp(bar_row(2,1):bar_row(1,2), min(bar_col(:,2)):min(bar_col(:,2))+1) = 1;
    end

    temp = imfill(temp, 'holes');
    ROI = imdilate(curvy_mask, strel('disk', 12));      % 50 and 12 for the level 5 and 7 respectively
    if sum(size(ROI) == size(temp)) ~=2
        ROI = imresize(ROI, size(temp));
    end
    ROI = temp.*imcomplement(ROI);
    ROI  = imerode(ROI, strel('line',12,180));          % 50 and 12 for the level 5 and 7 respectively
    
    if isShow
        figure('units','normalized','outerposition',[0 0 1 1]),
        subplot(141), imshow(gray);  
        subplot(142), imshow(mask, []);  
        subplot(143), imshow(curvy_mask, []);
        
        % visualize the overlay image
        gray_overlay = gray;
        contour = bwperim(ROI);
        contour = imdilate(contour, strel('disk', 3));
        temp = gray; temp(contour) = 255;
        gray_overlay(:,:,1) = temp;
        temp = gray; temp(contour) = 0;
        gray_overlay(:,:,2) = temp;
        gray_overlay(:,:,3) = temp;

        subplot(144), imshow(gray_overlay);
    end
end

