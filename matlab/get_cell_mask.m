function seg_M = get_cell_mask(img, show_image)            
    %% Convert image to HSV
    hsv = rgb2hsv(img);
    hsv = hsv(:,:,2);
      
    %% Multi Otsu thresholding
    thresh = [0.0510    0.1451    0.2392    0.7647];
    thresh_mask = imquantize(hsv, thresh);
    thresh_mask(thresh_mask == 1) = 0;
    thresh_mask(thresh_mask > 1) = 1;
    thresh_mask = imfill(thresh_mask,'holes');
   
    %% check if it contains object other than the cell_patch inserted
    if sum(thresh_mask(:)) < 900
        seg_M = zeros([size(img,1), size(img,2)]);
    else      
        seg_M = thresh_mask;
    end
    
    if show_image
        figure, 
        subplot(131), imshow(img)
        subplot(132), imshow(thresh_mask, []), impixelinfo
        subplot(133), imshow(seg_M,[]), impixelinfo
        set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    end
end 

