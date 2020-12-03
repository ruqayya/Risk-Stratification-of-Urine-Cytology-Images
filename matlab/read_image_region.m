function [patchI, cellMask] = read_image_region(wsi_path, image_start_row, image_end_row, image_start_col, image_end_col,  image_reduction_level, block_size)  
    patchI = imread(wsi_path, 'ReductionLevel', image_reduction_level,...
                      'PixelRegion',{[image_start_row+1, image_end_row], [image_start_col+1, image_end_col]});
    blur_map = get_blur_map( patchI, block_size )/255;
    seg_map = get_cell_mask(patchI, 0);  
    cellMask = blur_map.*uint8(seg_map);
end

