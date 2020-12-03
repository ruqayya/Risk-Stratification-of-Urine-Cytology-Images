function [ blur_map ] = get_blur_map( I, block_size )
    if size(I, 3) == 3
       I = rgb2gray(I);
    end

    blur_map = blockproc(I, [block_size block_size],@processBlock);
end

function block = processBlock(block_struct)
    block = block_struct.data;
    blur = blurMetric(block, 0);
    if blur < 0.18
        block(:,:,:) = 0;
    else 
        block(:,:,:) = 255;
    end
end

