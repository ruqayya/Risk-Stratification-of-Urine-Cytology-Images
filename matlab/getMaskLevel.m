function BottomResolution = getMaskLevel( wsi_path,  max_pixels)
    %% identify the level number corresponding to 1.25x (which is level 5 if the highest resolution is 40x and 6 when it is 60x)
    ImageInfo = imfinfo(wsi_path, 'JP2');
    numberOfLevels = ImageInfo.WaveletDecompositionLevels+1;
    
    ResolutionFactors = 2.^(0:(numberOfLevels-1));   
    MaxResolution = sqrt((ImageInfo.Width*ImageInfo.Height)/double(max_pixels));
    Resolution = find(MaxResolution <= ResolutionFactors, 1, 'first');
    Resolution = Resolution-1;                      
    
    MinResolution = numberOfLevels-1;
    BottomResolution = min([MinResolution Resolution]);
end

