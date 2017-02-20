
%
%
% Cropping 29x29 patches
%
% FRANCISCO CARRILLO PÉREZ
% IMAGE ANALYSIS AND COMPUTER VISION
% POLITECNICO DI MILANO (2017)
%
%

%% CROP
clear
I = imread('/media/datos/Dropbox/4ºaño/Image Analysis and Computer Vision/NanoFibers/Anomalous/ITIA1115.tif');
M = imread('/media/datos/Dropbox/4ºaño/Image Analysis and Computer Vision/NanoFibers/Anomalous/ITIA1115_gt.png');
vec = -14:14;
counter_of_normal_patches = 1;
counter_of_anomalous_patches = 1;
counter_of_patches = 17144;
for x=1:1024
    for y=1:696
       %This is the pixel were the patch is going to be crop around
       pixel=I(x,y);
       pixel_gt = M(x,y);
       if(x < 29 || y < 29)
       
       else
           patchI = I(x+vec,y+vec);
           patch_gt = M(x+vec,y+vec);
           if(all(patch_gt))
               
               if(counter_of_anomalous_patches < 200)
                path = strcat('/media/datos/Dropbox/4ºaño/Image Analysis and Computer Vision/NanoFibers/DataSet/Anomalous/New/patch',int2str(counter_of_patches));
                imwrite(patchI,strcat(path,'.png'));
                counter_of_anomalous_patches = counter_of_anomalous_patches + 1;
                counter_of_patches = counter_of_patches + 1;
               end
                
           else
               
               if(not(all(patch_gt)))
                   
                   if(counter_of_normal_patches < 500)
                       path = strcat('/media/datos/Dropbox/4ºaño/Image Analysis and Computer Vision/NanoFibers/DataSet/Normal/New/patch',int2str(counter_of_patches));
                       imwrite(patchI,strcat(path,'.png'));
                       counter_of_normal_patches = counter_of_normal_patches + 1;
                       counter_of_patches = counter_of_patches + 1;
                   end
                
               end
                
           end
           
           
       end
    end

end
