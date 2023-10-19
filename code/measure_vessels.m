%measure volume, surface area, and so on.
clc
clear
imgPathAll = dir('./data_feat_new/miss1/*.nii.gz'); %2023.9.10
for n=1:length(imgPathAll)
    imgPath = ['./data_feat_new/miss1/' imgPathAll(n).name]
    I = load_nii(imgPath); 
    CC = bwconncomp(I.img);
    clear I
    stats = regionprops3(CC,'SurfaceArea','Volume');
    num(n)=CC.NumObjects
    clear CC
    v(n)=sum(stats.Volume)
    s(n)=sum(stats.SurfaceArea)
    clear stats
end
    
