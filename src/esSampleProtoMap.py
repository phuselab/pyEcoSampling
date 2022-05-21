# function [M_tMap, protoMap protoMap_raw normSal] = esSampleProtoMap(sMap,...
#                                                           currFrame,...
#                                                           nBestProto)
# %esSampleProtoMap - Generates the patch map M(t)
# %
# % Synopsis
# %          [protoMap protoMap_raw normSal] = esSampleProtoMap(sMap,currFrame, nBestProto)
# %
# % Description
# %     In a first step generates the raw patch map by thresholding  the normalized salience map
# %     so as to achieve 95% significance level for deciding whether the given saliency values are
# %     in the extreme tails
# %     In a second step get the N_V best patches ranked through their size and returns the actual
# %     M(t) map
# %
# %
# % Inputs ([]s are optional)
# %   (matrix) sMap         the salience map, 0/1 overlay representation
# %   (matrix) currFrame    the current frame
# %   (integer) nBestProto  the N_V most valuable patches
# %
# %
# % Outputs ([]s are optional)
# %   (matrix) M_tMap       the patch map M(t)
# %   (matrix) protoMap     the object layer representation of patch map M(t)
# %   (matrix) protoMap_raw   the raw patch map
# %   (matrix) normSal      the normalized salience map
# %
# %
# %
# % Requirements
# %   Image Processing toolbox
# %
# % References
# %   [1] G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts
# %       IEEE Trans. Systems Man Cybernetics - Part B (to appear)
# %
# %
# % Author
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# %
# % Changes
# %   12/12/2012  First Edition
# %

# protoMap_raw= false(size(sMap)); % allocating space for the raw map

# % Normalizing salience
# normSal=sMap;
# maxsal=max(max(normSal));
# minsal=min(min(normSal));
# normSal= (normSal-minsal)./(maxsal-minsal);
# normSal=normSal*100;

# % Salience thresholding
# % Method 1: adaptive on mu
# %mu=mean2(sal);
# %stdv=std2(sal);
# %ind=find(sal>=4*mu); %standard measure 3*mu
# % Method 2: percentile based
# ind=find(normSal >= prctile(normSal(:),90));
# protoMap_raw(ind)=1;
# protoMap_raw(ind)= currFrame(ind);


# % Samples the N_V best patches
# openingwindow=7;
# M_tMap = esSampleNbestProto(protoMap_raw,nBestProto,openingwindow);


# protoMap= M_tMap;
# fbw = M_tMap==0;
# protoMap(fbw) = 1;
# fbw = M_tMap==1;
# protoMap(fbw) = 0;
# end


# % -------------------------------------------------
# function M_tMap = esSampleNbestProto(protoMap_raw, nBest, win)
# %esSampleNbestProto - Samples the N_V best patches ranked through their size and returns the actual
# %                     M(t) map

# % Use morphological operations
# se = strel('disk',win);
# protoMap_raw = imopen(protoMap_raw,se);

# % Calculating 8 connected components via Rosenfeld and Pfaltz
# [L,obj] = bwlabel(protoMap_raw,8);

# % Returns the foreground connected component in the binary image
# % supplied that have the specified ranked size(s).
# maxL = max(L(:));					      % max number of connected components
# h = hist(L(find(protoMap_raw)),[1:maxL]); % find indexes of labeled elements, by column scanning,
#                                           %    and compute occurence of labels from 1 to maxL
# [sh,sr] = sort(-h);                       % sorting with respect to the number of occurrence
#                                           %    sh is the num occurrences (negative: -8   -6  -4  -2),
#                                           %    sr = corresponding labels
# M_tMap = protoMap_raw & 0;
# if nBest > maxL
#      nBest = maxL;
# end
# for i=1:nBest
#     M_tMap = M_tMap | (L==sr(i)) ;      % returns the nBest by  dimension
# end
# end
