# function [numproto new_protoParam] = esSampleProtoParameters(M_tMap, old_protoParam)
# %esSampleProtoParameters - Generates the patch map M(t) parameters $$\theta_p$$
# %
# % Synopsis
# %          [numproto new_protoParam] = esSampleProtoParameters(M_tMap, old_protoParam)
# %
# % Description
# %     In a first step finds the boundaries of the actual patches
# %     In a second step get the N_V best patches ranked through their size and returns the actual
# %     M(t) map
# %
# %
# % Inputs ([]s are optional)
# %   (matrix) M_tMap            the patch map M(t)
# %   (struct) old_protoParam    the patch parameters at time step t-1
# %
# %
# % Outputs ([]s are optional)
# %   (integer) numproto         the actual number of patches
# %   (struct) new_protoParam    the patch parameters at current time step t:
# %                                the proto-objects boundaries: B{p}
# %                                 - new_protoParam.B
# %                                the proto-objects fitting ellipses parameters:
# %                                   a(1)x^2 + a(2)xy + a(3)y^2 + a(4)x + a(5)y + a(6) = 0
# %                                 - new_protoParam.a      conics parameters: a{p}
# %                                the normal form parameters: ((x-cx)/r1)^2 + ((y-cy)/r2)^2 = 1
# %                                 - new_protoParam.r1     normal form parameters
# %                                 - new_protoParam.r2     normal form parameters
# %                                 - new_protoParam.cx     normal form parameters
# %                                 - new_protoParam.cy     normal form parameters
# %                                the rotation parameter
# %                                 - new_protoParam.theta  normal form parameters
# %
# % Requirements
# %   fitellip.m
# %
# % References
# %
# %    G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts,
# %                                     IEEE Trans. SMC-B, to appear
# %
# %    R. Hal?r and J. Flusser, Numerically stable direct least squares fitting of ellipses,
# %                             in Proc. Int. Conf. in Central Europe on Computer Graphics,
# %                             Visualization and Interactive Digital Media,
# %                             vol. 1, 1998, pp. 125?132.
# %
# %
# %
# % Author
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# %
# % Changes
# %   12/12/2012  First Edition
# %

# % Computing patch boundaries
# [B,L] = bwboundaries(M_tMap,'noholes');

# % the actual patch number
# numproto=length(B);

# if numproto~=0
#     bX = cell(1,numproto);
#     bY = cell(1,numproto);
#     a  = cell(1,numproto);
#     for p = 1:numproto
#         boundary = B{p};
#         %plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
#         bX{p}=boundary(:,1);  bY{p}=boundary(:,2);

#         %Halir and Flusser patch parameter ellipse fitting
#         a{p} = fitellip(bX{p},bY{p});

#         % proto-object ellipse parameter in normal form
#         v{p} = solveellipse(a{p});
#         r1(p)=v{p}(1); r2(p)=v{p}(2); cx(p)=v{p}(3); cy(p)=v{p}(4); theta(p)=v{p}(5);

#     end
#     % assign the new parameters
#     new_protoParam.B     = B; % the proto-objects boundaries: B{p}
#     % the proto-objects fitting ellipses parameters:
#     new_protoParam.a     = a; % conics parameters: a{p}
#     new_protoParam.r1    = r1;% normal form parameters
#     new_protoParam.r2    = r2;% normal form parameters
#     new_protoParam.cx    = cx;% normal form parameters
#     new_protoParam.cy    = cy;% normal form parameters
#     new_protoParam.theta = theta;% normal form parameters

# else
#     %use the old ones
#     new_protoParam.B     = old_protoParam.B ;
#     new_protoParam.a     = old_protoParam.a ;
#     new_protoParam.r1    = old_protoParam.r1;
#     new_protoParam.r2    = old_protoParam.r2 ;
#     new_protoParam.cx    = old_protoParam.cx ;
#     new_protoParam.cy    = old_protoParam.cy ;
#     new_protoParam.theta = old_protoParam.theta;

# end
# end
