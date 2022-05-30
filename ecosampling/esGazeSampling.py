# function [finalFOA  dir_new   candx candy candidateFOA] = esGazeSampling(gazeSampParam, foaSize, FOA_attractors,nrow,ncol,...
#                                                                         predFOA,dir_old, alpha_stblParam, xCordIP, yCordIP)

# %esGazeSampling - Samples the new gaze position
# %
# %
# % Synopsis
# %          [finalFOA  dir_new, FOA_attractors candidateFOA] = esGazeSampling(gazeSampParam,  foaSize, FOA_attractors,nrow,ncol,...
# %                                                               predFOA,dir_old, alpha_stblParam, xCordIP, yCordIP)
# %
# % Description
# %     Function computing the actual  gaze relocation by using a
# %     Langevin like stochastic differential equation,
# %     whose noise source is sampled from a mixture of $$\alpha$$-stable distributions.
# %     By using NUM_INTERNALSIM generates an equal number of candidate
# %     parameters, that is of candidate new possible shifts which are then
# %     passed to the esLangevinSimSampling() function, which executes the
# %     Langevin step
# %     If a valid new gaze point is returned this is set as the final FOA
# %     Otherwise, a gaussian noise "perturbated" candidate FOA is tried.
# %     The dafault solution, if all goes wrong, is to keep the old FOA
# %
# %
# % Inputs ([]s are optional)
# %   (struct) gazeSampParam              The parameters for gaze sampling
# %   -  NUM_INTERNALSIM                     number of internal candidate new
# %                                          points (the internal simulation)
# %   -  MAX_NUMATTEMPTS                     max num of attempts for getting
# %                                          a neww FOA
# %   - (scalar) z                          The oculomotor action choice.
# %   (scalar) foaSize                    The dimension of the Focus of Attention region
# %   (matrix) FOA_attractors             N_V x 2 matrix representing the
# %                                        FoA attractors
# %   (scalar) nrow                       number of rows of the map
# %   (scalar) ncol                       number of columns of the map
# %   (vector) predFOA                    1 x 2 vector representing the previous FoA coordinates
# %   (scalar) dir_old                    the previous direction of the gaze shift
# %   (struct) alpha_stblParam            The alpha-stable distribution
# %                                           parameters for the K motor
# %                                           actions.
# %   - (vector) alpha                    1 x K vector characteristic exponent
# %   - (vector) beta                     1 x K vector skewness
# %   - (vector) gamma                    1 x K vector scale
# %   - (vector) delta                    1 x K vector location
# %   (vector) xCordIP                    coordinates of IPs
# %   (vector) yCordIP
# %
# %
# % Outputs ([]s are optional)
# %
# %   (vector) finalFOA                    1 x 2 vector representing the new FoA coordinates
# %   (scalar) dir_new                     the new direction of the gaze shift
# %   (vector) candx                       coordinates of candidate FOAs from internal simulation
# %   (vector) candy
# %   (vector) candidateFOA                1 x 2 vector the default candidateFOA new FoA coordinates
# %
# % Example:
# %
# %
# %
# % See also
# %   esLangevinSimSampling
# %
# % Requirements
# %
# %
# % References
# %   [1] G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts
# %       IEEE Trans. Systems Man Cybernetics - Part B (to appear)
# %   [2] G. Boccignone and M. Ferraro, The active sampling of gaze-shifts,
# %       in Image Analysis and Processing ICIAP 2011,
# %       ser. Lecture Notes in Computer Science,
# %       G. Maino and G. Foresti, Eds.	Springer Berlin / Heidelberg, 2011,
# %       vol. 6978, pp. 187?196.
# %
# %
# %
# % Author
# %   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
# %
# % Changes
# %   12/12/2012  First Edition
# %

#     VERBOSE=false;
#     s=foaSize/4;

#     % Calculate    direction toward candidateFOA
#     preferred_dir=[];

#     % set the  center of mass of attractors as a potential candidate FOA
#     candidateFOA = round(sum(FOA_attractors,1)/size(FOA_attractors,1)); %center of mass

#     % Direction Sampling
#     % ...different possibilities:

#     % 0: uniform random sampling
#     dir_new = 2*pi*rand(1,gazeSampParam.NUM_INTERNALSIM);

#     % 1: keep going approximately in the same direction (memory)
#     %dir_new = 2*pi.*rand(1,gazeSampParam.NUM_INTERNALSIM)- pi  + dir_old;

#     % 2: determine a preferred direction toward  a candidate FOA
#     %xx            = candidateFOA(1) - predFOA(1);
#     %yy            = candidateFOA(2)-predFOA(2);
#     %preferred_dir = atan2(yy, xx); %in radians

#     % dir_new= preferred_dir.*rand(1,gazeSampParam.NUM_INTERNALSIM);

#     % 3: determine a preferred direction toward  a candidate FOA with
#     % memory
#     %dir_new = 2*preferred_dir.*rand(1,gazeSampParam.NUM_INTERNALSIM)- preferred_dir + dir_old;

#     % SDE new direction of the gaze shift
#     sdeParam.dir_new = dir_new;

#     % Shaping the potential of Langevin equation
#     %   clearly the sum is over one term if SIMPLE_ATTRACTOR=1
#     dhx = -(predFOA(1)-FOA_attractors(:,1));
#     dHx = sum(dhx,1);
#     dhy = -(predFOA(2)-FOA_attractors(:,2));
#     dHy = sum(dhy,1);

#     %    SDE gradient potential  x coordinate
#     sdeParam.dHx     = dHx;

#     %    SDE gradient potential  y coordinate
#     sdeParam.dHy     = dHy;

#     % setting some sampling parameters befor calling the Langevin SDE
#     T=1;                % maximum time
#     N=30;
#     h=T/N;              % time step
#     %    SDE integration step
#     sdeParam.h       = h;

#     %the oculomotor state
#     z = gazeSampParam.z;

#     %    SDE alpha-stable characteristic exponent parameter
#     sdeParam.alpha   = alpha_stblParam.alpha(z);

#     %    SDE alpha-stable scale parameter
#     sdeParam.gamma   = alpha_stblParam.gamma(z);

#     accept=0;
#     count_attempts=0;
#     % Cycling until a sampled FOA is  accepted: thus making sure we have
#     % one
#     while(~accept && (count_attempts < gazeSampParam.MAX_NUMATTEMPTS) )
#         % Generate randomly a jump length r,
#         % Uses STABRND.M
#         % Stable Random Number Generator (McCulloch 12/18/96)
#         %
#         %   xi = stabrnd(alpha, beta, c, delta, m, n);
#         %
#         % Returns m x n matrix of iid stable random numbers with
#         %   characteristic exponent alpha in [.1,2], skewness parameter
#         %   beta in [-1,1], scale c > 0, and location parameter delta.
#         %
#         % Based on the method of J.M. Chambers, C.L. Mallows and B.W.
#         %   Stuck, "A Method for Simulating Stable Random Variables,"
#         %   JASA 71 (1976): 340-4.

#         % setting alpha-stable distribution parameters according to the
#         % regime specified by rv z
#         xi = stabrnd(alpha_stblParam.alpha(z), alpha_stblParam.beta(z),...
#                      alpha_stblParam.gamma(z), alpha_stblParam.delta(z),...
#                      1, gazeSampParam.NUM_INTERNALSIM);

#         % Setting Langevin SDE parameters
#         %    SDE alpha-stable component
#         sdeParam.xi      = xi;

#         % Langevin gaze shift sampling
#         %   finalFOA is the sampled new Gaze position
#         [finalFOA dir_new accept candx candy] = esLangevinSimSampling(sdeParam, predFOA,  nrow,ncol,...
#                                                                       xCordIP, yCordIP);
#         count_attempts = count_attempts+1;
#         if VERBOSE
#                 fprintf('\n Trying...count_attempts = %d', count_attempts);
#         end
#     end %while

#     % if something didn't work for some reason...
#     %   using a perturbed argmax solution
#     if(~accept)
#         %for normal regime simple choice on most salient point
#         new= randnorm(1, candidateFOA', [], eye(2)*s);
#         new=new';
#         validcord = find((new(1) >= 1) & (new(1) < nrow) & (new(2) >= 1) & (new(2) < ncol));
#         if VERBOSE
#              fprintf('\n Sampled %d valid candidate FOAs from normal', numel(validcord));
#         end
#         if numel(validcord) > 0
#         finalFOA=new;
#         accept=1; %surely accepted
#             if VERBOSE
#                 disp('DEFAULT SOLUTION: Perturbed ArgMax solution!!')
#             end
#         end
#     end

#     % the last chance: use the previous FOA
#     if(~accept)
#         %keep on previous one
#         finalFOA = predFOA;
#         accept=1; %surely accepted
#         if VERBOSE
#             disp('BACKUP SOLUTION!!!!!!Keeping OLD FOA')
#         end
#     end

#     if VERBOSE
#         fprintf('Current FOA %g, %g \n', predFOA(1), predFOA(2));
#         fprintf('Candidate Max FOA %g, %g \n', candidateFOA(1), candidateFOA(2));
#         fprintf('Old dir %g \n', rad2deg(dir_old));
#         fprintf('Preferred  dir %g \n', rad2deg(preferred_dir));

#         fprintf('New dir %g \n', rad2deg(dir_new));
#         %fprintf('Jump lenght %g \n', abs(xi));
#         fprintf('Final FOA %g, %g \n', finalFOA(1), finalFOA(2));
#         dis = sqrt( (finalFOA(1)- candidateFOA(1))^2 + (finalFOA(2)- candidateFOA(2))^2 );
#         fprintf('Distance from candidate FOA %g \n', dis);

#         fprintf('-----FOA SAMPLING TERMINATED \n');
#         pause(0.1)
#     end



