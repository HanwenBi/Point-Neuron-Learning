function [PnCoord, PnWeight,PnWscale, Loss] = PointNeuron(k, MicField, ...
    MicCoord, InCoord, InWeight,StepW,StepC,Lambda,IterN)
% Creation   : 15-09-24
% Author     : Hanwen Bi (hanwen.bi@anu.edu.au)
% Version    : 21-09-24
% Descirption:
%       The function iteratively updates the locations
%       and weights of point neurons by gradient descent.
% Inputs:
%       1) k        : The wavenumber.
%       2) MicField : The microphone measured sound field in the frequency 
%                     domain (Q-by-1).
%       3) MicCoord : the Cartesian coordinates of the microphone points 
%                     (Q-by-3).
%       4) InCoord  : the Cartesian coordinates of the initial neurons 
%                     (P-by-3).
%       5) InWeight : the initial weight of point neurons (P-by-1).
%       6) StepC    : Step size for updating point neuron location 
%                     in xyz (3-by-1).
%       7) StepW    : Step size for updating point neuron weights.
%       8) Lambda   : Model complexity penalty (L1 norm).
%       9) IterN    : No of iterations.
%
% 
% Outputs:
%       1) PnCoord  : The optimal Cartesian location of point neurons 
%                     (P-by-3).
%       2) PnWeight : The optimal weights of point neurons (P-by-3).
%       3) PnWscale : Scale parts of the mix-wave function (P-by-1).
%       4) Loss     : Loss of the network.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%    Net Settings 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MicN        = length(MicField);
GradB       = 0.3;
%%%% Number of point neuron 
PnN         = length(InWeight);
%%%% Point neuron coordinate
PnX         = zeros(PnN,IterN+1); 
PnY         = zeros(PnN,IterN+1);
PnZ         = zeros(PnN,IterN+1);
%%%% Point neuron weight
PnW         = zeros(PnN,IterN+1); 

%%%% Gradient of point neuron coordinate
GradientX   = zeros(PnN,IterN);
GradientY   = zeros(PnN,IterN);
GradientZ   = zeros(PnN,IterN);
%%%% Gradient of point neuron weight
GradientW   = zeros(PnN,IterN);
%%%% Loss
Loss        = []; 
%%%%%% Initialization 
PnX(:,1)    = InCoord(:,1);
PnY(:,1)    = InCoord(:,2);
PnZ(:,1)    = InCoord(:,3);
PnW(:,1)    = InWeight;
PnCoord     = zeros(PnN,3);


%%%%% Optimal threshold
OptimalL    = 2e7;

%%%%% Threshold for loss converge

LossThres   = 5e-7;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%    Net Processing 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

for i=1:IterN
    %%%%% Forward Propagation %%%%%%%%%%%%
    %%% X-coordiante difference between mic and point neuron
    DiffX       = [];
    %%% Y-coordiante difference between mic and point neuron
    DiffY       = [];
    %%% Z-coordiante difference between mic and point neuron
    DiffZ       = [];
    %%% Distance^2 between mic and point neuron
    DistanceS   = [];
    
    DistanceE   = [];
    %%% Green function
    Hn          = [];
    
    Pnscale     = [];
    
    %%%% Sound field of point neuron over mics
    MicP        = zeros(MicN,1);
   
    for m = 1:MicN
        DiffX(:,m)      = PnX(:,i)-MicCoord(m,1);
        DiffY(:,m)      = PnY(:,i)-MicCoord(m,2);
        DiffZ(:,m)      = PnZ(:,i)-MicCoord(m,3);
        DistanceS(:,m)  = DiffX(:,m).^2+DiffY(:,m).^2+DiffZ(:,m).^2;
        DistanceE(:,m)  = PnX(:,i).^2+PnY(:,i).^2+PnZ(:,i).^2;
        Hn(:,m)         = exp(1i*k*sqrt(DistanceS(:,m)))./ ...
                          sqrt(DistanceS(:,m))/(4*pi);
        Pnscale(:,m)    = sqrt(DistanceE(:,m)).*exp(-1i*k* ...
                          sqrt(DistanceE(:,m)));
        MicP(m,1)       = (PnW(:,i).')*(Hn(:,m).*Pnscale(:,m));
    end
    
    Loss(i)     = norm(MicP-MicField)+Lambda*norm(PnW(:,i),1);
    
    %%%% Print Loss
    if rem(i,100)==0
        fprintf('%10s %d\n','Loss');
        disp(Loss(i))
    end
    
    if Loss(i)<=OptimalL
        PnCoord(:,1)  = PnX(:,i);
        PnCoord(:,2)  = PnY(:,i);
        PnCoord(:,3)  = PnZ(:,i);
        PnWeight      = PnW(:,i);
        PnWscale      = Pnscale(:,1);
        OptimalL      = Loss(i);       
    end
    
    %%%%% Test loss converge
    if i > 1
        if abs(Loss(i-1) - Loss(i)) < LossThres
            break
        end
    end
    %%%%%%%%%% Back Propagation %%%%%%%%%%%%%

    
    for s=1:PnN
        %%%%%%% Update point neuron weight
        GradientW(s,i) = conj((2*Hn(s,:).*Pnscale(s,:))* ...
                         conj((MicP-MicField)));
        SparGW         = (real(PnW(s,i))+1i*imag(PnW(s,i)))/ ...
                          abs(PnW(s,i));
        PnW(s,i+1)     = PnW(s,i)-StepW*GradientW(s,i)-StepW*Lambda*SparGW;
        
        %%%%%%% Update point neuron coordinate 
        %%% Derivative of the Green function with respect to distance
        DHn            = PnW(s,i)*(1i*k*sqrt(DistanceS(s,:))-1).* ...
                         exp(1i*k*sqrt(DistanceS(s,:)))./(4*pi* ...
                         DistanceS(s,:)).*Pnscale(s,:);
                          
        DPn            = PnW(s,i)*(-(1i*k*sqrt(DistanceE(s,:))-1)).* ...
                              exp(-1i*k*sqrt(DistanceE(s,:))).*Hn(s,:);
                          

        GradientX(s,i) = 2*real((DiffX(s,:).*(1./sqrt(DistanceS(s,:))) ...
                              .*DHn+PnX(s,i)*(1./sqrt(DistanceE(s,:))) ...
                              .*DPn)*conj(MicP-MicField));
                          
        GradientY(s,i) = 2*real((DiffY(s,:).*(1./sqrt(DistanceS(s,:))) ...
                              .*DHn+PnY(s,i)*(1./sqrt(DistanceE(s,:))) ...
                              .*DPn)*conj(MicP-MicField));
        GradientZ(s,i) = 2*real((DiffZ(s,:).*(1./sqrt(DistanceS(s,:))) ...
                              .*DHn+PnZ(s,i)*(1./sqrt(DistanceE(s,:))) ...
                              .*DPn)*conj(MicP-MicField));
                          
        %%%% Avoiding point neurons too close to mics 
        if abs(GradientX(s,i)*StepC(1))>0.2*GradB
            GradientX(s,i)  = 0.2*GradB*GradientX(s,i)/ ...
                              abs(GradientX(s,i)*StepC(1));
        end
        
        if abs(GradientY(s,i)*StepC(2))>0.2*GradB
            GradientY(s,i)  = 0.2*GradB*GradientY(s,i)/ ...
                              abs(GradientY(s,i)*StepC(2));
        end
        if abs(GradientZ(s,i)*StepC(3))>0.2*GradB
            GradientZ(s,i)  = 0.2*GradB*GradientZ(s,i)/ ...
                              abs(GradientZ(s,i)*StepC(3));
        end

        %%%%% Update point neuron coordinates
        PnX(s,i+1)     = PnX(s,i)-StepC(1)*GradientX(s,i);
        PnY(s,i+1)     = PnY(s,i)-StepC(2)*GradientY(s,i);
        PnZ(s,i+1)     = PnZ(s,i)-StepC(3)*GradientZ(s,i);

        
    end
end


end
