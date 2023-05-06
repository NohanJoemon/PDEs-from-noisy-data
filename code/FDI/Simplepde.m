Qval = zeros(41,11) ; 
termsfull = strings(41,11) ; 
valuesfull = zeros(41,11) ; 
for ni=0:1:40
    % ================ Importing the dataset =======================
    fname = '..\..\Datasets\4_'+string(ni)+'.pkl';
    fid=py.open(fname,'rb');
    data=py.pickle.load(fid);
    d=struct(data);
    u = double(d.u);
    x = double(d.x);
    t = double(d.t);
    dx = x(2)-x(1);
    dt = t(2)-t(1);
    
    % ================= Building the library =======================
    U_dxn=differfitx(u,[-15:1:15],dx,10,4);
    U_dtn=differfitx(u.',[-15:1:15],dt,7,2);
    U=U_dxn(:,:,1);
    U2=U.^2;
    Ut1=U_dtn(:,:,2).';
    Ux1=U_dxn(:,:,2);
    Ux2=U_dxn(:,:,3);
    Ux3=U_dxn(:,:,4);
    UUx1=U.*Ux1;
    U2Ux1=U.*U.*Ux1;
    UUx2=U.*Ux2;
    U2Ux2=U.*U.*Ux2;
    UUx3=U.*Ux3;
    U2Ux3=U.*U.*Ux3;
    A=[f(Ux1),f(Ux2),f(Ux3),f(U),f(U2),f(UUx1),f(U2Ux1),f(UUx2),f(U2Ux2),f(UUx3),f(U2Ux3)];
    desc_A=["u_{x}","u_{xx}","u_{xxx}","u","u^2","uu_{x}","u^2u_{x}","uu_{xx}","u^2u_{xx}", "uu_{xxx}","u^2u_{xxx}"];
    b=f(Ut1);
    A=[real(A)];b=[real(b)];
    
    
    
   % ================= Finding Q ===============================
    x0=A\b;
    [MA,NA]=size(A);
    Q=zeros(1,NA);
    for i=1:NA
        AA=A;
        xx0=x0;
        AA(:,i)=[];
        xx0(i)=[];
        x=AA\b;
        xe=x-xx0;
        Q(i)=sum(abs(xe));
    end
    Q=Q/sum(Q);
    Qval(ni+1,:)=Q;
    [Q_sorted,Q_sorted_idx] = sort(Q,'descend');
    kpt = knee_pt(Q_sorted);
    terms = desc_A(Q_sorted_idx);
    solution=A(:,Q_sorted_idx(1:kpt))\b;
    solution = solution';
    values = solution;

    for ijx=1:kpt
        if abs(solution(ijx))>0.001
            values(ijx) = solution(ijx);
        else
            values(ijx) = 0.000;
        end
    end
    for ijx=kpt+1:length(terms)
        values(ijx)=0.000;
    end
    [terms_sorted,terms_sorted_idx] = sort(terms,'descend');
    termsfull(ni+1,:)=terms_sorted;
    valuesfull(ni+1,:)=values(terms_sorted_idx);
    
end

% ========= FREQ DOMAIN TRANSFORM FUNCTION ======================
function r=f(u)
    u_tem=u(20:end-20,20:end-20);
    u_f=fft2(u_tem);
    indy=1:10;indx=1:10;NA=length(indx)*length(indy);
    r=reshape(u_f(indx,indy),NA,1);
end



% ========= KNEE-POINT FUNCTION ======================================
function [res_x, idx_of_result] = knee_pt(y,x,just_return)
%function [res_x, idx_of_result] = knee_pt(y,x,just_return)
%Returns the x-location of a (single) knee of curve y=f(x)
%  (this is useful for e.g. figuring out where the eigenvalues peter out)
%
%Also returns the index of the x-coordinate at the knee
%
%Parameters:
% y (required) vector (>=3 elements)
% x (optional) vector of the same size as y
% just_return (optional) boolean
%
%If just_return is True, the function will not error out and simply return a Nan on
%detected error conditions
%
%Important:  The x and y  don't need to be sorted, they just have to
%correspond: knee_pt([1,2,3],[3,4,5]) = knee_pt([3,1,2],[5,3,4])
%
%Important: Because of the way the function operates y must be at least 3
%elements long and the function will never return either the first or the
%last point as the answer.
%
%Defaults:
%If x is not specified or is empty, it's assumed to be 1:length(y) -- in
%this case both returned values are the same.
%If just_return is not specified or is empty, it's assumed to be false (ie the
%function will error out)
%
%
%The function operates by walking along the curve one bisection point at a time and
%fitting two lines, one to all the points to left of the bisection point and one
%to all the points to the right of of the bisection point.
%The knee is judged to be at a bisection point which minimizes the
%sum of errors for the two fits.
%
%the errors being used are sum(abs(del_y)) or RMS depending on the
%(very obvious) internal switch.  Experiment with it if the point returned
%is not to your liking -- it gets pretty subjective...
%
%
%Example: drawing the curve for the submission
% x=.1:.1:3; y = exp(-x)./sqrt(x); [i,ix]=knee_pt(y,x); 
% figure;plot(x,y);
% rectangle('curvature',[1,1],'position',[x(ix)-.1,y(ix)-.1,.2,.2])
% axis('square');
%
%Food for thought: In the best of possible worlds, per-point errors should
%be corrected with the confidence interval (i.e. a best-line fit to 2
%points has a zero per-point fit error which is kind-a wrong).
%Practially, I found that it doesn't make much difference.
% 
%dk /2012
%{
% test vectors:
[i,ix]=knee_pt([30:-3:12,10:-2:0])  %should be 7 and 7
knee_pt([30:-3:12,10:-2:0]')  %should be 7
knee_pt(rand(3,3))  %should error out
knee_pt(rand(3,3),[],false)  %should error out
knee_pt(rand(3,3),[],true)  %should return Nan
knee_pt([30:-3:12,10:-2:0],[1:13])  %should be 7
knee_pt([30:-3:12,10:-2:0],[1:13]*20)  %should be 140
knee_pt([30:-3:12,10:-2:0]+rand(1,13)/10,[1:13]*20)  %should be 140
knee_pt([30:-3:12,10:-2:0]+rand(1,13)/10,[1:13]*20+rand(1,13)) %should be close to 140
x = 0:.01:pi/2; y = sin(x); [i,ix]=knee_pt(y,x)  %should be around .9 andaround 90
[~,reorder]=sort(rand(size(x)));xr = x(reorder); yr=y(reorder);[i,ix]=knee_pt(yr,xr)  %i should be the same as above and xr(ix) should be .91
knee_pt([10:-1:1])  %degenerate condition -- returns location of the first "knee" error minimum: 2
%}
%set internal operation flags
use_absolute_dev_p = true;  %ow quadratic
%deal with issuing or not not issuing errors
issue_errors_p = true;
if (nargin > 2 && ~isempty(just_return) && just_return)
    issue_errors_p = false;
end
%default answers
res_x = nan;
idx_of_result = nan;
%check...
if (isempty(y))
    if (issue_errors_p)
        error('knee_pt: y can not be an empty vector');
    end
    return;
end
%another check
if (sum(size(y)==1)~=1)
    if (issue_errors_p)
        error('knee_pt: y must be a vector');
    end
    
    return;
end
%make a vector
y = y(:);
%make or read x
if (nargin < 2 || isempty(x))
    x = (1:length(y))';
else
    x = x(:);
end
%more checking
if (ndims(x)~= ndims(y) || ~all(size(x) == size(y)))
    if (issue_errors_p)
        error('knee_pt: y and x must have the same dimensions');
    end
    
    return;
end
%and more checking
if (length(y) < 3)
    if (issue_errors_p)
        error('knee_pt: y must be at least 3 elements long');
    end
    return;
end
%make sure the x and y are sorted in increasing X-order
if (nargin > 1 && any(diff(x)<0))
    [~,idx]=sort(x);
    y = y(idx);
    x = x(idx);
else
    idx = 1:length(x);
end
%the code below "unwraps" the repeated regress(y,x) calls.  It's
%significantly faster than the former for longer y's
%
%figure out the m and b (in the y=mx+b sense) for the "left-of-knee"
sigma_xy = cumsum(x.*y);
sigma_x  = cumsum(x);
sigma_y  = cumsum(y);
sigma_xx = cumsum(x.*x);
n        = (1:length(y))';
det = n.*sigma_xx-sigma_x.*sigma_x;
mfwd = (n.*sigma_xy-sigma_x.*sigma_y)./det;
bfwd = -(sigma_x.*sigma_xy-sigma_xx.*sigma_y) ./det;
%figure out the m and b (in the y=mx+b sense) for the "right-of-knee"
sigma_xy = cumsum(x(end:-1:1).*y(end:-1:1));
sigma_x  = cumsum(x(end:-1:1));
sigma_y  = cumsum(y(end:-1:1));
sigma_xx = cumsum(x(end:-1:1).*x(end:-1:1));
n        = (1:length(y))';
det = n.*sigma_xx-sigma_x.*sigma_x;
mbck = flipud((n.*sigma_xy-sigma_x.*sigma_y)./det);
bbck = flipud(-(sigma_x.*sigma_xy-sigma_xx.*sigma_y) ./det);
%figure out the sum of per-point errors for left- and right- of-knee fits
error_curve = nan(size(y));
for breakpt = 2:length(y-1)
    delsfwd = (mfwd(breakpt).*x(1:breakpt)+bfwd(breakpt))-y(1:breakpt);
    delsbck = (mbck(breakpt).*x(breakpt:end)+bbck(breakpt))-y(breakpt:end);
    %disp([sum(abs(delsfwd))/length(delsfwd), sum(abs(delsbck))/length(delsbck)])
    if (use_absolute_dev_p)
        % error_curve(breakpt) = sum(abs(delsfwd))/sqrt(length(delsfwd)) + sum(abs(delsbck))/sqrt(length(delsbck));
        error_curve(breakpt) = sum(abs(delsfwd))+ sum(abs(delsbck));
    else
        error_curve(breakpt) = sqrt(sum(delsfwd.*delsfwd)) + sqrt(sum(delsbck.*delsbck));
    end
end
%find location of the min of the error curve
[~,loc] = min(error_curve);
res_x = x(loc);
idx_of_result = idx(loc);
end
