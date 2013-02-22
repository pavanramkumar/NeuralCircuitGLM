function compute_train_and_test_R2(Xfc, Yfc, TrainSELECT_LO, TestSELECT_LO, TrainSELECT_HI, TestSELECT_HI, BLO, BHI, L, n, cv)
global R2r R2t

KIN = [3*L+1:size(Xfc,2)];
SELF = [3*(n-1)+1:3*n];
KINSELF = [3*(n-1)+1:3*n, 3*L+1:size(Xfc,2)];
SELFCROSS = 1:3*L;
ALL = 1:length(BLO)-1;

% Kinematics only
YfcTrain_hat = exp([ones(length(TrainSELECT_LO),1), Xfc(TrainSELECT_LO,KIN)]*BLO([1 KIN+1]));
R2r.LO.KIN(n,cv) = compute_pseudo_R2(Yfc(TrainSELECT_LO,:), YfcTrain_hat);
YfcTest_hat = exp([ones(length(TestSELECT_LO),1), Xfc(TestSELECT_LO,KIN)]*BLO([1 KIN+1]));
R2t.LO.KIN(n,cv) = compute_pseudo_R2(Yfc(TestSELECT_LO,:), YfcTest_hat);

YfcTrain_hat = exp([ones(length(TrainSELECT_HI),1), Xfc(TrainSELECT_HI,KIN)]*BHI([1 KIN+1]));
R2r.HI.KIN(n,cv) = compute_pseudo_R2(Yfc(TrainSELECT_HI,:), YfcTrain_hat);
YfcTest_hat = exp([ones(length(TestSELECT_HI),1), Xfc(TestSELECT_HI,KIN)]*BHI([1 KIN+1]));
R2t.HI.KIN(n,cv) = compute_pseudo_R2(Yfc(TestSELECT_HI,:), YfcTest_hat);

% Self terms only
YfcTrain_hat = exp([ones(length(TrainSELECT_LO),1), Xfc(TrainSELECT_LO,SELF)]*BLO([1 SELF+1]));
R2r.LO.SELF(n,cv) = compute_pseudo_R2(Yfc(TrainSELECT_LO,:), YfcTrain_hat);
YfcTest_hat = exp([ones(length(TestSELECT_LO),1), Xfc(TestSELECT_LO,SELF)]*BLO([1 SELF+1]));
R2t.LO.SELF(n,cv) = compute_pseudo_R2(Yfc(TestSELECT_LO,:), YfcTest_hat);

YfcTrain_hat = exp([ones(length(TrainSELECT_HI),1), Xfc(TrainSELECT_HI,SELF)]*BHI([1 SELF+1]));
R2r.HI.SELF(n,cv) = compute_pseudo_R2(Yfc(TrainSELECT_HI,:), YfcTrain_hat);
YfcTest_hat = exp([ones(length(TestSELECT_HI),1), Xfc(TestSELECT_HI,SELF)]*BHI([1 SELF+1]));
R2t.HI.SELF(n,cv) = compute_pseudo_R2(Yfc(TestSELECT_HI,:), YfcTest_hat);

% Kinematics and self terms
YfcTrain_hat = exp([ones(length(TrainSELECT_LO),1), Xfc(TrainSELECT_LO,KINSELF)]*BLO([1 KINSELF+1]));
R2r.LO.KINSELF(n,cv) = compute_pseudo_R2(Yfc(TrainSELECT_LO,:), YfcTrain_hat);
YfcTest_hat = exp([ones(length(TestSELECT_LO),1), Xfc(TestSELECT_LO,KINSELF)]*BLO([1 KINSELF+1]));
R2t.LO.KINSELF(n,cv) = compute_pseudo_R2(Yfc(TestSELECT_LO,:), YfcTest_hat);

YfcTrain_hat = exp([ones(length(TrainSELECT_HI),1), Xfc(TrainSELECT_HI,KINSELF)]*BHI([1 KINSELF+1]));
R2r.HI.KINSELF(n,cv) = compute_pseudo_R2(Yfc(TrainSELECT_HI,:), YfcTrain_hat);
YfcTest_hat = exp([ones(length(TestSELECT_HI),1), Xfc(TestSELECT_HI,KINSELF)]*BHI([1 KINSELF+1]));
R2t.HI.KINSELF(n,cv) = compute_pseudo_R2(Yfc(TestSELECT_HI,:), YfcTest_hat);

% Self and cross terms (no kinematics)
YfcTrain_hat = exp([ones(length(TrainSELECT_LO),1), Xfc(TrainSELECT_LO,SELFCROSS)]*BLO([1 SELFCROSS+1]));
R2r.LO.SELFCROSS(n,cv) = compute_pseudo_R2(Yfc(TrainSELECT_LO,:), YfcTrain_hat);
YfcTest_hat = exp([ones(length(TestSELECT_LO),1), Xfc(TestSELECT_LO,SELFCROSS)]*BLO([1 SELFCROSS+1]));
R2t.LO.SELFCROSS(n,cv) = compute_pseudo_R2(Yfc(TestSELECT_LO,:), YfcTest_hat);

YfcTrain_hat = exp([ones(length(TrainSELECT_HI),1), Xfc(TrainSELECT_HI,SELFCROSS)]*BHI([1 SELFCROSS+1]));
R2r.HI.SELFCROSS(n,cv) = compute_pseudo_R2(Yfc(TrainSELECT_HI,:), YfcTrain_hat);
YfcTest_hat = exp([ones(length(TestSELECT_HI),1), Xfc(TestSELECT_HI,SELFCROSS)]*BHI([1 SELFCROSS+1]));
R2t.HI.SELFCROSS(n,cv) = compute_pseudo_R2(Yfc(TestSELECT_HI,:), YfcTest_hat);

% All covariates
YfcTrain_hat = exp([ones(length(TrainSELECT_LO),1), Xfc(TrainSELECT_LO,ALL)]*BLO([1 ALL+1]));
R2r.LO.ALL(n,cv) = compute_pseudo_R2(Yfc(TrainSELECT_LO,:), YfcTrain_hat);
YfcTest_hat = exp([ones(length(TestSELECT_LO),1), Xfc(TestSELECT_LO,ALL)]*BLO([1 ALL+1]));
R2t.LO.ALL(n,cv) = compute_pseudo_R2(Yfc(TestSELECT_LO,:), YfcTest_hat);

YfcTrain_hat = exp([ones(length(TrainSELECT_HI),1), Xfc(TrainSELECT_HI,ALL)]*BHI([1 ALL+1]));
R2r.HI.ALL(n,cv) = compute_pseudo_R2(Yfc(TrainSELECT_HI,:), YfcTrain_hat);
YfcTest_hat = exp([ones(length(TestSELECT_HI),1), Xfc(TestSELECT_HI,ALL)]*BHI([1 ALL+1]));
R2t.HI.ALL(n,cv) = compute_pseudo_R2(Yfc(TestSELECT_HI,:), YfcTest_hat);

