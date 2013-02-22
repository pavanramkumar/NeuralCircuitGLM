function R2 = compute_pseudo_R2(Y, Yhat)

Modelr = sum(Y.*log(eps+Yhat) - Yhat);
Intercr = sum(Y*log(eps+mean(Y)) - mean(Y));
Sat_r = sum(Y.*log(eps+Y) - Y);


R2 = (1-(Sat_r-Modelr)./(Sat_r-Intercr))';