# Pair smoke: replicate totals after depth matching

Eval set `/project/def-maxwl/mforooz/ONION/CO-D2/eval/eic_v1_2/evalset.json`, 256 regions of 16384 bins. A pair outside 1 +- 0.05 gives an Undefined headline-7 ratio (`metrics.variance_ratio`).

| a | b | depth a | depth b | sum(a)/sum(b) | outside |
|---|---|---|---|---|---|
| ENCSR000ANA.pooled.uniform.base | ENCSR271TFS.pooled.uniform.base | 23278807 | 21976416 | 1.0076 | no |
| ENCSR000DTH.pooled.uniform.base | ENCSR000DTE.pooled.uniform.base | 19157013 | 33229733 | 0.9891 | no |
| ENCSR000DWB.pooled.uniform.base | ENCSR000AKR.pooled.uniform.base | 25916411 | 69785955 | 0.9947 | no |
| ENCSR000EMD.pooled.uniform.base | ENCSR009KWQ.pooled.uniform.base | 66283144 | 283455250 | 0.7563 | yes |
| ENCSR000EMI.pooled.uniform.base | ENCSR030HFV.pooled.uniform.base | 28975855 | 270621224 | 0.6709 | yes |
| ENCSR000ENE.pooled.uniform.base | ENCSR006DKF.pooled.uniform.base | 56872150 | 193111456 | 0.6791 | yes |
| ENCSR000EQL.pooled.uniform.base | ENCSR218FSP.pooled.uniform.base | 64044979 | 297401722 | 0.7328 | yes |
| ENCSR000EVZ.pooled.uniform.base | ENCSR000AKV.pooled.uniform.base | 38841530 | 20851599 | 0.8403 | yes |
| ENCSR000EWB.pooled.uniform.base | ENCSR000AKQ.pooled.uniform.base | 36577542 | 83480563 | 0.9541 | no |
| ENCSR000EWC.pooled.uniform.base | ENCSR000AKS.pooled.uniform.base | 26393176 | 69949131 | 1.0405 | no |
| ENCSR000XAA.pooled.uniform.base | ENCSR803MVC.pooled.uniform.base | 45950494 | 50560652 | 0.9875 | no |
| ENCSR035DUS.pooled.uniform.base | ENCSR564GWG.pooled.uniform.base | 51804720 | 19526730 | 1.0009 | no |
| ENCSR038OIN.pooled.uniform.base | ENCSR005YZH.pooled.uniform.base | 52516087 | 15877943 | 1.0276 | no |
| ENCSR065HOR.pooled.uniform.base | ENCSR196LEI.pooled.native.base | 47374888 | 55390578 | 0.9835 | no |
| ENCSR080PZL.pooled.uniform.base | ENCSR060BFU.pooled.uniform.base | 43584083 | 32188277 | 0.9955 | no |
| ENCSR113MBR.pooled.uniform.base | ENCSR548KIL.pooled.uniform.base | 133722514 | 63354910 | 1.0317 | no |
| ENCSR177QFY.pooled.uniform.base | ENCSR405ESP.pooled.uniform.base | 62900981 | 49489740 | 1.4604 | yes |
| ENCSR208WDY.pooled.uniform.base | ENCSR701FGA.pooled.uniform.base | 53675549 | 56134064 | 1.0028 | no |
| ENCSR223APP.pooled.native.base | ENCSR077FZT.pooled.native.base | 19154525 | 25919413 | 0.9979 | no |
| ENCSR234YIU.pooled.uniform.base | ENCSR926NMC.pooled.uniform.base | 22286014 | 30674926 | 0.6025 | yes |
| ENCSR249INE.pooled.uniform.base | ENCSR189JIH.pooled.uniform.base | 78318855 | 48939572 | 1.0012 | no |
| ENCSR254NLC.pooled.uniform.base | ENCSR169BGF.pooled.uniform.base | 135364172 | 25995841 | 0.9936 | no |
| ENCSR256ESY.pooled.uniform.base | ENCSR238WMO.pooled.uniform.base | 125918863 | 60697054 | 1.0270 | no |
| ENCSR258UUX.pooled.uniform.base | ENCSR647HAQ.pooled.uniform.base | 45749727 | 30739136 | 1.0000 | no |
| ENCSR301HRV.pooled.uniform.base | ENCSR000APY.pooled.uniform.base | 40316294 | 54124841 | 1.0319 | no |
| ENCSR322MEI.pooled.uniform.base | ENCSR000ANC.pooled.uniform.base | 22185288 | 25307165 | 0.9953 | no |
| ENCSR328UMC.pooled.uniform.base | ENCSR000EMK.pooled.uniform.base | 27474306 | 31694991 | 1.0122 | no |
| ENCSR333VFQ.pooled.native.base | ENCSR239SNR.pooled.native.base | 19105117 | 25012137 | 1.0050 | no |
| ENCSR350YQC.pooled.uniform.base | ENCSR172CVZ.pooled.uniform.base | 28649455 | 21599275 | 1.0011 | no |
| ENCSR391WDE.pooled.uniform.base | ENCSR800IIW.pooled.uniform.base | 131983603 | 46392519 | 1.0258 | no |
| ENCSR406PCK.pooled.uniform.base | ENCSR379BVL.pooled.uniform.base | 86426566 | 66979692 | 0.9978 | no |
| ENCSR408ZEW.pooled.uniform.base | ENCSR992VZG.pooled.uniform.base | 51932082 | 14982864 | 1.6748 | yes |
| ENCSR438QZN.pooled.uniform.base | ENCSR111WGZ.pooled.uniform.base | 61735211 | 26166780 | 1.7040 | yes |
| ENCSR444TMC.pooled.uniform.base | ENCSR004ZWW.pooled.uniform.base | 35821330 | 25153544 | 0.9958 | no |
| ENCSR477RTP.pooled.uniform.base | ENCSR000EKG.pooled.uniform.base | 190276463 | 15201 | 0.8514 | yes |
| ENCSR500ZGF.pooled.uniform.base | ENCSR434MDA.pooled.uniform.base | 94306766 | 18955288 | 1.3907 | yes |
| ENCSR501JET.pooled.uniform.base | ENCSR004AKD.pooled.native.base | 22775618 | 50941670 | 1.0008 | no |
| ENCSR564VIX.pooled.uniform.base | ENCSR000ARA.pooled.uniform.base | 52752189 | 43758129 | 1.0372 | no |
| ENCSR571IIS.pooled.uniform.base | ENCSR000APX.pooled.uniform.base | 8254862 | 71936304 | 1.2171 | yes |
| ENCSR579TJT.pooled.uniform.base | ENCSR553OEU.pooled.uniform.base | 27911697 | 58037544 | 0.9983 | no |
| ENCSR604THF.pooled.uniform.base | ENCSR433FJP.pooled.uniform.base | 16850875 | 30638444 | 1.0065 | no |
| ENCSR611DJQ.pooled.uniform.base | ENCSR619EZG.pooled.uniform.base | 46461458 | 61341067 | 0.9984 | no |
| ENCSR668LDD.pooled.uniform.base | ENCSR000EWA.pooled.uniform.base | 57932223 | 30127689 | 1.6290 | yes |
| ENCSR694LBI.pooled.uniform.base | ENCSR550XZG.pooled.uniform.base | 98399149 | 50054036 | 1.0377 | no |
| ENCSR701KJD.pooled.uniform.base | ENCSR006GPM.pooled.native.base | 28118070 | 80592354 | 1.0064 | no |
| ENCSR702RZM.pooled.native.base | ENCSR012KUR.pooled.uniform.base | 35166567 | 19464642 | 1.0136 | no |
| ENCSR718BTD.pooled.uniform.base | ENCSR346FVK.pooled.uniform.base | 41623909 | 28848116 | 0.9908 | no |
| ENCSR719QMW.pooled.uniform.base | ENCSR448TUF.pooled.uniform.base | 30131050 | 22704802 | 1.0092 | no |
| ENCSR728SZE.pooled.uniform.base | ENCSR000AMH.pooled.uniform.base | 46681250 | 26430312 | 0.9913 | no |
| ENCSR734MEM.pooled.uniform.base | ENCSR491VZZ.pooled.uniform.base | 23288647 | 20570784 | 1.0043 | no |
| ENCSR738SXD.pooled.uniform.base | ENCSR453MUW.pooled.uniform.base | 25605144 | 37740460 | 0.9943 | no |
| ENCSR746CUY.pooled.uniform.base | ENCSR439EHQ.pooled.native.base | 54321544 | 44574417 | 0.9896 | no |
| ENCSR748RBT.pooled.uniform.base | ENCSR153NDQ.pooled.uniform.base | 65778819 | 99309281 | 1.0019 | no |
| ENCSR763IDK.pooled.uniform.base | ENCSR841AJO.pooled.uniform.base | 103982574 | 79839720 | 1.0069 | no |
| ENCSR794OFW.pooled.uniform.base | ENCSR000EJN.pooled.uniform.base | 62283805 | 47125017 | 1.3826 | yes |
| ENCSR824UNY.pooled.uniform.base | ENCSR555QHZ.pooled.native.base | 52731395 | 58847485 | 0.9894 | no |
| ENCSR832JVP.pooled.uniform.base | ENCSR262VXI.pooled.native.base | 129767641 | 48982937 | 0.9770 | no |
| ENCSR851EBF.pooled.uniform.base | ENCSR117PYB.pooled.uniform.base | 37024952 | 49865190 | 0.9854 | no |
| ENCSR854OXF.pooled.uniform.base | ENCSR863BVD.pooled.uniform.base | 26753940 | 33235324 | 0.9921 | no |
| ENCSR857XRG.pooled.uniform.base | ENCSR942XCE.pooled.uniform.base | 75668050 | 16534247 | 1.6697 | yes |
| ENCSR869GBP.pooled.uniform.base | ENCSR642DSR.pooled.uniform.base | 62500706 | 67802237 | 0.9866 | no |
| ENCSR880SUY.pooled.uniform.base | ENCSR000ANP.pooled.uniform.base | 24050873 | 27588660 | 0.9330 | yes |
| ENCSR883AQJ.pooled.uniform.base | ENCSR395USV.pooled.uniform.base | 26095123 | 14547836 | 0.6477 | yes |
| ENCSR895YWA.pooled.uniform.base | ENCSR250VGX.pooled.uniform.base | 26029251 | 22285028 | 1.0079 | no |
| ENCSR901SIL.pooled.uniform.base | ENCSR487BEW.pooled.uniform.base | 98522194 | 17812308 | 1.6779 | yes |
| ENCSR902EEI.pooled.uniform.base | ENCSR115PQP.pooled.uniform.base | 18545821 | 28254966 | 0.9984 | no |
| ENCSR925LJZ.pooled.uniform.base | ENCSR496DCY.pooled.uniform.base | 20969469 | 17119341 | 0.6206 | yes |
| ENCSR928HYM.pooled.uniform.base | ENCSR186OBR.pooled.uniform.base | 18775546 | 19027450 | 1.0072 | no |
| ENCSR952GVX.pooled.uniform.base | ENCSR441UHO.pooled.uniform.base | 22826417 | 24263040 | 1.0025 | no |
| ENCSR956CTX.pooled.uniform.base | ENCSR354XWM.pooled.uniform.base | 134926262 | 47031717 | 1.0138 | no |
| ENCSR988JLN.pooled.uniform.base | ENCSR503YOF.pooled.uniform.base | 49634839 | 17228611 | 1.3229 | yes |

20 of 71 pairs are outside the tolerance.
