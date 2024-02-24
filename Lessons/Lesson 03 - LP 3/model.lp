\* Source Pyomo model name=unknown *\

min 
total_cost:
+464 transp(can1_ware1)
+10000 transp(can1_ware2)
+654 transp(can1_ware3)
+867 transp(can1_ware4)
+10000 transp(can2_ware1)
+416 transp(can2_ware2)
+690 transp(can2_ware3)
+791 transp(can2_ware4)
+995 transp(can3_ware1)
+682 transp(can3_ware2)
+10000 transp(can3_ware3)
+685 transp(can3_ware4)

s.t.

c_e_supply_ct(1)_:
+1 transp(can1_ware1)
+1 transp(can1_ware2)
+1 transp(can1_ware3)
+1 transp(can1_ware4)
= 75

c_e_supply_ct(2)_:
+1 transp(can2_ware1)
+1 transp(can2_ware2)
+1 transp(can2_ware3)
+1 transp(can2_ware4)
= 125

c_e_supply_ct(3)_:
+1 transp(can3_ware1)
+1 transp(can3_ware2)
+1 transp(can3_ware3)
+1 transp(can3_ware4)
= 100

c_e_demand_ct(1)_:
+1 transp(can1_ware1)
+1 transp(can2_ware1)
+1 transp(can3_ware1)
= 80

c_e_demand_ct(2)_:
+1 transp(can1_ware2)
+1 transp(can2_ware2)
+1 transp(can3_ware2)
= 65

c_e_demand_ct(3)_:
+1 transp(can1_ware3)
+1 transp(can2_ware3)
+1 transp(can3_ware3)
= 70

c_e_demand_ct(4)_:
+1 transp(can1_ware4)
+1 transp(can2_ware4)
+1 transp(can3_ware4)
= 85

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= transp(can1_ware1) <= +inf
   0 <= transp(can1_ware2) <= +inf
   0 <= transp(can1_ware3) <= +inf
   0 <= transp(can1_ware4) <= +inf
   0 <= transp(can2_ware1) <= +inf
   0 <= transp(can2_ware2) <= +inf
   0 <= transp(can2_ware3) <= +inf
   0 <= transp(can2_ware4) <= +inf
   0 <= transp(can3_ware1) <= +inf
   0 <= transp(can3_ware2) <= +inf
   0 <= transp(can3_ware3) <= +inf
   0 <= transp(can3_ware4) <= +inf
end
