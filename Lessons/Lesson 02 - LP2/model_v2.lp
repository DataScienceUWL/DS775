\* Source Pyomo model name=Wyndor *\

max 
profit:
+3 x(drs)
+5 x(wdw)

s.t.

c_u_Constraint1_:
+1 x(drs)
<= 4

c_u_Constraint2_:
+2 x(wdw)
<= 12

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x(drs) <= +inf
   0 <= x(wdw) <= +inf
end
