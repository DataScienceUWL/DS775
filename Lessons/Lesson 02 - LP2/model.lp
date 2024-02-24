\* Source Pyomo model name=unknown *\

max 
profit:
+3 weekly_prod(Doors)
+5 weekly_prod(Windows)

s.t.

c_u_capacity(1)_:
+1 weekly_prod(Doors)
<= 4

c_u_capacity(2)_:
+2 weekly_prod(Windows)
<= 12

c_u_capacity(3)_:
+3 weekly_prod(Doors)
+2 weekly_prod(Windows)
<= 18

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= weekly_prod(Doors) <= +inf
   0 <= weekly_prod(Windows) <= +inf
end
