\* Source Pyomo model name=TannerCo *\

max 
obj:
-2.5 x(subs)
+3 x(toys)

s.t.

c_u_cts(1)_:
-1 x(subs)
+2 x(toys)
<= 3001

c_u_cts(2)_:
-1 x(subs)
+1 x(toys)
<= 1000

c_u_cts(3)_:
+1 x(toys)
<= 2500

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x(subs) <= +inf
   0 <= x(toys) <= +inf
end
