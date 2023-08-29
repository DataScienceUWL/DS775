\* Source Pyomo model name=Generic *\

max 
obj:
+5 x(x1)
+4 x(x2)
-1 x(x3)
+3 x(x4)

s.t.

c_u_cts(1)_:
+3 x(x1)
+2 x(x2)
-3 x(x3)
+1 x(x4)
<= 24

c_u_cts(2)_:
+3 x(x1)
+3 x(x2)
+1 x(x3)
+3 x(x4)
<= 36

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x(x1) <= +inf
   0 <= x(x2) <= +inf
   0 <= x(x3) <= +inf
   0 <= x(x4) <= +inf
end
