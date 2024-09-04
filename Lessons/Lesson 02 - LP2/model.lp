\* Source Pyomo model name=TannerCo *\

max 
obj:
+3 x(toys)
-2.5 x(subs)

s.t.

c_u_cts(1)_:
+2 x(toys)
-1 x(subs)
<= 3001

c_u_cts(2)_:
+1 x(toys)
-1 x(subs)
<= 1000

c_u_cts(3)_:
+1 x(toys)
<= 2500

bounds
   0 <= x(toys) <= +inf
   0 <= x(subs) <= +inf
end
