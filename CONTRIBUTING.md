# Contributing guidelines

## Coding style

c++ coding style should follow the way of  

```bash
astyle -A2 -s4 -N -p -H -k1 -j *.hh,*.cc
```

or  

```bash
astyle --style=attach --indent=spaces=4 --indent-namespaces --pad-oper --pad-header --align-pointer=type --add-braces *.hh,*.cc
```

for more details of the rules, see the document of [astyle](http://astyle.sourceforge.net/astyle.html).
