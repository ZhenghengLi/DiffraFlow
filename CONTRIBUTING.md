# Contributing guidelines

## Coding style

C++ and Java coding style should follow the way of  

```bash
astyle -A2 -s4 -N -p -H -k1 -j -xU -O *.hh,*.cc
```

or  

```bash
astyle --style=attach --indent=spaces=4 --indent-namespaces --pad-oper --pad-header \
    --align-pointer=type --add-braces --indent-after-parens --keep-one-line-blocks *.hh,*.cc
```

for details of the rules, see the document of [astyle](http://astyle.sourceforge.net/astyle.html).

## Naming style

1. File name:
   * One file one class, and use the class name as file name.
   * header file ends with .hh and source file ends with .cc.
