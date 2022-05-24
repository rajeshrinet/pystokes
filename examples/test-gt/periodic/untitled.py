def elementG_import(G, *args): ## extract an element from the matrix G for parallelisation
    return sympy.powdenest(sympy.simplify(G(*args)).expand(power_exp=True), force=True)

