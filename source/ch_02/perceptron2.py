def perceptron(b, w1, w2, x1, x2):
    v = b + w1 * x1 + w2 * x2
    if v <= 0:
        return 0
    if v > 0:
        return 1


def AND(x1, x2):
    return perceptron(-0.7, 0.5, 0.5, x1, x2)


def NAND(x1, x2):
    return perceptron(0.7, -0.5, -0.5, x1, x2)


def OR(x1, x2):
    return perceptron(-0.2, 0.5, 0.5, x1, x2)


def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))


def logic_test(fun):
    print("     |  0  |  1  |")
    print("-----------------")
    print("  0  |  " + str(fun(0, 0)) + "  |  " + str(fun(1, 0)) + "  |")
    print("-----------------")
    print("  1  |  " + str(fun(0, 1)) + "  |  " + str(fun(1, 1)) + "  |")
    print("-----------------")
    print("")


print("=== AND test ===")
logic_test(AND)
print("=== NAND test ===")
logic_test(NAND)
print("=== OR test ===")
logic_test(OR)
print("=== XOR test ===")
logic_test(XOR)
