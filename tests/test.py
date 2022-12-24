from math import tan, atan, sqrt, pi, degrees, radians

def calc_alpha(h,l):
    return atan(2*h/l)

def calc_h1(h,l):
    return sqrt(h**2 + (l/2)**2)

def calc_b(h, alpha):
    return h / tan(alpha)

def calc_d(l,b):
    return sqrt((l/2)**2 + b**2)

def calc_c(h, d):
    return sqrt(h**2 + d**2)

def calc_main_aire(h,l,long):
    h1 = calc_h1(h,l)
    alpha = calc_alpha(h,l)
    b = calc_b(h1, alpha)
    print("alpha: ", degrees(alpha))

    return h1 * (long-b)

def calc_side_aire(h,l):
    alpha = calc_alpha(h,l)
    b = calc_b(h, alpha)
    d = calc_d(l,b)
    c = calc_c(h, d)

    return 1/2 * l * sqrt(c**2 - (l/2)**2)

h=3
l=5
long=10

main_aire = calc_main_aire(h,l,long)
side_aire = calc_side_aire(h,l)
result = 2 * (main_aire + side_aire)
print("main_aire: ", main_aire)
print("side_aire: ", side_aire)
print("result", result)