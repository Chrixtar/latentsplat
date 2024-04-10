from fractions import Fraction


def is_integer(f: Fraction) -> bool:
    return f.denominator == 1


def get_integer(f: Fraction) -> int:
    assert f.denominator == 1, "Fraction is not integer"
    return f.numerator

def get_inv(f: Fraction) -> Fraction:
    return Fraction(f.denominator, f.numerator)
