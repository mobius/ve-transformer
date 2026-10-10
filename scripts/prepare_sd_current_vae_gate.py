"""Enable independently validated four-thread output-row dispatch."""
ORIGINAL = "&& s0==1 && s1==1 && p0==1 && p1==1 && d0==1 && d1==1 && nth==8;"
REPLACEMENT = "&& s0==1 && s1==1 && p0==1 && p1==1 && d0==1 && d1==1 && (nth==8 || nth==4);"
def instrument(text):
    if text.count(ORIGINAL) != 1:
        raise RuntimeError("one frozen output-row thread gate required")
    return text.replace(ORIGINAL, REPLACEMENT)
