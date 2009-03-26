
class Parameter(object):
    pass

class Number(Parameter):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        
class Integer(Number):
    def __str__(self):
        return "(%i %i)" % (self.min, self.max)


class Double(Number):
    def __str__(self):
        return "(%f %f)" % (self.min, self.max)


class Char(Parameter):
    pass

class ListOf(Parameter):
    def __init__(self, num, type):
        self.num = num
        self.type = type
        

class Space(object):
    def __init__(self, spec):
        self.spec = spec
        
    def __str__(self):
        doubles = []
        chars = []
        ints = []
        
        for k, v in self.spec.items():
            if isinstance(v, ListOf):
                t = v.type
            else:
                t = v
                
            if isinstance(t, Double) :
                doubles.append((k,v))
            elif isinstance(t, Integer):
                ints.append((k,v))
            elif isinstance(t, Char):
                chars.append((k,v))

        doubles.sort()
        ints.sort()
        charcount = sum([ x.num if isinstance(x, ListOf) else 1 for x in chars])
        result = []
        if ints:
            result.append("INTS %s" %  " ".join([ str(y) for (x,y) in ints]))
        if doubles:
            result.append("DOUBLES %s" %
                          " ".join([ str(y) for (x,y) in doubles]))
        if chars:
            result.append("CHARCOUNT %s" % charcount)
        return " ".join(result)