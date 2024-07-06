def foo(a, b, c, d, e, f, g):
    print(a, b, c, d, e, f, g)

# call foo on numbers 0 through 6!
foo(*[i for i in range(7)])

def goo(*args):
    print(args)

# call goo on whatever you want!
goo(*(0, 1, 2))

def koo(name1, name2):
    print(name1, name2)

koo("julian", "linda")
args = {
    "name1": "julian",
    "name2": "linda",
}
koo(args["name1"], args["name2"])
koo(**args)

args = {
    "addr": "stanford avenue",
    "name1": "julian",
    "name2": "linda",
    "name3": "doggo",
}

def woo(addr="stanford ave", **kwargs):
    name1 = kwargs["name1"]
    name2 = kwargs["name2"]
    name3 = kwargs["name3"]
    print(addr, name1, name2, name3)

woo(**args)

