MAX_STEPS = 10


def newton(f, xi):
    df = f.diff(x)
    ddf = df.diff(x)
    visited = []
    for _ in range(MAX_STEPS):
        xi = xi - df(x=xi)/ddf(x=xi)
        yi = f(x=xi)
        visited.append((xi, yi))
    return visited


def secant(f, x1, x0):
    df = f.diff(x)
    ddf = df.diff(x)
    visited = []

    for i in range(MAX_STEPS):
        x1 = x1 - ((x1 - x0)/(df(x=x1) - df(x=x0))) * df(x=x1)
        y1 = f(x=x1)
        visited.append((x1, y1))

    return visited


f = x ^ 2 + x ^ 4
plt = plot(f, (x, -4, 1))

for visit in newton(f, -3):
    plt += point(visit, color="red", size=25)

for visit in secant(f, -3, -4):
    plt += point(visit, color="green", size=25)

show(plt)


f = x ^ (4/3)
#plt = plot(f, (x, -4, 1))

#for visit in newton(f, -3):
#   plt += point(visit, color="red", size=25)
#show(plt)

df = f.diff(x)
ddf = df.diff(x)

print(df, ddf)

print(df/ddf)
