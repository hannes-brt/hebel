ceil_div = lambda x, y: int((x + y - 1) / y)
div_up = lambda x, y: y * ceil_div(x, y)