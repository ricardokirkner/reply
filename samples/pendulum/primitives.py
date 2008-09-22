#!/usr/bin/env python
import math

import pyglet

def circle(position, radius, color):
    circumference = 2*math.pi*radius
    xc, yc = position
    step_size = 5
    steps = max(4, int(circumference/step_size))

    adelta = 2*math.pi/steps
    points = [xc,yc,xc+radius,yc]
    for step in range(1,steps+1):
        x = radius*math.cos(step*adelta)
        y = radius*math.sin(step*adelta)
        points += [xc+x,yc+y]

    num_points = steps+2
    vertex_list = pyglet.graphics.vertex_list(num_points,
        ('v2f', points),
        ('c4B', list(color)*num_points)
        )
    return vertex_list

def rectangle(x1, y1, x2, y2, color):
    return pyglet.graphics.vertex_list(4,
            ('v2f', [x1, y1, x2, y1, x2, y2, x1, y2]),
            ('c4B', color*4)
        )

def up_triange(x,y, h, w, color):
    return pyglet.graphics.vertex_list(3,
            ('v2f', [x, y, x-w/2, y+h, x+w/2, y+h]),
            ('c4B', color*3)
        )

def down_triange(x,y, h, w, color):
    return pyglet.graphics.vertex_list(3,
            ('v2f', [x, y, x-w/2, y-h, x+w/2, y-h]),
            ('c4B', color*3)
        )